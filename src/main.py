"""AeroTrace-RCA command-line entry point.

Usage::

    python -m src.main data/raw/flight.csv --report            # full RCA (default)
    python -m src.main data/raw/flight.csv --analyze           # detection + triage only
    python -m src.main --test                                  # offline self-test on synthetic data
    python -m src.main flight.csv --format both --offline -v   # Markdown + JSON, no LLMs, debug logs
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import Settings, configure_logging, get_settings
from src.agents.supervisor import PipelineError, PipelineResult, Supervisor
from src.ingestion import SUPPORTED_SUFFIXES, write_sample_files
from src.ingestion.csv_parser import INJECTED_ANOMALIES
from src.models.telemetry_schema import Severity
from src.utils.logger import get_logger

console = Console()
logger = get_logger("main")

_SEVERITY_STYLE = {Severity.CRITICAL: "bold red", Severity.WARNING: "yellow", Severity.INFO: "cyan"}


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="aerotrace",
        description="AeroTrace-RCA - multi-agent rocket telemetry anomaly detection and root cause analysis.",
    )
    parser.add_argument(
        "input", nargs="?", type=Path,
        help=f"Telemetry file ({', '.join(sorted(SUPPORTED_SUFFIXES))}). Defaults to the newest file in data/raw.",
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--analyze", dest="mode", action="store_const", const="analyze", help="Detect and triage anomalies only.")
    modes.add_argument("--report", dest="mode", action="store_const", const="report", help="Full RCA with hypotheses and report (default).")
    modes.add_argument("--test", dest="mode", action="store_const", const="test", help="Run an offline end-to-end self-test.")
    parser.set_defaults(mode="report")
    parser.add_argument("--format", choices=("markdown", "json", "both"), default="markdown", help="Report output format.")
    parser.add_argument("--offline", action="store_true", help="Disable all LLM calls (rule-based agents only).")
    parser.add_argument("--mission-context", default=None, help="Free-text mission/vehicle description for the LLM agents.")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings and errors only.")
    return parser


def resolve_input(path: Path | None, settings: Settings) -> Path:
    """Return the input file, defaulting to the newest raw file or freshly generated sample data.

    Raises:
        FileNotFoundError: If an explicit path does not exist.
    """
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path
    candidates = sorted(
        (p for p in settings.raw_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        console.print(f"[dim]No input given - using newest raw file: {candidates[0].name}[/dim]")
        return candidates[0]
    console.print("[yellow]No telemetry in data/raw - generating synthetic sample flight data.[/yellow]")
    return write_sample_files(settings.raw_dir)["nasa_long"]


# ---------------------------------------------------------------------- #
# Rendering
# ---------------------------------------------------------------------- #
def display_result(result: PipelineResult, show_decisions: bool = False) -> None:
    """Print a pipeline result with Rich formatting."""
    if result.analysis:
        a = result.analysis
        table = Table(title=f"Dataset: {a.source_file}", title_justify="left", header_style="bold")
        for column in ("Sensor", "Unit", "Samples", "Rate Hz", "Min", "Max", "Imputed"):
            table.add_column(column, justify="left" if column in {"Sensor", "Unit"} else "right")
        for s in a.sensors:
            table.add_row(s.sensor_id, s.unit, str(s.samples), f"{s.sample_rate_hz:g}", f"{s.minimum:.4g}", f"{s.maximum:.4g}", str(s.imputed_samples))
        console.print(table)
        for note in a.data_quality_notes:
            console.print(f"  [dim]• {note}[/dim]")

    table = Table(title="Anomalies", title_justify="left", header_style="bold")
    for column in ("Anomaly", "Dur s", "Score σ", "Category", "Phase"):
        table.add_column(column)
    for e in result.anomalies:
        label = f"[{_SEVERITY_STYLE[e.severity]}]{e.label}[/]" + (" [dim](expected)[/dim]" if e.expected else "")
        table.add_row(
            label, f"{e.duration_s:.1f}", f"{e.score:.1f}", str(e.classification.get("category", "-")), e.flight_phase or "-",
        )
    console.print(table if result.anomalies else "[green]No anomalies detected.[/green]")

    if result.hypotheses:
        table = Table(title="Root-cause hypotheses", title_justify="left", header_style="bold")
        for column in ("#", "Hypothesis", "Confidence", "Subsystem", "Anomalies", "Source"):
            table.add_column(column)
        for i, h in enumerate(result.hypotheses, 1):
            table.add_row(str(i), h.title, f"{h.confidence:.0%}", h.subsystem, str(len(h.related_anomalies)), h.source)
        console.print(table)

    if result.report:
        console.print(Panel(result.report.summary, title="Executive summary", border_style="blue"))
        console.print(Panel(result.report.conclusion, title="Conclusion", border_style="green"))

    if show_decisions:
        table = Table(title="Supervisor decision log", title_justify="left", header_style="bold")
        for column in ("Time", "Agent", "Decision", "Detail"):
            table.add_column(column)
        for d in result.decisions:
            table.add_row(d.timestamp[11:23], d.agent, d.decision, d.detail)
        console.print(table)

    lines = [f"[bold]Run:[/bold] {result.run_id}   [bold]Duration:[/bold] {result.duration_s:.1f}s   "
             f"[bold]LLM providers available:[/bold] {', '.join(result.providers) or 'none (rule-based)'}"]
    lines += [f"[bold]{kind}:[/bold] {path}" for kind, path in result.output_paths.items()]
    lines += [f"[red]error:[/red] {err}" for err in result.errors]
    console.print(Panel("\n".join(lines), title="Outputs", border_style="magenta"))


# ---------------------------------------------------------------------- #
# Self-test
# ---------------------------------------------------------------------- #
def run_self_test() -> int:
    """Run the full pipeline offline on synthetic data and verify the outputs.

    Returns:
        Process exit code (0 on success).
    """
    console.rule("[bold]AeroTrace-RCA self-test")
    checks: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory(prefix="aerotrace_test_", ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        settings = Settings(data_dir=tmp_path / "data", report_dir=tmp_path / "data" / "reports")
        settings.ensure_directories()
        samples = write_sample_files(settings.raw_dir)
        try:
            supervisor = Supervisor(settings=settings, use_llm=False)
            result = supervisor.run(samples["nasa_long"], mode="report", output_formats=("markdown", "json"))
        except PipelineError as exc:
            console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
            return 1

        checks.append(("Ingested 6 sensors x 3000 samples", len(result.readings) == 18000, f"{len(result.readings)} readings"))
        actionable = result.actionable_anomalies
        for sensor, met in INJECTED_ANOMALIES.items():
            hit = any(a.sensor_id == sensor and abs((a.mission_elapsed_s or 0) - met) <= 5 for a in actionable)
            checks.append((f"Detected injected anomaly on {sensor} @ T+{met:g}s", hit, ""))
        false_positive = [a for a in actionable if a.severity != Severity.INFO and not any(
            a.sensor_id == s and abs((a.mission_elapsed_s or 0) - m) <= 5 for s, m in INJECTED_ANOMALIES.items())]
        checks.append(("No WARNING/CRITICAL false positives", not false_positive, ", ".join(a.id for a in false_positive)))
        checks.append(("Planned transients marked expected", any(a.expected for a in result.anomalies), ""))
        glitch = next((a for a in actionable if a.sensor_id == "fuel_tank_pressure"), None)
        checks.append(("Fuel-pressure spike triaged as sensor fault",
                       glitch is not None and glitch.classification.get("category") == "sensor_fault", ""))
        checks.append(("Hypotheses generated", bool(result.hypotheses), f"{len(result.hypotheses)} hypotheses"))
        explained = {aid for h in result.hypotheses for aid in h.related_anomalies}
        unexplained = [a.label for a in actionable if a.id not in explained]
        checks.append(("Every actionable anomaly explained by a hypothesis", not unexplained, "; ".join(unexplained)))
        fuel = next((a for a in result.anomalies if a.sensor_id == "fuel_tank_pressure" and not a.expected), None)
        checks.append(("Sensor-fault classifier: fuel spike is sensor_fault",
                       fuel is not None and fuel.classification.get("fault_class") == "sensor_fault", ""))
        real = [a for a in actionable if a.sensor_id != "fuel_tank_pressure"]
        checks.append(("Sensor-fault classifier: other anomalies are real events",
                       all(a.classification.get("fault_class") == "real_event" for a in real), ""))
        for kind in ("report", "summary", "json", "decision_log"):
            path = result.output_paths.get(kind)
            checks.append((f"Wrote {kind}", bool(path and path.exists() and path.stat().st_size > 0), path.name if path else "missing"))
        checks.append(("Pipeline completed without step errors", not result.errors, "; ".join(result.errors)))

    table = Table(title="Self-test results", header_style="bold")
    table.add_column("Check")
    table.add_column("Result")
    table.add_column("Detail")
    for name, ok, detail in checks:
        table.add_row(name, "[green]PASS[/green]" if ok else "[red]FAIL[/red]", detail)
    console.print(table)
    failed = sum(not ok for _, ok, _ in checks)
    console.print(f"[bold {'red' if failed else 'green'}]{len(checks) - failed}/{len(checks)} checks passed[/]")
    return 1 if failed else 0


# ---------------------------------------------------------------------- #
def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    configure_logging("DEBUG" if args.verbose else "WARNING" if args.quiet else None)

    if args.mode == "test":
        return run_self_test()

    settings = get_settings()
    try:
        input_path = resolve_input(args.input, settings)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/bold red]")
        return 2

    formats = ("markdown", "json") if args.format == "both" else (args.format,)
    console.rule(f"[bold]AeroTrace-RCA - {args.mode} - {input_path.name}")
    try:
        supervisor = Supervisor(settings=settings, use_llm=False if args.offline else None)
        result = supervisor.run(input_path, mode=args.mode, output_formats=formats, mission_context=args.mission_context)
    except PipelineError as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        return 1
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/yellow]")
        return 130
    display_result(result, show_decisions=args.verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
