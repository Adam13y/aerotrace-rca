"""One-command AeroTrace-RCA run.

1. Checks whether Ollama is running.
2. Checks whether the configured Ollama model is installed (and whether Claude is configured).
3. Generates sample data if ``data/raw`` has no telemetry.
4. Runs the full analysis pipeline.
5. Prints a summary of everything generated.

Usage::

    python scripts/run_pipeline.py [input_file] [--format markdown|json|both] [--offline]
    python scripts/run_pipeline.py --kb-stats      # knowledge-base statistics only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import configure_logging, get_settings  # noqa: E402
from src.agents.supervisor import PipelineError, Supervisor  # noqa: E402
from src.ingestion import SUPPORTED_SUFFIXES, write_sample_files  # noqa: E402
from src.memory.historical_rag import HistoricalRAG, OllamaEmbedder, load_documents  # noqa: E402
from src.main import display_result  # noqa: E402
from src.utils.helpers import LLMUnavailableError, list_ollama_models, ollama_model_installed  # noqa: E402

console = Console()


def preflight() -> dict[str, bool]:
    """Check LLM providers and print their status.

    Returns:
        Flags ``ollama_running``, ``model_installed`` and ``claude_configured``.
    """
    settings = get_settings()
    status = {"ollama_running": False, "model_installed": False, "claude_configured": settings.claude_api_key is not None}
    table = Table(title="Pre-flight checks", header_style="bold")
    table.add_column("Check")
    table.add_column("Status")
    table.add_column("Detail")
    try:
        installed = list_ollama_models(settings.ollama_base_url)
        status["ollama_running"] = True
        status["model_installed"] = ollama_model_installed(settings.ollama_model, installed)
        table.add_row("Ollama server", "[green]running[/green]", settings.ollama_base_url)
        table.add_row(
            "Ollama model",
            "[green]installed[/green]" if status["model_installed"] else "[yellow]missing[/yellow]",
            settings.ollama_model if status["model_installed"] else f"run: ollama pull {settings.ollama_model}",
        )
    except LLMUnavailableError as exc:
        table.add_row("Ollama server", "[yellow]not reachable[/yellow]", str(exc))
        table.add_row("Ollama model", "[dim]skipped[/dim]", "")
    table.add_row(
        "Claude API",
        "[green]configured[/green]" if status["claude_configured"] else "[yellow]not configured[/yellow]",
        settings.claude_model if status["claude_configured"] else "set ANTHROPIC_API_KEY in .env",
    )
    console.print(table)
    if not (status["model_installed"] or status["claude_configured"]):
        console.print("[yellow]No LLM available - agents will use deterministic rule-based fallbacks.[/yellow]")
    elif not status["model_installed"]:
        console.print("[yellow]Local model unavailable - local agent work will be routed to Claude.[/yellow]")
    elif not status["claude_configured"]:
        console.print("[yellow]Claude not configured - supervisor reasoning will use the local model.[/yellow]")
    return status


def ensure_input(path: Path | None) -> Path:
    """Return the input file, generating sample data when ``data/raw`` is empty.

    Raises:
        FileNotFoundError: If an explicit path does not exist.
    """
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    raw_dir = get_settings().raw_dir
    existing = sorted(p for p in raw_dir.glob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES)
    preferred = [p for p in existing if p.name == "sample_flight_nasa_long.csv"]
    if preferred or existing:
        chosen = (preferred or existing)[0]
        console.print(f"Using existing telemetry: [bold]{chosen}[/bold]")
        return chosen
    console.print("[cyan]No telemetry found - generating synthetic sample flight data.[/cyan]")
    return write_sample_files(raw_dir)["nasa_long"]


def kb_stats() -> int:
    """Print knowledge-base statistics: cases on disk and in the vector store, by category and source type."""
    settings = get_settings()
    try:
        documents = load_documents(settings.historical_docs_dir)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    rag = HistoricalRAG(settings.historical_store_dir, OllamaEmbedder(settings.ollama_base_url, settings.embedding_model))
    indexed = rag.stats()
    by_category: dict[str, dict[str, int]] = {}
    for doc in documents:
        by_category.setdefault(doc.category, {}).setdefault(doc.source_type, 0)
        by_category[doc.category][doc.source_type] += 1
    source_types = sorted({doc.source_type for doc in documents})

    console.rule("[bold]Knowledge base statistics")
    console.print(f"Documents folder: {settings.historical_docs_dir}")
    console.print(f"Total cases on disk: [bold]{len(documents)}[/bold]   "
                  f"words: {sum(len(d.text.split()) for d in documents):,}")
    table = Table(title="Cases by category", header_style="bold")
    table.add_column("Category")
    for source_type in source_types:
        table.add_column(source_type, justify="right")
    table.add_column("Total", justify="right")
    table.add_column("Indexed", justify="right")
    for category in sorted(by_category):
        counts = by_category[category]
        table.add_row(category, *(str(counts.get(t, 0)) for t in source_types), str(sum(counts.values())),
                      str(indexed["by_category"].get(category, 0)))
    table.add_row("[bold]total[/bold]", *(f"[bold]{sum(c.get(t, 0) for c in by_category.values())}[/bold]"
                                          for t in source_types),
                  f"[bold]{len(documents)}[/bold]", f"[bold]{indexed['cases']}[/bold]")
    console.print(table)

    type_table = Table(title="Cases by source type", header_style="bold")
    for column in ("Source type", "Cases", "Avg words", "Indexed"):
        type_table.add_column(column, justify="left" if column == "Source type" else "right")
    for source_type in source_types:
        docs = [d for d in documents if d.source_type == source_type]
        type_table.add_row(source_type, str(len(docs)), f"{sum(len(d.text.split()) for d in docs) / len(docs):.0f}",
                           str(indexed["by_source_type"].get(source_type, 0)))
    console.print(type_table)

    if indexed["cases"] == 0:
        console.print("[yellow]Vector store not built - run: python scripts/build_knowledge_base.py[/yellow]")
    else:
        console.print(f"Vector store: {indexed['cases']} cases / {indexed['chunks']} chunks "
                      f"({indexed['embedding_model']}) at {settings.historical_store_dir}")
        if indexed["cases"] != len(documents):
            console.print("[yellow]Vector store is out of date with the documents folder - "
                          "re-run scripts/build_knowledge_base.py[/yellow]")
    return 0


def main() -> int:
    """Run the convenience pipeline. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Check prerequisites and run the AeroTrace-RCA pipeline.")
    parser.add_argument("input", nargs="?", type=Path, help="Telemetry file (defaults to data/raw sample).")
    parser.add_argument("--format", choices=("markdown", "json", "both"), default="both")
    parser.add_argument("--offline", action="store_true", help="Skip all LLM calls.")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--kb-stats", action="store_true", help="Show knowledge-base statistics and exit.")
    args = parser.parse_args()
    if args.kb_stats:
        configure_logging("WARNING")
        return kb_stats()

    configure_logging("DEBUG" if args.verbose else "WARNING")
    console.rule("[bold]AeroTrace-RCA pipeline")
    status = preflight()
    try:
        input_path = ensure_input(args.input)
    except FileNotFoundError as exc:
        console.print(f"[red]Input not found: {exc}[/red]")
        return 2

    formats = ("markdown", "json") if args.format == "both" else (args.format,)
    use_llm = False if args.offline else (status["model_installed"] or status["claude_configured"])
    try:
        result = Supervisor(use_llm=use_llm).run(input_path, mode="report", output_formats=formats)
    except PipelineError as exc:
        console.print(f"[bold red]Pipeline failed:[/bold red] {exc}")
        return 1

    display_result(result, show_decisions=args.verbose)
    counts = result.report.severity_counts if result.report else {}
    console.rule("[bold]Generated")
    console.print(
        f"Analysed [bold]{len(result.readings):,}[/bold] readings from {input_path.name}: "
        f"{len(result.actionable_anomalies)} actionable anomalies "
        f"({counts.get('CRITICAL', 0)} critical, {counts.get('WARNING', 0)} warning), "
        f"{len(result.hypotheses)} hypotheses."
    )
    for kind, path in result.output_paths.items():
        console.print(f"  • {kind:13} {path}")
    return 0 if not result.errors else 1


if __name__ == "__main__":
    sys.exit(main())
