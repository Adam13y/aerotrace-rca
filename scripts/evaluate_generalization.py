"""Run the pipeline on the unseen evaluation flights and score it against ground truth.

The pipeline only receives each flight's CSV; ``data/eval/ground_truth.json`` is read here,
after the run, for scoring. Scoring rules are defined in the ground-truth file.

Usage::

    python scripts/generate_eval_flights.py
    python scripts/evaluate_generalization.py [--flights NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from config.settings import configure_logging, get_settings  # noqa: E402
from src.agents.supervisor import Supervisor  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent.parent / "data" / "eval"
console = Console()


def _overlaps(anomaly: dict[str, Any], fault: dict[str, Any]) -> bool:
    start = anomaly["mission_elapsed_s"]
    end = start + anomaly["duration_s"]
    return anomaly["sensor_id"] == fault["sensor"] and start <= fault["end"] + 5 and end >= fault["start"] - 3


def score(report: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Score one report against its ground truth."""
    actionable = [a for a in report["anomalies"] if not a["expected"]]
    hypotheses = report["hypotheses"]
    group_of: dict[str, str] = {}  # anomaly id -> cause group ("FP" for false positives)
    detected: dict[int, list[str]] = {}
    for anomaly in actionable:
        match = next((i for i, f in enumerate(truth["faults"]) if _overlaps(anomaly, f)), None)
        group_of[anomaly["id"]] = truth["faults"][match]["cause_group"] if match is not None else "FP"
        if match is not None:
            detected.setdefault(match, []).append(anomaly["id"])
    label = {a["id"]: a["label"] for a in report["anomalies"]}
    sensor = {a["id"]: a["sensor_id"] for a in report["anomalies"]}
    result: dict[str, Any] = {"failures": []}

    missed = [f["sensor"] for i, f in enumerate(truth["faults"]) if i not in detected]
    false_pos = [label[a["id"]] for a in actionable if group_of[a["id"]] == "FP"]
    result["detected"] = f"{len(detected)}/{len(truth['faults'])}"
    result["false_positives"] = false_pos
    for m in missed:
        result["failures"].append(f"injected fault on {m} was not detected")
    for fp in false_pos:
        result["failures"].append(f"false-positive anomaly: {fp}")

    top = hypotheses[0] if hypotheses else None
    if truth.get("expect_no_critical"):
        critical = [a["label"] for a in actionable if a["severity"] == "CRITICAL"]
        confident = [h["title"] for h in hypotheses if h["confidence"] >= 0.5]
        ok = not critical and not confident
        result["root_ok"] = ok
        if critical:
            result["failures"].append(f"CRITICAL anomalies on a nominal flight: {critical}")
        if confident:
            result["failures"].append(f"hypotheses with >=50% confidence on a nominal flight: {confident}")
    elif top is None:
        result["root_ok"] = False
        result["failures"].append("no hypotheses produced")
    else:
        text = f"{top['title']} {top['description']}".lower()
        cited_sensors = {sensor[i] for i in top["related_anomalies"]}
        cites_root = bool(cited_sensors & set(truth["top_must_cite_any_of"]))
        terms_ok = not truth["top_terms_any"] or any(t in text for t in truth["top_terms_any"])
        forbidden = [t for t in truth["top_terms_none"] if t in text]
        groups_in_top = {group_of.get(i, "FP") for i in top["related_anomalies"]} - {"FP"}
        both = len(groups_in_top) > 1
        result["root_ok"] = cites_root and terms_ok and not forbidden and not both
        if not cites_root:
            result["failures"].append(f"#1 '{top['title']}' does not cite the root-cause sensor ({truth['top_must_cite_any_of']})")
        if not terms_ok:
            result["failures"].append(f"#1 '{top['title']}' names none of {truth['top_terms_any']}")
        if forbidden:
            result["failures"].append(f"#1 '{top['title']}' blames a wrong mechanism (mentions {forbidden})")
        if both:
            result["failures"].append(f"#1 '{top['title']}' merges independent faults {sorted(groups_in_top)}")

    cited = {i for h in hypotheses for i in h["related_anomalies"]}
    unexplained = [label[i] for ids in detected.values() for i in ids if i not in cited]
    result["all_explained"] = not unexplained
    for u in unexplained:
        result["failures"].append(f"real anomaly not explained by any hypothesis: {u}")

    false_links = []
    for h in hypotheses:
        groups = {group_of.get(i, "FP") for i in h["related_anomalies"]}
        real = groups - {"FP"}
        if len(real) > 1 or (real and "FP" in groups):
            false_links.append(f"'{h['title']}' links " + " + ".join(label[i] for i in h["related_anomalies"]))
    result["false_links"] = false_links
    result["failures"] += [f"false causal link: {fl}" for fl in false_links]
    return result


def main() -> int:
    """Run and score all evaluation flights."""
    parser = argparse.ArgumentParser(description="Score the pipeline on unseen evaluation flights.")
    parser.add_argument("--flights", nargs="*", help="Subset of flight names to run.")
    args = parser.parse_args()
    configure_logging("WARNING")
    truth_all = json.loads((EVAL_DIR / "ground_truth.json").read_text(encoding="utf-8"))["flights"]
    names = args.flights or list(truth_all)
    settings = get_settings()
    console.print(f"OLLAMA_REPRODUCIBLE={settings.ollama_reproducible}  model={settings.ollama_model}")

    rows, details = [], {}
    for name in names:
        console.rule(f"[bold]{name}")
        supervisor = Supervisor(settings=settings)
        result = supervisor.run(EVAL_DIR / "flights" / f"{name}.csv", output_formats=("markdown", "json"))
        report = json.loads(result.output_paths["json"].read_text(encoding="utf-8"))
        scored = score(report, truth_all[name])
        details[name] = {"score": scored, "report_json": str(result.output_paths["json"]),
                         "decision_log": str(result.output_paths["decision_log"]),
                         "hypotheses": [(h["title"], h["confidence"], [a["label"] for a in report["anomalies"]
                                                                       if a["id"] in h["related_anomalies"]])
                                        for h in report["hypotheses"]],
                         "top_description": report["hypotheses"][0]["description"] if report["hypotheses"] else None,
                         "duration_s": round(result.duration_s, 1)}
        rows.append((name, report["hypotheses"][0]["title"] if report["hypotheses"] else "-", scored))
        console.print(f"done in {result.duration_s:.0f}s - {len(scored['failures'])} failure(s)")

    table = Table(title="Generalisation results", header_style="bold")
    for column in ("Flight", "#1 hypothesis", "Detected", "Root cause #1", "All explained", "False links", "False pos."):
        table.add_column(column)
    mark = lambda ok: "[green]yes[/green]" if ok else "[red]NO[/red]"  # noqa: E731
    for name, top, s in rows:
        table.add_row(name, top, s["detected"], mark(s["root_ok"]), mark(s["all_explained"]),
                      "[green]none[/green]" if not s["false_links"] else f"[red]{len(s['false_links'])}[/red]",
                      str(len(s["false_positives"])))
    console.print(table)
    out = EVAL_DIR / "results.json"
    out.write_text(json.dumps(details, indent=2, default=str), encoding="utf-8")
    console.print(f"Details: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
