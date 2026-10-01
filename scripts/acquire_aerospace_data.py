"""Generate the aerospace failure knowledge base used for RAG.

Web scraping is not available, so the knowledge base is rendered from curated source data
bundled in ``scripts/kb_sources/*.json``:

* 80 **synthetic** failure-mode summaries (8 categories x 10, 250-350 words each). They are
  technically representative engineering write-ups, not records of real events; each names a
  real incident only where one genuinely exhibited that failure mode.
* 20 **historical** case summaries of real incidents (400-500 words each).

Output layout::

    data/historical_failures/<category>/<case id>.txt     (synthetic, one folder per category)
    data/historical_failures/historical/<case id>.txt     (real incidents)

Every file starts with ``Key: value`` header lines (Title, Category, Failure mode, Source type, ...)
that ``scripts/build_knowledge_base.py`` stores as vector-store metadata.

Usage::

    python scripts/acquire_aerospace_data.py            # validate all sources and (re)generate files
    python scripts/acquire_aerospace_data.py --check scripts/kb_sources/propulsion.json   # validate one file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = Path(__file__).resolve().parent / "kb_sources"
OUTPUT_DIR = PROJECT_ROOT / "data" / "historical_failures"
MANIFEST = ".generated_manifest.json"

CATEGORIES = (
    "propulsion", "pressurization", "structural", "thermal",
    "electrical", "avionics", "separation", "aerodynamic",
)
CASES_PER_CATEGORY = 10
HISTORICAL_CASES = 20
WORD_LIMITS = {"synthetic": (250, 350), "historical": (400, 500)}

SYNTHETIC_FIELDS = (
    "id", "title", "category", "failure_mode", "failure_mode_description", "causal_chain",
    "telemetry_signatures", "root_cause", "historical_reference", "lessons_learned",
)
HISTORICAL_FIELDS = (
    "id", "title", "category", "failure_mode", "date", "vehicle", "sources",
    "failure_mode_description", "causal_chain", "telemetry_signatures", "root_cause", "lessons_learned",
)
_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_][a-z0-9]+)*$")


def render_body(case: dict[str, Any], source_type: str) -> str:
    """Render the sectioned body text of one case."""
    sections = [
        ("FAILURE MODE", case["failure_mode_description"]),
        ("CAUSAL CHAIN", case["causal_chain"]),
        ("TELEMETRY SIGNATURES", "\n".join(f"- {s}" for s in case["telemetry_signatures"])),
        ("ROOT CAUSE", case["root_cause"]),
    ]
    if source_type == "synthetic":
        sections.append(("HISTORICAL REFERENCE", case["historical_reference"]))
    sections.append(("LESSONS LEARNED", "\n".join(f"- {s}" for s in case["lessons_learned"])))
    return "\n\n".join(f"{name}\n{text.strip()}" for name, text in sections)


def render_file(case: dict[str, Any], source_type: str) -> str:
    """Render the complete file: header lines, blank line, body."""
    header = [
        f"Title: {case['title']}",
        f"Category: {case['category']}",
        f"Failure mode: {case['failure_mode']}",
        f"Source type: {source_type}",
        f"Case ID: {case['id']}",
    ]
    if source_type == "historical":
        header += [f"Date: {case['date']}", f"Vehicle: {case['vehicle']}", f"Sources: {case['sources']}"]
    else:
        header.append("Note: Synthetic engineering summary of a failure mode, not a record of a specific event.")
    return "\n".join(header) + "\n\n" + render_body(case, source_type) + "\n"


def word_count(text: str) -> int:
    """Number of whitespace-separated words."""
    return len(text.split())


def validate_case(case: Any, source_type: str) -> list[str]:
    """Return a list of problems with one case (empty if valid)."""
    if not isinstance(case, dict):
        return ["case is not a JSON object"]
    label = case.get("id", "<no id>")
    fields = HISTORICAL_FIELDS if source_type == "historical" else SYNTHETIC_FIELDS
    problems = [f"{label}: missing field '{f}'" for f in fields if not case.get(f)]
    if problems:
        return problems
    if not _ID_PATTERN.match(str(case["id"])):
        problems.append(f"{label}: id must be lowercase letters/digits with - or _")
    if case["category"] not in CATEGORIES:
        problems.append(f"{label}: category '{case['category']}' not one of {CATEGORIES}")
    for list_field in ("telemetry_signatures", "lessons_learned"):
        value = case[list_field]
        if not isinstance(value, list) or len(value) < 2 or not all(isinstance(v, str) and v.strip() for v in value):
            problems.append(f"{label}: '{list_field}' must be a list of at least 2 non-empty strings")
    if problems:
        return problems
    low, high = WORD_LIMITS[source_type]
    words = word_count(render_body(case, source_type))
    if not low <= words <= high:
        problems.append(f"{label}: body has {words} words, must be {low}-{high}")
    return problems


def load_source(path: Path) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Load one source file. Returns (source_type, cases, problems)."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return "unknown", [], [f"{path.name}: invalid JSON ({exc})"]
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        return "unknown", [], [f"{path.name}: expected an object with a 'cases' list"]
    source_type = "historical" if document.get("source_type") == "historical" else "synthetic"
    problems = [f"{path.name}: {p}" for case in document["cases"] for p in validate_case(case, source_type)]
    return source_type, document["cases"], problems


def check_file(path: Path) -> int:
    """Validate one source file and print per-case word counts."""
    source_type, cases, problems = load_source(path)
    for case in cases:
        if isinstance(case, dict) and all(case.get(f) for f in ("failure_mode_description", "causal_chain")):
            try:
                words = word_count(render_body(case, source_type))
            except (KeyError, TypeError):
                words = -1
            print(f"  {case.get('id', '?'):45} {case.get('category', '?'):15} {words:4} words")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print(f"{path.name}: {len(cases)} {source_type} case(s), {len(problems)} problem(s)")
    return 1 if problems else 0


def generate(sources_dir: Path, output_dir: Path) -> int:
    """Validate every source file and write the knowledge-base text files."""
    synthetic: list[dict[str, Any]] = []
    historical: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(sources_dir.glob("*.json")):
        source_type, cases, file_problems = load_source(path)
        problems += file_problems
        (historical if source_type == "historical" else synthetic).extend(c for c in cases if isinstance(c, dict))

    ids = Counter(c.get("id") for c in synthetic + historical)
    problems += [f"duplicate case id '{i}'" for i, n in ids.items() if n > 1]
    per_category = Counter(c.get("category") for c in synthetic)
    problems += [
        f"category '{cat}' has {per_category.get(cat, 0)} synthetic cases, expected {CASES_PER_CATEGORY}"
        for cat in CATEGORIES if per_category.get(cat, 0) != CASES_PER_CATEGORY
    ]
    if len(historical) != HISTORICAL_CASES:
        problems.append(f"{len(historical)} historical cases, expected {HISTORICAL_CASES}")
    if problems:
        print("Validation failed - no files written:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST
    previous = set(json.loads(manifest_path.read_text(encoding="utf-8"))) if manifest_path.exists() else set()
    written: list[str] = []
    for source_type, cases in (("synthetic", synthetic), ("historical", historical)):
        for case in cases:
            folder = "historical" if source_type == "historical" else case["category"]
            relative = f"{folder}/{case['id']}.txt"
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(render_file(case, source_type), encoding="utf-8")
            written.append(relative)
    # Remove files this script generated previously that no longer exist in the sources.
    # Files added by hand (not in the manifest) are never touched.
    for stale in sorted(previous - set(written)):
        (output_dir / stale).unlink(missing_ok=True)
        print(f"  removed stale generated file {stale}")
    manifest_path.write_text(json.dumps(sorted(written), indent=1), encoding="utf-8")

    print(f"Generated {len(written)} cases in {output_dir}")
    for cat in CATEGORIES:
        print(f"  {cat:15} {per_category[cat]:3} synthetic")
    print(f"  {'historical':15} {len(historical):3} real incidents")
    return 0


def main() -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description="Generate the aerospace failure knowledge base.")
    parser.add_argument("--check", type=Path, help="Validate a single source JSON file and exit.")
    parser.add_argument("--sources", type=Path, default=SOURCES_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.check:
        return check_file(args.check)
    return generate(args.sources, args.output)


if __name__ == "__main__":
    sys.exit(main())
