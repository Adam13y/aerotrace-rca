"""Telemetry ingestion: CSV, JSON and FITS readers plus a unified loader."""

from __future__ import annotations

from pathlib import Path

from src.ingestion.csv_parser import (
    CSVParserConfig,
    CSVTelemetryParser,
    export_normalized_csv,
    generate_sample_telemetry,
    write_sample_files,
)
from src.ingestion.fits_reader import FITSTelemetryReader
from src.ingestion.json_parser import JSONTelemetryParser
from src.models.telemetry_schema import TelemetryReading

SUPPORTED_SUFFIXES = {".csv": "csv", ".json": "json", ".jsonl": "json", ".fits": "fits", ".fit": "fits", ".fts": "fits"}


def load_telemetry(path: str | Path, config: CSVParserConfig | None = None) -> list[TelemetryReading]:
    """Load telemetry from any supported file type, chosen by extension.

    Raises:
        ValueError: If the file extension is not supported.
    """
    path = Path(path)
    kind = SUPPORTED_SUFFIXES.get(path.suffix.lower())
    if kind == "csv":
        return CSVTelemetryParser(config).parse(path)
    if kind == "json":
        return JSONTelemetryParser(config).parse(path)
    if kind == "fits":
        return FITSTelemetryReader(config).parse(path)
    raise ValueError(f"Unsupported telemetry file type '{path.suffix}' (supported: {sorted(SUPPORTED_SUFFIXES)})")


__all__ = [
    "CSVParserConfig",
    "CSVTelemetryParser",
    "FITSTelemetryReader",
    "JSONTelemetryParser",
    "SUPPORTED_SUFFIXES",
    "export_normalized_csv",
    "generate_sample_telemetry",
    "load_telemetry",
    "write_sample_files",
]
