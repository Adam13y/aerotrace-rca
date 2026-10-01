"""JSON / JSON-Lines telemetry ingestion.

Supported shapes:

1. A list of records (long or wide), or JSON-Lines with one record per line::

       [{"timestamp": "...", "sensor_id": "lox_tank_pressure", "value": 50.1, "unit": "psi"}, ...]

2. A channel-oriented document::

       {"base_time": "2026-01-01T00:00:00Z",
        "sensors": {"lox_tank_pressure": {"unit": "psi", "samples": [[0.0, 50.1], [0.1, 50.0]]}}}

3. A wrapper ``{"records": [...]}`` / ``{"data": [...]}`` / ``{"telemetry": [...]}``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingestion.csv_parser import CSVParserConfig, CSVTelemetryParser
from src.models.telemetry_schema import TelemetryReading
from src.utils.helpers import normalize_timestamp
from src.utils.logger import get_logger

logger = get_logger("ingestion.json")

_WRAPPER_KEYS = ("records", "data", "telemetry", "readings", "samples")


class JSONTelemetryParser:
    """Parse telemetry stored as JSON or JSON-Lines."""

    def __init__(self, config: CSVParserConfig | None = None) -> None:
        """Create a parser. Shares missing-value and timestamp options with the CSV parser."""
        self.config = config or CSVParserConfig()
        self._tabular = CSVTelemetryParser(self.config)

    def parse(self, path: str | Path) -> list[TelemetryReading]:
        """Parse a ``.json`` or ``.jsonl`` file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the document structure is not recognised.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        text = path.read_text(encoding="utf-8")
        try:
            document: Any = (
                [json.loads(line) for line in text.splitlines() if line.strip()]
                if path.suffix.lower() == ".jsonl"
                else json.loads(text)
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON ({exc})") from exc
        readings = self.parse_document(document, source=path.name)
        logger.info("Parsed %d readings from %s", len(readings), path.name)
        return readings

    def parse_document(self, document: Any, source: str = "json") -> list[TelemetryReading]:
        """Parse an already-decoded JSON document.

        Raises:
            ValueError: If the document structure is not recognised.
        """
        if isinstance(document, dict):
            if isinstance(document.get("sensors"), dict):
                return self._parse_channels(document, source)
            for key in _WRAPPER_KEYS:
                if isinstance(document.get(key), list):
                    return self._parse_records(document[key], source)
            raise ValueError(f"{source}: unrecognised JSON object (keys: {sorted(document)[:10]})")
        if isinstance(document, list):
            return self._parse_records(document, source)
        raise ValueError(f"{source}: top-level JSON must be an object or array")

    def _parse_records(self, records: list[Any], source: str) -> list[TelemetryReading]:
        if not records:
            raise ValueError(f"{source}: no records")
        if not all(isinstance(r, dict) for r in records):
            raise ValueError(f"{source}: every record must be a JSON object")
        flattened = pd.json_normalize(records, sep="_")
        # Merge nested metadata columns back out; they are not telemetry channels.
        flattened = flattened.drop(columns=[c for c in flattened.columns if c.startswith("metadata_")])
        return self._tabular.parse_frame(flattened, source=source)

    def _parse_channels(self, document: dict[str, Any], source: str) -> list[TelemetryReading]:
        base_time = (
            normalize_timestamp(document["base_time"]) if document.get("base_time") else self.config.base_time
        )
        rows: list[dict[str, Any]] = []
        for sensor_id, channel in document["sensors"].items():
            if not isinstance(channel, dict) or not isinstance(channel.get("samples"), list):
                logger.warning("%s: channel %r has no 'samples' list; skipped", source, sensor_id)
                continue
            unit = str(channel.get("unit", ""))
            for sample in channel["samples"]:
                if isinstance(sample, dict):
                    t_raw, value = sample.get("t", sample.get("timestamp")), sample.get("value")
                elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
                    t_raw, value = sample[0], sample[1]
                else:
                    continue
                try:
                    timestamp = normalize_timestamp(t_raw, base_time)
                except ValueError:
                    continue
                rows.append({"timestamp": timestamp, "sensor_id": sensor_id, "value": value, "unit": unit})
        if not rows:
            raise ValueError(f"{source}: no usable samples in 'sensors'")
        frame = pd.DataFrame(rows)
        parser = CSVTelemetryParser(
            CSVParserConfig(
                layout="long",
                missing_strategy=self.config.missing_strategy,
                max_interpolation_gap=self.config.max_interpolation_gap,
                base_time=base_time,
            )
        )
        return parser.parse_frame(frame, source=source)
