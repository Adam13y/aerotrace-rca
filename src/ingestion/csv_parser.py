"""CSV telemetry ingestion and synthetic flight-data generation.

Two layouts are supported and detected automatically:

* **Long / NASA-style** - one row per sample::

      timestamp,sensor_id,value,unit
      2026-01-01T00:00:00.000Z,lox_tank_pressure,50.12,psi

  Column aliases such as ``mnemonic``/``channel`` and ``eng_value``/``units`` are accepted.

* **Wide / SpaceX-webcast-style** - one row per time step, one column per sensor,
  units embedded in the header (``Chamber Pressure (psi)`` or ``chamber_pressure_psi``)
  and a relative time column such as ``T+ (s)``, ``time_s`` or ``MET``.

Timestamps are normalised to timezone-aware UTC and exported in ISO-8601.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from src.models.telemetry_schema import TelemetryReading
from src.utils.helpers import DEFAULT_BASE_TIME, normalize_timestamp
from src.utils.logger import get_logger

logger = get_logger("ingestion.csv")

MissingStrategy = Literal["interpolate", "ffill", "drop"]
CsvLayout = Literal["auto", "long", "wide"]

_TIME_ALIASES = {
    "time", "timestamp", "t", "ts", "met", "times", "timesec", "timeseconds", "elapsed",
    "elapsedtime", "missionelapsedtime", "utc", "datetime", "date", "tplus", "tpluss", "mets",
}
_SENSOR_ALIASES = {"sensorid", "sensor", "mnemonic", "channel", "channelid", "parameter", "tag", "name"}
_VALUE_ALIASES = {"value", "engvalue", "engineeringvalue", "val", "reading", "measurement", "raw"}
_UNIT_ALIASES = {"unit", "units", "engunit", "engunits"}
#: Lower-case unit suffix -> canonical spelling.
_KNOWN_UNITS = {
    "psi": "psi", "psia": "psia", "psig": "psig", "bar": "bar", "kpa": "kPa", "pa": "Pa", "mpa": "MPa",
    "c": "degC", "degc": "degC", "k": "K", "f": "degF", "degf": "degF", "g": "g", "grms": "gRMS",
    "v": "V", "vdc": "VDC", "a": "A", "amp": "A", "w": "W", "hz": "Hz", "rpm": "rpm", "m": "m",
    "km": "km", "mps": "m/s", "kmh": "km/h", "kph": "km/h", "deg": "deg", "pct": "%",
}


def _norm(name: str) -> str:
    """Lowercase a column name and strip non-alphanumerics."""
    return re.sub(r"[^a-z0-9]", "", name.lower().replace("+", "plus"))


def _slugify(name: str) -> str:
    """Turn a human-readable sensor name into a snake_case identifier."""
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", name.strip().lower())).strip("_")


def split_sensor_header(header: str) -> tuple[str, str]:
    """Split a wide-format header into ``(sensor_id, unit)``.

    Examples:
        >>> split_sensor_header("Chamber Pressure (psi)")
        ('chamber_pressure', 'psi')
        >>> split_sensor_header("bus_voltage_v")
        ('bus_voltage', 'V')
    """
    match = re.fullmatch(r"\s*(.+?)\s*[\(\[]\s*([^\)\]]+?)\s*[\)\]]\s*", header)
    if match:
        return _slugify(match.group(1)), match.group(2)
    slug = _slugify(header)
    parts = slug.rsplit("_", 1)
    if len(parts) == 2 and parts[1] in _KNOWN_UNITS:
        return parts[0], _KNOWN_UNITS[parts[1]]
    return slug, ""


@dataclass(slots=True)
class CSVParserConfig:
    """Options controlling CSV parsing."""

    layout: CsvLayout = "auto"
    missing_strategy: MissingStrategy = "interpolate"
    max_interpolation_gap: int = 10
    base_time: datetime = DEFAULT_BASE_TIME
    source_label: str | None = None


class CSVTelemetryParser:
    """Parse rocket telemetry CSV files into :class:`TelemetryReading` objects."""

    def __init__(self, config: CSVParserConfig | None = None) -> None:
        """Create a parser with the given configuration (defaults if omitted)."""
        self.config = config or CSVParserConfig()

    def parse(self, path: str | Path) -> list[TelemetryReading]:
        """Parse a CSV file.

        Args:
            path: Path to the CSV file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the layout cannot be determined or no valid samples remain.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        try:
            frame = pd.read_csv(path, comment="#", skipinitialspace=True)
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{path} is empty") from exc
        readings = self.parse_frame(frame, source=self.config.source_label or path.name)
        logger.info("Parsed %d readings from %s", len(readings), path.name)
        return readings

    def parse_frame(self, frame: pd.DataFrame, source: str = "dataframe") -> list[TelemetryReading]:
        """Parse an in-memory DataFrame using the configured layout.

        Raises:
            ValueError: If the layout cannot be determined or no valid samples remain.
        """
        if frame.empty:
            raise ValueError(f"{source}: no rows")
        layout = self.config.layout if self.config.layout != "auto" else self._detect_layout(frame)
        logger.debug("%s detected as %s-format telemetry", source, layout)
        long_frame = self._from_long(frame) if layout == "long" else self._from_wide(frame)
        long_frame = self._handle_missing(long_frame)
        if long_frame.empty:
            raise ValueError(f"{source}: no valid samples after cleaning")
        return [
            TelemetryReading(
                timestamp=row.timestamp,
                sensor_id=row.sensor_id,
                value=float(row.value),
                unit=row.unit,
                metadata={"source": source, "format": layout, **({"imputed": True} if row.imputed else {})},
            )
            for row in long_frame.itertuples(index=False)
        ]

    # ------------------------------------------------------------------ #
    def _detect_layout(self, frame: pd.DataFrame) -> Literal["long", "wide"]:
        normalised = {_norm(c) for c in frame.columns}
        if normalised & _SENSOR_ALIASES and normalised & _VALUE_ALIASES:
            return "long"
        if normalised & _TIME_ALIASES:
            return "wide"
        raise ValueError(f"Cannot detect CSV layout from columns: {list(frame.columns)}")

    @staticmethod
    def _find_column(frame: pd.DataFrame, aliases: set[str], required: bool = True) -> str | None:
        for column in frame.columns:
            if _norm(str(column)) in aliases:
                return str(column)
        if required:
            raise ValueError(f"Missing required column (one of {sorted(aliases)})")
        return None

    def _parse_times(self, series: pd.Series) -> pd.Series:
        """Vectorised timestamp normalisation with a per-value fallback."""
        if pd.api.types.is_datetime64_any_dtype(series):
            return pd.Series([normalize_timestamp(v, self.config.base_time) for v in series])
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return pd.Series([normalize_timestamp(float(v), self.config.base_time) for v in numeric])
        parsed: list[datetime | None] = []
        for value in series:
            try:
                parsed.append(normalize_timestamp(value, self.config.base_time))
            except ValueError:
                parsed.append(None)
        bad = sum(p is None for p in parsed)
        if bad:
            logger.warning("Dropping %d rows with unparseable timestamps", bad)
        return pd.Series(parsed, dtype="object")

    def _from_long(self, frame: pd.DataFrame) -> pd.DataFrame:
        time_col = self._find_column(frame, _TIME_ALIASES)
        sensor_col = self._find_column(frame, _SENSOR_ALIASES)
        value_col = self._find_column(frame, _VALUE_ALIASES)
        unit_col = self._find_column(frame, _UNIT_ALIASES, required=False)
        result = pd.DataFrame(
            {
                "timestamp": self._parse_times(frame[time_col]).values,
                "sensor_id": frame[sensor_col].astype(str).str.strip().map(_slugify),
                "value": pd.to_numeric(frame[value_col], errors="coerce"),
                "unit": frame[unit_col].fillna("").astype(str).str.strip() if unit_col else "",
            }
        )
        return result.dropna(subset=["timestamp"]).loc[lambda d: d["sensor_id"] != ""]

    def _from_wide(self, frame: pd.DataFrame) -> pd.DataFrame:
        time_col = self._find_column(frame, _TIME_ALIASES)
        timestamps = self._parse_times(frame[time_col])
        pieces: list[pd.DataFrame] = []
        for column in frame.columns:
            if column == time_col or _norm(str(column)) in _TIME_ALIASES:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if values.notna().sum() == 0:
                logger.debug("Skipping non-numeric column %r", column)
                continue
            sensor_id, unit = split_sensor_header(str(column))
            pieces.append(
                pd.DataFrame({"timestamp": timestamps.values, "sensor_id": sensor_id, "value": values, "unit": unit})
            )
        if not pieces:
            raise ValueError("Wide CSV contains no numeric sensor columns")
        return pd.concat(pieces, ignore_index=True).dropna(subset=["timestamp"])

    def _handle_missing(self, frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["value"] = frame["value"].replace([np.inf, -np.inf], np.nan)
        frame["imputed"] = frame["value"].isna()
        missing = int(frame["imputed"].sum())
        strategy = self.config.missing_strategy
        if missing:
            logger.info("Handling %d missing values with strategy '%s'", missing, strategy)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.sort_values(["sensor_id", "timestamp"], kind="stable")
        if strategy == "drop" or not missing:
            frame = frame.dropna(subset=["value"])
            frame["imputed"] = False
        else:
            filled: list[pd.DataFrame] = []
            for _, group in frame.groupby("sensor_id", sort=False):
                group = group.copy()
                if strategy == "interpolate":
                    indexed = group.set_index("timestamp")["value"]
                    group["value"] = indexed.interpolate(
                        method="time", limit=self.config.max_interpolation_gap, limit_area="inside"
                    ).to_numpy()
                else:
                    group["value"] = group["value"].ffill(limit=self.config.max_interpolation_gap)
                filled.append(group)
            frame = pd.concat(filled).dropna(subset=["value"])
        # Units: forward-fill within each sensor so sparse unit columns propagate.
        frame["unit"] = frame.groupby("sensor_id")["unit"].transform(
            lambda u: u.where(u != "").ffill().bfill().fillna("")
        )
        frame["timestamp"] = [ts.to_pydatetime() for ts in frame["timestamp"]]
        return frame.reset_index(drop=True)


def export_normalized_csv(readings: list[TelemetryReading], path: str | Path) -> Path:
    """Write readings as long-format CSV with ISO-8601 timestamps.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": [r.iso_timestamp for r in readings],
            "sensor_id": [r.sensor_id for r in readings],
            "value": [r.value for r in readings],
            "unit": [r.unit for r in readings],
            "imputed": [bool(r.metadata.get("imputed", False)) for r in readings],
        }
    )
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #
SENSOR_UNITS: dict[str, str] = {
    "chamber_pressure": "psi",
    "lox_tank_pressure": "psi",
    "fuel_tank_pressure": "psi",
    "engine_bay_temp": "degC",
    "turbopump_vibration": "g",
    "avionics_bus_voltage": "V",
}

#: Anomalies injected by :func:`generate_sample_telemetry` (MET seconds), for tests and demos.
INJECTED_ANOMALIES: dict[str, float] = {
    "lox_tank_pressure": 142.0,
    "turbopump_vibration": 143.5,
    "engine_bay_temp": 145.0,
    "avionics_bus_voltage": 210.0,
    "fuel_tank_pressure": 230.0,
}


def _smoothstep(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """Smooth 0->1 transition between ``start`` and ``end``."""
    x = np.clip((t - start) / (end - start), 0.0, 1.0)
    return x * x * (3 - 2 * x)


def generate_sample_telemetry(
    duration_s: float = 300.0,
    sample_rate_hz: float = 10.0,
    seed: int = 42,
    base_time: datetime = DEFAULT_BASE_TIME,
    inject_anomalies: bool = True,
    missing_fraction: float = 0.004,
) -> pd.DataFrame:
    """Generate a realistic single-stage ascent telemetry set in wide layout.

    The profile includes engine start, a throttle bucket around Max-Q (55-83 s),
    main engine cutoff at ~285 s, slow thermal soak, vibration that peaks around
    Max-Q, and sensor noise. With ``inject_anomalies`` the following faults are added:

    * T+142 s - LOX tank pressurisation drop (-6 psi) with slow recovery.
    * T+143.5 s - turbopump vibration burst (+6 g, ~1 s).
    * T+145 s - engine-bay temperature excursion (+35 degC) - correlated cascade.
    * T+210 s - avionics bus under-voltage sag (independent electrical event).
    * T+230 s - single-sample spike on fuel tank pressure (sensor glitch).

    Returns:
        A wide DataFrame with a ``met_s`` column, an ISO ``timestamp`` column and one
        column per sensor (``<sensor>_<unit>``).
    """
    rng = np.random.default_rng(seed)
    t = np.round(np.arange(0.0, duration_s, 1.0 / sample_rate_hz), 6)
    n = t.size

    ignition = _smoothstep(t, 0.0, 3.0)
    throttle = 1.0 - 0.28 * _smoothstep(t, 55.0, 60.0) + 0.28 * _smoothstep(t, 78.0, 83.0)
    cutoff = 1.0 - _smoothstep(t, 284.0, 287.0)
    engine = ignition * throttle * cutoff

    chamber = 1400.0 * engine + rng.normal(0, 4.0, n)
    lox = 50.0 + 0.6 * np.sin(2 * np.pi * t / 37.0) - 0.004 * t + rng.normal(0, 0.12, n)
    fuel = 45.0 - 0.006 * t + 0.3 * np.sin(2 * np.pi * t / 51.0) + rng.normal(0, 0.10, n)
    temp = 20.0 + 100.0 * (1 - np.exp(-t / 90.0)) * (0.3 + 0.7 * ignition) + rng.normal(0, 0.4, n)
    maxq = 1.5 * np.exp(-0.5 * ((t - 70.0) / 7.0) ** 2)
    vib = (0.3 + 1.7 * engine) + maxq + np.abs(rng.normal(0, 0.12, n))
    volt = 28.0 - 0.15 * ignition + rng.normal(0, 0.04, n)

    if inject_anomalies:
        # LOX pressurisation drop: fast decay, slow regulator recovery.
        drop = _smoothstep(t, 142.0, 143.5) * np.exp(-np.clip(t - 143.5, 0, None) / 12.0)
        lox -= 6.0 * drop
        vib += 6.0 * np.exp(-0.5 * ((t - 144.0) / 0.35) ** 2)
        temp += 35.0 * _smoothstep(t, 145.0, 149.0) * np.exp(-np.clip(t - 149.0, 0, None) / 30.0)
        sag = (t >= 210.0) & (t < 210.8)
        volt[sag] -= 1.8
        fuel[np.argmin(np.abs(t - 230.0))] += 25.0

    columns = {
        "chamber_pressure": chamber,
        "lox_tank_pressure": lox,
        "fuel_tank_pressure": fuel,
        "engine_bay_temp": temp,
        "turbopump_vibration": vib,
        "avionics_bus_voltage": volt,
    }
    frame = pd.DataFrame({"met_s": t})
    frame.insert(0, "timestamp", [normalize_timestamp(float(s), base_time).isoformat(timespec="milliseconds") for s in t])
    for sensor, values in columns.items():
        values = np.round(values, 4)
        if missing_fraction > 0:
            # Never blank out the injected anomaly samples themselves.
            mask = rng.random(n) < missing_fraction
            if inject_anomalies and sensor in INJECTED_ANOMALIES:
                mask &= np.abs(t - INJECTED_ANOMALIES[sensor]) > 3.0
            values = np.where(mask, np.nan, values)
        frame[f"{sensor}_{SENSOR_UNITS[sensor]}"] = values
    return frame


def to_long_format(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert :func:`generate_sample_telemetry` output to NASA-style long layout."""
    sensor_columns = [c for c in wide.columns if c not in {"timestamp", "met_s"}]
    long_frame = wide.melt(id_vars=["timestamp"], value_vars=sensor_columns, var_name="column", value_name="value")
    unit_by_column = {c: split_sensor_header(c) for c in sensor_columns}
    long_frame["sensor_id"] = long_frame["column"].map(lambda c: unit_by_column[c][0])
    long_frame["unit"] = long_frame["column"].map(lambda c: unit_by_column[c][1])
    return long_frame[["timestamp", "sensor_id", "value", "unit"]]


def to_spacex_style(wide: pd.DataFrame) -> pd.DataFrame:
    """Convert generator output to a SpaceX-webcast-style wide layout (``T+ (s)``, ``Name (unit)``)."""
    renamed = {"met_s": "T+ (s)"}
    for sensor, unit in SENSOR_UNITS.items():
        renamed[f"{sensor}_{unit}"] = f"{sensor.replace('_', ' ').title()} ({unit})"
    return wide.drop(columns=["timestamp"]).rename(columns=renamed)


def write_sample_files(output_dir: str | Path, seed: int = 42) -> dict[str, Path]:
    """Write the synthetic flight in long (NASA) and wide (SpaceX) CSV layouts.

    Returns:
        Mapping of layout name to written path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wide = generate_sample_telemetry(seed=seed)
    paths = {
        "nasa_long": output_dir / "sample_flight_nasa_long.csv",
        "spacex_wide": output_dir / "sample_flight_spacex_wide.csv",
    }
    to_long_format(wide).to_csv(paths["nasa_long"], index=False)
    to_spacex_style(wide).to_csv(paths["spacex_wide"], index=False)
    logger.info("Sample telemetry written to %s (generated %s)", output_dir, datetime.now(UTC).isoformat())
    return paths
