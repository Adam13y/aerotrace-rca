"""Pydantic data models shared by every agent in the pipeline."""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator

_SENSOR_WORDS = {
    "lox": "LOX", "lh2": "LH2", "ch4": "CH4", "rp1": "RP-1", "copv": "COPV", "imu": "IMU",
    "gnc": "GNC", "temp": "temperature", "vib": "vibration", "press": "pressure",
}


def sensor_display_name(sensor_id: str) -> str:
    """Turn a sensor ID into plain English, e.g. ``lox_tank_pressure`` -> ``LOX tank pressure``."""
    words = [_SENSOR_WORDS.get(w, w) for w in sensor_id.replace("-", "_").lower().split("_") if w]
    text = " ".join(words)
    return text[:1].upper() + text[1:] if text else sensor_id


def _met_text(seconds: float | None) -> str:
    """Mission elapsed time as ``T+142.4s``."""
    if seconds is None:
        return "T+?"
    return f"T{'+' if seconds >= 0 else '-'}{abs(seconds):.1f}s"


def _ensure_utc(value: datetime) -> datetime:
    """Attach UTC to naive datetimes and convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class Severity(StrEnum):
    """Anomaly severity levels, ordered from most to least severe."""

    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"

    @property
    def rank(self) -> int:
        """Numeric rank where higher means more severe."""
        return {"CRITICAL": 3, "WARNING": 2, "INFO": 1}[self.value]


class TelemetryReading(BaseModel):
    """A single timestamped sensor sample."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime
    sensor_id: str = Field(min_length=1)
    value: float
    unit: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", mode="after")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _ensure_utc(value)

    @field_validator("value", mode="after")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("Telemetry value must be finite")
        return value

    @property
    def iso_timestamp(self) -> str:
        """Timestamp rendered as ISO-8601 with millisecond precision."""
        return self.timestamp.isoformat(timespec="milliseconds")


class AnomalyEvent(BaseModel):
    """A contiguous anomalous interval detected on one sensor."""

    id: str = Field(default_factory=lambda: f"ANM-{uuid.uuid4().hex[:8].upper()}")
    timestamp: datetime
    end_timestamp: datetime | None = None
    sensor_id: str
    severity: Severity
    description: str
    readings: list[TelemetryReading] = Field(default_factory=list)
    unit: str = ""
    mission_elapsed_s: float | None = None
    duration_s: float = 0.0
    peak_value: float | None = None
    baseline_value: float | None = None
    score: float = 0.0
    #: Longest run of consecutive samples whose amplitude exceeded the z-score threshold or a hard limit.
    consecutive_samples: int = 0
    detection_methods: list[str] = Field(default_factory=list)
    expected: bool = False
    flight_phase: str | None = None
    correlated_with: list[str] = Field(default_factory=list)
    classification: dict[str, Any] = Field(default_factory=dict)

    @field_validator("timestamp", "end_timestamp", mode="after")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _ensure_utc(value)

    @property
    def change_kind(self) -> str:
        """Plain-English shape of the deviation: spike, dip, rise or drop."""
        if self.peak_value is None or self.baseline_value is None:
            return "anomaly"
        if self.duration_s <= 0.3:
            return "spike" if self.peak_value > self.baseline_value else "dip"
        return "rise" if self.peak_value > self.baseline_value else "drop"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def label(self) -> str:
        """Human-readable reference, e.g. ``LOX tank pressure drop (CRITICAL) at T+142.4s``.

        This is what the LLM agents read and cite; the ``id`` stays the internal key.
        """
        return (
            f"{sensor_display_name(self.sensor_id)} {self.change_kind} "
            f"({self.severity.value}) at {_met_text(self.mission_elapsed_s)}"
        )

    def to_prompt_dict(self, labels: Mapping[str, str] | None = None) -> dict[str, Any]:
        """Compact, LLM-friendly representation (no raw readings, no internal IDs).

        Args:
            labels: ID -> label lookup used to render ``correlated_with``.
        """
        labels = labels or {}
        return {
            "anomaly": self.label,
            "sensor_id": self.sensor_id,
            "severity": self.severity.value,
            "met_s": None if self.mission_elapsed_s is None else round(self.mission_elapsed_s, 2),
            "duration_s": round(self.duration_s, 2),
            "flight_phase": self.flight_phase,
            "unit": self.unit,
            "baseline_value": None if self.baseline_value is None else round(self.baseline_value, 3),
            "peak_value": None if self.peak_value is None else round(self.peak_value, 3),
            "score": round(self.score, 2),
            "methods": self.detection_methods,
            "expected_transient": self.expected,
            "correlated_with": [labels.get(i, i) for i in self.correlated_with],
            "description": self.description,
            "classification": self.classification or None,
        }


def anomaly_labels(events: Iterable[AnomalyEvent]) -> dict[str, str]:
    """Map anomaly IDs to their human-readable labels."""
    return {event.id: event.label for event in events}


class Hypothesis(BaseModel):
    """A candidate root cause with its supporting and refuting evidence."""

    title: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    related_anomalies: list[str] = Field(default_factory=list)
    subsystem: str = "unknown"
    recommended_actions: list[str] = Field(default_factory=list)
    #: Observations that would confirm this cause and rule out competing ones.
    distinguishing_evidence: list[str] = Field(default_factory=list)
    #: "root_cause", "independent_events" (coincident but unrelated) or "insufficient_evidence".
    hypothesis_type: str = "root_cause"
    #: Validator annotations, e.g. "unobservable - no sensor coverage: structural".
    labels: list[str] = Field(default_factory=list)
    source: str = "rule-based"

    @field_validator("confidence", mode="before")
    @classmethod
    def _clamp(cls, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"confidence must be numeric, got {value!r}") from exc
        if number > 1.0 and number <= 100.0:
            number /= 100.0  # tolerate percentages from LLM output
        return min(max(number, 0.0), 1.0)


class SensorSummary(BaseModel):
    """Descriptive statistics for one sensor channel."""

    sensor_id: str
    unit: str
    samples: int
    start: datetime
    end: datetime
    sample_rate_hz: float
    minimum: float
    maximum: float
    mean: float
    std: float
    imputed_samples: int = 0


class TelemetryAnalysis(BaseModel):
    """Dataset-level overview produced by the telemetry analyst."""

    source_file: str
    total_readings: int
    start: datetime
    end: datetime
    duration_s: float
    sensors: list[SensorSummary]
    data_quality_notes: list[str] = Field(default_factory=list)


class RCAReport(BaseModel):
    """The final root-cause-analysis report."""

    title: str
    summary: str
    anomalies: list[AnomalyEvent]
    hypotheses: list[Hypothesis]
    conclusion: str
    recommendations: list[str]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    analysis: TelemetryAnalysis | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def severity_counts(self) -> dict[str, int]:
        """Count of anomalies per severity level."""
        counts = {level.value: 0 for level in Severity}
        for anomaly in self.anomalies:
            counts[anomaly.severity.value] += 1
        return counts
