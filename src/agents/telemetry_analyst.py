"""Telemetry analyst agent: data-quality review, flight-phase tagging and correlation.

This agent is deterministic. It characterises the dataset before detection and
enriches detected anomalies with the flight phase and cross-sensor correlations
that the triage and hypothesis agents rely on.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np

from src.models.telemetry_schema import AnomalyEvent, SensorSummary, TelemetryAnalysis, TelemetryReading
from src.utils.helpers import readings_to_frame
from src.utils.logger import get_logger

logger = get_logger("agents.telemetry_analyst")


@dataclass(slots=True)
class FlightPhase:
    """A named mission-elapsed-time interval."""

    name: str
    start: float
    end: float


@dataclass(slots=True)
class AnalystConfig:
    """Configuration for :class:`TelemetryAnalyst`."""

    correlation_window_seconds: float = 5.0
    flight_phases: list[FlightPhase] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> AnalystConfig:
        """Build from the ``telemetry_analyst`` section of ``agents.yaml``."""
        data = data or {}
        phases = [
            FlightPhase(str(p["name"]), float(p["start"]), float(p["end"]))
            for p in data.get("flight_phases", []) or []
        ]
        return cls(
            correlation_window_seconds=float(data.get("correlation_window_seconds", 5.0)),
            flight_phases=phases,
        )


class TelemetryAnalyst:
    """Characterise telemetry and contextualise anomalies."""

    def __init__(self, config: AnalystConfig | None = None) -> None:
        """Create the analyst with the given configuration."""
        self.config = config or AnalystConfig()

    def analyze(self, readings: Sequence[TelemetryReading], source: str = "telemetry") -> TelemetryAnalysis:
        """Produce per-sensor statistics and data-quality notes.

        Raises:
            ValueError: If ``readings`` is empty.
        """
        if not readings:
            raise ValueError("Cannot analyse an empty telemetry set")
        frame = readings_to_frame(readings)
        start, end = frame["timestamp"].min(), frame["timestamp"].max()
        summaries: list[SensorSummary] = []
        notes: list[str] = []
        for sensor_id, group in frame.groupby("sensor_id", sort=True):
            values = group["value"].to_numpy(dtype=float)
            seconds = (group["timestamp"] - group["timestamp"].min()).dt.total_seconds().to_numpy()
            diffs = np.diff(seconds)
            positive = diffs[diffs > 0]
            median_dt = float(np.median(positive)) if positive.size else 0.0
            imputed = int(group["imputed"].sum())
            summaries.append(
                SensorSummary(
                    sensor_id=str(sensor_id),
                    unit=str(group["unit"].iloc[0]),
                    samples=len(group),
                    start=group["timestamp"].min().to_pydatetime(),
                    end=group["timestamp"].max().to_pydatetime(),
                    sample_rate_hz=round(1.0 / median_dt, 3) if median_dt > 0 else 0.0,
                    minimum=float(values.min()),
                    maximum=float(values.max()),
                    mean=float(values.mean()),
                    std=float(values.std()),
                    imputed_samples=imputed,
                )
            )
            notes.extend(self._quality_notes(str(sensor_id), values, diffs, median_dt, imputed))
        analysis = TelemetryAnalysis(
            source_file=source,
            total_readings=len(frame),
            start=start.to_pydatetime(),
            end=end.to_pydatetime(),
            duration_s=float((end - start).total_seconds()),
            sensors=summaries,
            data_quality_notes=notes,
        )
        logger.info(
            "Analysed %d readings across %d sensors over %.1fs (%d data-quality notes)",
            analysis.total_readings, len(summaries), analysis.duration_s, len(notes),
        )
        return analysis

    @staticmethod
    def _quality_notes(sensor_id: str, values: np.ndarray, diffs: np.ndarray, median_dt: float, imputed: int) -> list[str]:
        notes: list[str] = []
        if imputed:
            notes.append(f"{sensor_id}: {imputed} missing samples were imputed ({imputed / max(len(values), 1):.1%}).")
        if np.any(diffs == 0):
            notes.append(f"{sensor_id}: {int(np.sum(diffs == 0))} duplicate timestamps.")
        if median_dt > 0:
            gaps = diffs[diffs > 5 * median_dt]
            if gaps.size:
                notes.append(
                    f"{sensor_id}: {gaps.size} data gap(s) longer than 5x the nominal sample interval "
                    f"(largest {gaps.max():.2f}s)."
                )
            jitter = float(np.std(diffs[diffs > 0]) / median_dt) if np.any(diffs > 0) else 0.0
            if jitter > 0.1:
                notes.append(f"{sensor_id}: irregular sampling (interval jitter {jitter:.0%}).")
        if values.size > 10 and float(np.ptp(values)) == 0.0:
            notes.append(f"{sensor_id}: constant value {values[0]:.4g} for the whole record - possible stuck sensor.")
        return notes

    def phase_for(self, met: float | None) -> str | None:
        """Return the configured flight phase containing ``met``."""
        if met is None:
            return None
        return next((p.name for p in self.config.flight_phases if p.start <= met < p.end), None)

    def contextualize(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """Attach flight phase and cross-sensor correlations to anomalies.

        Two anomalies on different sensors are correlated when their intervals lie within
        ``correlation_window_seconds`` of each other. Expected (planned) transients are not
        used as correlation evidence.

        Returns:
            New event objects with ``flight_phase`` and ``correlated_with`` set.
        """
        window = self.config.correlation_window_seconds
        enriched: list[AnomalyEvent] = []
        for event in events:
            start = event.mission_elapsed_s or 0.0
            end = start + event.duration_s
            correlated = [
                other.id
                for other in events
                if other.id != event.id
                and other.sensor_id != event.sensor_id
                and not other.expected
                and not event.expected
                and (other.mission_elapsed_s or 0.0) <= end + window
                and (other.mission_elapsed_s or 0.0) + other.duration_s >= start - window
            ]
            enriched.append(
                event.model_copy(update={"flight_phase": self.phase_for(event.mission_elapsed_s), "correlated_with": correlated})
            )
        return enriched

    @staticmethod
    def correlation_groups(events: list[AnomalyEvent]) -> list[list[AnomalyEvent]]:
        """Group anomalies into connected components of the correlation graph, ordered by time."""
        by_id = {e.id: e for e in events}
        seen: set[str] = set()
        groups: list[list[AnomalyEvent]] = []
        for event in sorted(events, key=lambda e: e.timestamp):
            if event.id in seen:
                continue
            stack, group = [event.id], []
            while stack:
                current = stack.pop()
                if current in seen or current not in by_id:
                    continue
                seen.add(current)
                group.append(by_id[current])
                stack.extend(by_id[current].correlated_with)
            groups.append(sorted(group, key=lambda e: e.timestamp))
        return groups

    @staticmethod
    def mission_start(readings: Sequence[TelemetryReading]) -> datetime:
        """Earliest timestamp in the dataset (MET zero).

        Raises:
            ValueError: If ``readings`` is empty.
        """
        if not readings:
            raise ValueError("No readings")
        return min(r.timestamp for r in readings)
