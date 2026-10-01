"""Sensor-fault classifier: separates instrumentation glitches from real physical events.

Runs after triage and before hypothesis generation. It is deterministic on purpose: the
local LLM triage sometimes labels obvious single-sample spikes as physical events (e.g.
"pressurisation"), which leaves them unexplained by any root-cause hypothesis.

Rules, applied to every non-expected anomaly:

* **sensor_fault** - the amplitude exceeded the threshold for at most ``max_spike_samples``
  consecutive samples *and* the signal returned to its baseline straight afterwards
  (no sustained deviation). A hard-limit breach is never treated as a sensor fault.
* **real_event** - more than ``min_real_event_samples - 1`` (default: more than 3)
  consecutive samples exceeded the threshold.
* **indeterminate** - anything in between; the triage classification is kept.

The verdict is stored as ``classification["fault_class"]``; the hypothesis generator uses
it to explain sensor faults with a dedicated "Instrumentation Glitch" hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents.anomaly_detector import _category_for
from src.models.telemetry_schema import AnomalyEvent, Severity
from src.utils.logger import get_logger

logger = get_logger("agents.sensor_fault_classifier")

SENSOR_FAULT = "sensor_fault"
REAL_EVENT = "real_event"
INDETERMINATE = "indeterminate"


@dataclass(slots=True)
class SensorFaultConfig:
    """Thresholds for :class:`SensorFaultClassifier`."""

    max_spike_samples: int = 1
    min_real_event_samples: int = 4
    recovery_window_s: float = 1.0
    recovery_ratio: float = 0.25

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SensorFaultConfig:
        """Build from the ``sensor_fault_classifier`` section of ``agents.yaml``."""
        data = data or {}
        return cls(
            max_spike_samples=int(data.get("max_spike_samples", 1)),
            min_real_event_samples=int(data.get("min_real_event_samples", 4)),
            recovery_window_s=float(data.get("recovery_window_s", 1.0)),
            recovery_ratio=float(data.get("recovery_ratio", 0.25)),
        )


class SensorFaultClassifier:
    """Classify anomalies as sensor faults or real events from their signal shape."""

    def __init__(self, config: SensorFaultConfig | None = None) -> None:
        """Create the classifier with the given thresholds (defaults if omitted)."""
        self.config = config or SensorFaultConfig()

    def verdict(self, event: AnomalyEvent) -> tuple[str, str]:
        """Return ``(fault_class, reason)`` for one anomaly."""
        cfg = self.config
        run = event.consecutive_samples
        if run >= cfg.min_real_event_samples:
            return REAL_EVENT, f"{run} consecutive samples exceeded the detection threshold"
        if "limit" in event.detection_methods:
            return REAL_EVENT, "hard engineering limit breached"
        if run <= cfg.max_spike_samples:
            sustained = self._sustained_deviation(event)
            if sustained is None:
                return INDETERMINATE, f"{run}-sample excursion, but no post-event samples to check recovery"
            if not sustained:
                return SENSOR_FAULT, (
                    f"single-sample spike ({run} sample above threshold) that returned to baseline "
                    f"within {cfg.recovery_window_s:g}s - no sustained deviation"
                )
            return INDETERMINATE, f"{run}-sample excursion followed by a sustained deviation"
        return INDETERMINATE, f"{run} consecutive samples exceeded the threshold (between spike and real-event limits)"

    def _sustained_deviation(self, event: AnomalyEvent) -> bool | None:
        """Whether the signal stays away from baseline after the peak (``None`` if unknown)."""
        if event.peak_value is None or event.baseline_value is None:
            return None
        peak_deviation = abs(event.peak_value - event.baseline_value)
        if peak_deviation == 0:
            return None
        peak_reading = next((r for r in event.readings if r.value == event.peak_value), None)
        if peak_reading is None:
            return None
        after = [
            abs(r.value - event.baseline_value)
            for r in event.readings
            if 0 < (r.timestamp - peak_reading.timestamp).total_seconds() <= self.config.recovery_window_s
        ]
        # Ignore the sample right after the peak: it can carry a derivative/interpolation artefact.
        after = after[1:] if len(after) > 1 else after
        if not after:
            return None
        return float(np.median(after)) > self.config.recovery_ratio * peak_deviation

    def classify(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """Attach ``fault_class`` to every non-expected anomaly.

        Sensor faults get category ``sensor_fault`` (overriding the triage label, which is kept
        as ``triage_category``) and are capped at WARNING severity. Real events that triage
        mislabelled as ``sensor_fault`` are moved back to their physical category.
        """
        updated: list[AnomalyEvent] = []
        for event in events:
            if event.expected:
                updated.append(event)
                continue
            fault_class, reason = self.verdict(event)
            classification = {**event.classification, "fault_class": fault_class, "fault_class_reason": reason}
            severity = event.severity
            triage_category = classification.get("category")
            if fault_class == SENSOR_FAULT:
                if triage_category not in (None, SENSOR_FAULT):
                    classification["triage_category"] = triage_category
                    logger.info("%s reclassified %s -> sensor_fault (%s)", event.label, triage_category, reason)
                classification.update(category=SENSOR_FAULT, subsystem="instrumentation")
                if severity == Severity.CRITICAL:
                    classification.setdefault("original_severity", severity.value)
                    classification["severity_adjusted_by"] = "sensor_fault_classifier"
                    severity = Severity.WARNING
            elif fault_class == REAL_EVENT and triage_category == SENSOR_FAULT:
                classification["triage_category"] = triage_category
                classification["category"] = _category_for(event.sensor_id.lower())
                logger.info("%s reclassified sensor_fault -> %s (%s)", event.label, classification["category"], reason)
            updated.append(event.model_copy(update={"classification": classification, "severity": severity}))
        counts = {c: sum(e.classification.get("fault_class") == c for e in updated) for c in (SENSOR_FAULT, REAL_EVENT, INDETERMINATE)}
        logger.info("Sensor-fault screen: %s", counts)
        return updated
