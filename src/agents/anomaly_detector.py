"""Statistical anomaly detection agent with local-LLM triage.

Detection runs per sensor and combines four independent detectors:

* **z-score** - robust z-score of the residual against a centred rolling-median trend
  (so legitimate ascent trends are not flagged).
* **moving average** - deviation from the trailing moving average, measured in units
  of the trailing window's own standard deviation (floored at the sensor noise level).
* **derivative** - robust z-score of the first time derivative against its rolling
  median; catches steps and spikes that are small in amplitude but abrupt.
* **limits** - hard engineering limits from ``agents.yaml``; any breach is CRITICAL.

Flagged samples are merged into contiguous :class:`AnomalyEvent` intervals, scored in
sigma units, and graded CRITICAL / WARNING / INFO. Detections overlapping planned
transients (engine start, throttle changes, MECO) are downgraded to INFO and marked
``expected``. Finally, events are triaged by the local Ollama model (falling back to
Claude, then to deterministic rules).
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.models.telemetry_schema import AnomalyEvent, Severity, TelemetryReading, anomaly_labels
from src.utils.helpers import (
    LLMRouter,
    LLMUnavailableError,
    format_met,
    readings_to_frame,
    resolve_anomaly_reference,
)
from src.utils.logger import get_logger

logger = get_logger("agents.anomaly_detector")

CATEGORIES = (
    "sensor_fault", "propulsion", "pressurisation", "structural", "thermal",
    "electrical", "gnc", "expected_transient", "unknown",
)

CLASSIFIER_SYSTEM_PROMPT = """You are a launch-vehicle telemetry triage engineer working a flight anomaly review.
For each statistically detected anomaly decide what kind of event it most likely is.

Guidance:
- A very short (1-3 sample) spike on one channel with no correlated anomalies on other sensors is usually an
  instrumentation glitch (category "sensor_fault").
- Anomalies on several sensors within a few seconds of each other usually share a physical cause; classify each
  by the subsystem it is measured on and mention the correlation in the assessment.
- Events flagged expected_transient=true coincide with planned events (engine start, throttling, cutoff).
- Hard-limit breaches ("limit" in methods) are safety relevant; never downgrade them below WARNING.
- Only use severity_override when the statistical severity is clearly wrong; otherwise use null.
- Identify each anomaly by copying its "anomaly" label exactly as given.

Allowed categories: sensor_fault, propulsion, pressurisation, structural, thermal, electrical, gnc,
expected_transient, unknown.

Return JSON only, exactly in this shape:
{"classifications": [{"anomaly": "<anomaly label, copied exactly>", "category": "<category>", "subsystem": "<short subsystem name>",
  "assessment": "<one or two sentences>", "confidence": <0.0-1.0>, "severity_override": null}]}"""


@dataclass(slots=True)
class PlannedEvent:
    """A planned transient during which detections are considered expected."""

    name: str
    start: float
    end: float
    sensors: list[str] = field(default_factory=lambda: ["*"])

    def covers(self, sensor_id: str, start: float, end: float) -> bool:
        """Return whether this event overlaps ``[start, end]`` for ``sensor_id``."""
        if end < self.start or start > self.end:
            return False
        return any(fnmatch.fnmatch(sensor_id, pattern) for pattern in self.sensors)


@dataclass(slots=True)
class DetectorConfig:
    """Tuning parameters for :class:`AnomalyDetector`."""

    window_seconds: float = 10.0
    zscore_threshold: float = 5.0
    derivative_threshold: float = 6.0
    moving_average_threshold: float = 6.0
    info_score: float = 5.0
    warning_score: float = 8.0
    critical_score: float = 14.0
    merge_gap_seconds: float = 2.0
    min_points: int = 20
    max_readings_per_event: int = 40
    provider_preference: str = "local"
    llm_temperature: float = 0.0
    #: Walk back from the first flagged sample to where the deviation actually began.
    onset_backtrack: bool = True
    #: Deviation (in noise sigmas, on a ~1 s median-smoothed signal) that still counts as "already deviating".
    onset_threshold_sigma: float = 3.0
    #: Anomalies first detected before this mission time are suppressed (and logged) as startup transients.
    startup_ignore_seconds: float = 5.0
    #: An anomaly is "transient" if the ~1 s smoothed signal returns within 1 sigma of the pre-onset
    #: baseline (sigma of the window before onset) within this many seconds of onset; else "sustained".
    recovery_window_seconds: float = 10.0
    sensor_limits: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    planned_events: list[PlannedEvent] = field(default_factory=list)
    #: (sensor glob, minimum relative change in %, minimum severity) - first matching pattern applies.
    severity_floors: list[tuple[str, float, Severity]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> DetectorConfig:
        """Build a config from the ``anomaly_detector`` section of ``agents.yaml``."""
        data = dict(data or {})
        limits = {
            sensor: (spec.get("low"), spec.get("high"))
            for sensor, spec in (data.pop("sensor_limits", None) or {}).items()
            if isinstance(spec, dict)
        }
        planned = [
            PlannedEvent(
                name=str(e["name"]), start=float(e["start"]), end=float(e["end"]),
                sensors=list(e.get("sensors", ["*"])),
            )
            for e in (data.pop("planned_events", None) or [])
        ]
        floors = [
            (str(f["pattern"]).lower(), float(f["min_change_pct"]), Severity(str(f.get("severity", "WARNING")).upper()))
            for f in (data.pop("severity_floors", None) or [])
        ]
        known = {f for f in cls.__dataclass_fields__ if f not in {"sensor_limits", "planned_events", "severity_floors"}}
        unknown = set(data) - known
        if unknown:
            logger.warning("Ignoring unknown anomaly_detector options: %s", sorted(unknown))
        return cls(**{k: v for k, v in data.items() if k in known}, sensor_limits=limits, planned_events=planned,
                   severity_floors=floors)


@dataclass(slots=True)
class _SensorScores:
    """Per-sample detector outputs for one sensor."""

    met: np.ndarray
    values: np.ndarray
    zscore: np.ndarray
    derivative: np.ndarray
    moving_average: np.ndarray
    limit_breach: np.ndarray
    noise_sigma: float
    dt: float
    #: Robust z-score of the ~1 s averaged rate of change (vs. its rolling median); used for onset backtracking.
    slope_z: np.ndarray


class AnomalyDetector:
    """Detect, grade and triage anomalies in telemetry."""

    def __init__(self, config: DetectorConfig | None = None, router: LLMRouter | None = None, use_llm: bool = True) -> None:
        """Create the agent.

        Args:
            config: Detection parameters.
            router: LLM router for triage; ``None`` uses rule-based triage only.
            use_llm: Disable LLM triage even if a router is supplied.
        """
        self.config = config or DetectorConfig()
        self.router = router
        self.use_llm = use_llm and router is not None
        self.last_classifier: str = "none"
        self.last_suppressed: list[str] = []

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(self, readings: Sequence[TelemetryReading], mission_start: datetime | None = None) -> list[AnomalyEvent]:
        """Run statistical detection over all sensors.

        Args:
            readings: Telemetry samples (any order, any number of sensors).
            mission_start: Reference for mission-elapsed time; defaults to the earliest sample.

        Returns:
            Anomaly events sorted by time.
        """
        if not readings:
            logger.warning("No readings supplied to anomaly detector")
            return []
        self.last_suppressed = []
        frame = readings_to_frame(readings)
        t0 = pd.Timestamp(mission_start) if mission_start is not None else frame["timestamp"].min()
        if t0.tzinfo is None:
            t0 = t0.tz_localize("UTC")
        events: list[AnomalyEvent] = []
        for sensor_id, group in frame.groupby("sensor_id", sort=True):
            group = group.drop_duplicates(subset="timestamp", keep="last")
            if len(group) < self.config.min_points:
                logger.info("Skipping %s: only %d samples (< %d)", sensor_id, len(group), self.config.min_points)
                continue
            try:
                events.extend(self._detect_sensor(str(sensor_id), group, t0, readings))
            except (ValueError, FloatingPointError) as exc:
                logger.error("Detection failed for sensor %s: %s", sensor_id, exc)
        events.sort(key=lambda e: (e.timestamp, e.sensor_id))
        counts = {s.value: sum(e.severity == s for e in events) for s in Severity}
        logger.info("Detected %d anomalies %s", len(events), counts)
        return events

    def _score_sensor(self, group: pd.DataFrame, t0: pd.Timestamp, sensor_id: str) -> _SensorScores:
        met = (group["timestamp"] - t0).dt.total_seconds().to_numpy(dtype=float)
        values = group["value"].to_numpy(dtype=float)
        diffs = np.diff(met)
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        window = max(5, int(round(self.config.window_seconds / dt)))
        window += 1 - window % 2  # odd window for a symmetric centred median

        series = pd.Series(values)
        trend = series.rolling(window, center=True, min_periods=1).median().to_numpy()
        residual = values - trend
        noise_sigma = _robust_sigma(residual)
        zscore = (residual - np.median(residual)) / noise_sigma

        derivative = np.gradient(values, met) if values.size > 1 else np.zeros_like(values)
        d_trend = pd.Series(derivative).rolling(window, center=True, min_periods=1).median().to_numpy()
        d_residual = derivative - d_trend
        d_sigma = _robust_sigma(d_residual)
        d_score = (d_residual - np.median(d_residual)) / d_sigma

        min_periods = max(3, window // 2)
        trailing_mean = series.rolling(window, min_periods=min_periods).mean().shift(1).to_numpy()
        trailing_std = series.rolling(window, min_periods=min_periods).std().shift(1).to_numpy()
        sigma = np.where(np.isfinite(trailing_std), np.maximum(trailing_std, noise_sigma), np.nan)
        ma_score = np.where(np.isfinite(trailing_mean) & np.isfinite(sigma), (values - trailing_mean) / sigma, 0.0)

        low, high = self.config.sensor_limits.get(sensor_id, (None, None))
        breach = np.zeros(values.size, dtype=bool)
        if low is not None:
            breach |= values < float(low)
        if high is not None:
            breach |= values > float(high)

        # ~1 s averaged slope: noise averages out, sustained ramps stand out early, and a single-sample
        # spike's up/down slopes cancel (so spikes are not backtracked).
        k = max(3, int(round(1.0 / dt)))
        slope = pd.Series(derivative).rolling(k, center=True, min_periods=1).mean().to_numpy()
        slope_residual = slope - pd.Series(slope).rolling(window, center=True, min_periods=1).median().to_numpy()
        slope_z = (slope_residual - np.median(slope_residual)) / _robust_sigma(slope_residual)

        return _SensorScores(met, values, zscore, d_score, ma_score, breach, noise_sigma, dt, slope_z)

    def _detect_sensor(
        self,
        sensor_id: str,
        group: pd.DataFrame,
        t0: pd.Timestamp,
        readings: Sequence[TelemetryReading],
    ) -> list[AnomalyEvent]:
        cfg = self.config
        scores = self._score_sensor(group, t0, sensor_id)
        z_flag = np.abs(scores.zscore) >= cfg.zscore_threshold
        amplitude_flag = z_flag | scores.limit_breach
        d_flag = np.abs(scores.derivative) >= cfg.derivative_threshold
        m_flag = np.abs(scores.moving_average) >= cfg.moving_average_threshold
        flagged = z_flag | d_flag | m_flag | scores.limit_breach
        if not flagged.any():
            return []

        # Normalised per-sample score in sigma units (max over detectors).
        combined = np.maximum.reduce([
            np.abs(scores.zscore),
            np.abs(scores.derivative) * (cfg.zscore_threshold / cfg.derivative_threshold),
            np.abs(scores.moving_average) * (cfg.zscore_threshold / cfg.moving_average_threshold),
        ])
        unit = str(group["unit"].iloc[0]) if "unit" in group else ""
        smooth_k = max(3, int(round(1.0 / max(scores.dt, 1e-6))))
        smoothed = pd.Series(scores.values).rolling(smooth_k, center=True, min_periods=1).median().to_numpy()
        original_index = group["index"].to_numpy()
        timestamps = group["timestamp"].reset_index(drop=True)

        events: list[AnomalyEvent] = []
        for start, end in _segments(np.flatnonzero(flagged), scores.met, cfg.merge_gap_seconds):
            sl = slice(start, end + 1)
            peak = start + int(np.argmax(combined[sl]))
            score = float(combined[peak])
            methods = [
                name for name, mask in (
                    ("zscore", z_flag), ("derivative", d_flag), ("moving_average", m_flag),
                    ("limit", scores.limit_breach),
                ) if mask[sl].any()
            ]
            limit_breached = "limit" in methods
            if limit_breached:
                breach_idx = start + int(np.flatnonzero(scores.limit_breach[sl])[0])
                peak = breach_idx if score < cfg.critical_score else peak
                score = max(score, cfg.critical_score)
            if score < cfg.info_score and not limit_breached:
                continue
            detected_at = float(scores.met[start])
            if detected_at < cfg.startup_ignore_seconds:
                self.last_suppressed.append(f"{sensor_id} first detected at {format_met(detected_at)} (score {score:.1f})")
                continue

            onset = self._onset_index(scores, start) if cfg.onset_backtrack else start
            baseline = _baseline(scores.values, onset, scores.dt, cfg.window_seconds)
            met_start, met_end = float(scores.met[onset]), float(scores.met[end])
            peak_value = float(scores.values[peak])
            severity = (
                Severity.CRITICAL if score >= cfg.critical_score or limit_breached
                else Severity.WARNING if score >= cfg.warning_score
                else Severity.INFO
            )
            planned = next((p for p in cfg.planned_events if p.covers(sensor_id, met_start, met_end)), None)
            expected = planned is not None and not limit_breached
            floor_note = ""
            if expected:
                severity = Severity.INFO
            else:
                floor = self._severity_floor(sensor_id, baseline, peak_value)
                if floor is not None and floor[0].rank > severity.rank:
                    floor_note = (f" Severity raised {severity.value} -> {floor[0].value} by the {floor[1]} floor "
                                  f"({floor[2]:.1f}% change).")
                    severity = floor[0]

            description = _describe(
                sensor_id, unit, baseline, peak_value, float(scores.met[peak]), met_start, met_end, methods,
                score, scores.dt, cfg.sensor_limits.get(sensor_id), limit_breached,
            )
            recovery, recovery_s, settled = self._recovery(scores, smoothed, onset, peak, baseline)
            if recovery == "transient":
                description += f" Transient: recovered within {recovery_s:.1f} s."
            else:
                description += (f" Sustained: value remained at {settled:.4g} {unit} {cfg.recovery_window_seconds:g} s "
                                f"after onset (baseline {baseline:.4g} {unit}).")
            if onset < start:
                description += f" Onset backtracked to {format_met(met_start)} (first flagged at {format_met(detected_at)})."
            description += floor_note
            if expected and planned is not None:
                description += f" Coincides with planned event '{planned.name}' - treated as expected transient."

            pad = max(3, int(round(1.0 / scores.dt)))
            idx_range = range(max(0, onset - pad), min(len(original_index), end + pad + 1))
            if len(idx_range) > cfg.max_readings_per_event:
                step = int(np.ceil(len(idx_range) / cfg.max_readings_per_event))
                idx_range = list(idx_range)[::step]
                if peak not in idx_range:
                    idx_range = sorted({*idx_range, peak})
            event_readings = [readings[int(original_index[i])] for i in idx_range]

            start_ts, end_ts = timestamps.iloc[onset], timestamps.iloc[end]
            events.append(
                AnomalyEvent(
                    id=_event_id(sensor_id, start_ts.isoformat()),
                    timestamp=start_ts.to_pydatetime(),
                    end_timestamp=end_ts.to_pydatetime(),
                    sensor_id=sensor_id,
                    severity=severity,
                    description=description,
                    readings=event_readings,
                    unit=unit,
                    mission_elapsed_s=met_start,
                    detected_at_s=detected_at,
                    recovery=recovery,
                    recovery_s=recovery_s,
                    settled_value=settled,
                    duration_s=max(met_end - met_start, 0.0),
                    peak_value=peak_value,
                    baseline_value=baseline,
                    score=round(score, 2),
                    detection_methods=methods,
                    consecutive_samples=_longest_run(amplitude_flag[sl]),
                    expected=expected,
                )
            )
        return events

    def _onset_index(self, scores: _SensorScores, start: int) -> int:
        """First sample of the run leading into ``start`` that already crossed the (lower) onset threshold.

        Uses the detector's own trend-compensated scores: walking back from the first flagged
        sample, a sample still belongs to the event while its residual z-score, its derivative
        score or its ~1 s averaged slope score is at least ``onset_threshold_sigma`` in the
        event's direction. Never later than ``start``; looks back at most one detection window.
        """
        cfg = self.config
        window = max(5, int(round(cfg.window_seconds / max(scores.dt, 1e-6))))
        lowest = max(0, start - window)
        direction = np.sign(scores.zscore[start]) or np.sign(scores.derivative[start]) or 1.0
        threshold = cfg.onset_threshold_sigma

        def walk(*series: np.ndarray) -> int:
            i = start
            while i > lowest and any(direction * x[i - 1] >= threshold for x in series):
                i -= 1
            return i

        sample_level = walk(scores.zscore, scores.derivative)
        # The ~1 s centred slope average responds half a window before an abrupt change; shift its
        # result forward by that much so a sharp ramp is not dated earlier than it really started.
        half = max(3, int(round(1.0 / max(scores.dt, 1e-6)))) // 2
        smoothed = min(start, walk(scores.slope_z) + half)
        return min(sample_level, smoothed)

    def _recovery(
        self, scores: _SensorScores, smoothed: np.ndarray, onset: int, peak: int, baseline: float
    ) -> tuple[str, float | None, float | None]:
        """Classify an event as transient (with recovery time) or sustained (with settled value).

        "Baseline" is the pre-onset trend: a line fitted to the window before onset and extrapolated,
        so a signal that returns to its normal slow drift counts as recovered. Its 1 sigma is the
        scatter of the pre-onset samples around that line (never below the sensor noise).
        """
        cfg = self.config
        window = max(5, int(round(cfg.window_seconds / max(scores.dt, 1e-6))))
        lo = max(0, onset - window)
        pre_t, pre_v = scores.met[lo:onset], scores.values[lo:onset]
        if pre_v.size >= 5:
            slope, intercept = np.polyfit(pre_t, pre_v, 1)
            sigma = max(_robust_sigma(pre_v - (slope * pre_t + intercept)), scores.noise_sigma)
        else:
            slope, intercept, sigma = 0.0, baseline, scores.noise_sigma
        horizon = scores.met[onset] + cfg.recovery_window_seconds
        after = np.flatnonzero((scores.met > scores.met[peak]) & (scores.met <= horizon))
        for j in after:
            if abs(smoothed[j] - (slope * scores.met[j] + intercept)) <= sigma:
                return "transient", float(scores.met[j] - scores.met[onset]), None
        last = after[-1] if after.size else peak
        return "sustained", None, float(smoothed[last])

    def _severity_floor(self, sensor_id: str, baseline: float, peak: float) -> tuple[Severity, str, float] | None:
        """Minimum severity for a relative change on this sensor type, from ``severity_floors``."""
        if abs(baseline) < 1e-9:
            return None
        change = abs(peak - baseline) / abs(baseline) * 100.0
        for pattern, minimum, severity in self.config.severity_floors:
            if fnmatch.fnmatch(sensor_id.lower(), pattern):
                return (severity, f"'{pattern}' >{minimum:g}%", change) if change > minimum else None
        return None

    # ------------------------------------------------------------------ #
    # Triage / classification
    # ------------------------------------------------------------------ #
    def classify(self, events: list[AnomalyEvent]) -> list[AnomalyEvent]:
        """Attach a classification to each event (LLM first, rules as fallback).

        Correlation (``correlated_with``) and flight phase should already be set by the
        telemetry analyst so the triage can use them.

        Returns:
            New event objects with ``classification`` populated and severity possibly adjusted.
        """
        if not events:
            return []
        rule_based = {e.id: self._rule_classification(e) for e in events}
        llm_results: dict[str, dict[str, Any]] = {}
        self.last_classifier = "rule-based"
        if self.use_llm and self.router is not None:
            try:
                llm_results = self._llm_classifications(events)
            except LLMUnavailableError as exc:
                logger.warning("LLM triage unavailable, using rule-based classification: %s", exc)

        classified: list[AnomalyEvent] = []
        for event in events:
            rule = rule_based[event.id]
            llm = llm_results.get(event.id, {})
            result = {**rule, **{k: v for k, v in llm.items() if k != "severity_override"}}
            # The LLM only overrides severity when it explicitly asks for a *different* level;
            # otherwise the rule-based adjustment (e.g. sensor-glitch downgrade) still applies.
            llm_override = _normalise_override(llm.get("severity_override"))
            if llm_override == event.severity.value:
                llm_override = None
            override, adjusted_by = (
                (llm_override, llm.get("source", "llm")) if llm_override else (rule.get("severity_override"), "rule-based")
            )
            result.pop("severity_override", None)
            severity = self._apply_override(event, override, result, adjusted_by)
            classified.append(event.model_copy(update={"classification": result, "severity": severity}))
        return classified

    def run(self, readings: Sequence[TelemetryReading]) -> list[AnomalyEvent]:
        """Detect and classify anomalies in one call."""
        return self.classify(self.detect(readings))

    def _apply_override(self, event: AnomalyEvent, override: Any, result: dict[str, Any], adjusted_by: str) -> Severity:
        if not override:
            return event.severity
        try:
            new = Severity(str(override).upper())
        except ValueError:
            logger.debug("Ignoring invalid severity override %r for %s", override, event.id)
            return event.severity
        if "limit" in event.detection_methods and new == Severity.INFO:
            new = Severity.WARNING
        if new != event.severity:
            logger.info(
                "%s [%s] severity %s -> %s (%s: %s)", event.id, event.label, event.severity.value, new.value,
                adjusted_by, result.get("assessment", ""),
            )
            result["original_severity"] = event.severity.value
            result["severity_adjusted_by"] = adjusted_by
        return new

    def _rule_classification(self, event: AnomalyEvent) -> dict[str, Any]:
        sensor = event.sensor_id.lower()
        subsystem = _subsystem_for(sensor)
        samples = len([r for r in event.readings if event.timestamp <= r.timestamp <= (event.end_timestamp or event.timestamp)])
        short = samples <= 3 or event.duration_s <= 0.3
        result: dict[str, Any] = {"subsystem": subsystem, "source": "rule-based", "severity_override": None}
        if event.expected:
            result.update(category="expected_transient", confidence=0.8,
                          assessment="Deviation coincides with a planned vehicle event.")
        elif short and not event.correlated_with and "limit" not in event.detection_methods:
            result.update(
                category="sensor_fault", confidence=0.7,
                assessment="Very short single-channel spike with no corroborating sensors - likely instrumentation glitch.",
                severity_override=Severity.WARNING.value if event.severity == Severity.CRITICAL else None,
            )
        else:
            category = _category_for(sensor)
            correlated = f" Correlated with {len(event.correlated_with)} other anomaly(ies)." if event.correlated_with else ""
            result.update(
                category=category, confidence=0.6 if event.correlated_with else 0.5,
                assessment=f"Sustained deviation on {subsystem} channel.{correlated}",
            )
        return result

    def _llm_classifications(self, events: list[AnomalyEvent]) -> dict[str, dict[str, Any]]:
        assert self.router is not None
        results: dict[str, dict[str, Any]] = {}
        prefer = "cloud" if self.config.provider_preference == "cloud" else "local"
        labels = anomaly_labels(events)
        for batch_start in range(0, len(events), 15):
            batch = events[batch_start:batch_start + 15]
            prompt = (
                "Classify these telemetry anomalies. Mission elapsed time (met_s) is seconds from first sample.\n\n"
                + json.dumps([e.to_prompt_dict(labels) for e in batch], indent=1, default=str)
            )
            document, response = self.router.complete_json(
                CLASSIFIER_SYSTEM_PROMPT, prompt, prefer=prefer, temperature=self.config.llm_temperature
            )
            self.last_classifier = f"{response.provider}:{response.model}"
            items = document.get("classifications", []) if isinstance(document, dict) else document
            if not isinstance(items, list):
                raise LLMUnavailableError("Classifier returned unexpected JSON structure")
            for item in items:
                if not isinstance(item, dict):
                    continue
                event = resolve_anomaly_reference(item.get("anomaly", item.get("id")), batch)
                if event is None:
                    logger.warning("Triage output references unknown anomaly %r - ignored", item.get("anomaly"))
                    continue
                category = str(item.get("category", "unknown")).lower()
                results[event.id] = {
                    "category": category if category in CATEGORIES else "unknown",
                    "subsystem": str(item.get("subsystem") or "unknown"),
                    "assessment": str(item.get("assessment") or ""),
                    "confidence": _safe_float(item.get("confidence"), 0.5),
                    "severity_override": item.get("severity_override"),
                    "source": self.last_classifier,
                }
        logger.info("LLM triage classified %d/%d anomalies via %s", len(results), len(events), self.last_classifier)
        return results


# ---------------------------------------------------------------------- #
# Module helpers
# ---------------------------------------------------------------------- #
def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest run of ``True`` values."""
    longest = current = 0
    for flag in mask:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return longest


def _robust_sigma(values: np.ndarray) -> float:
    """MAD-based sigma estimate with sensible fallbacks."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 1.0
    sigma = 1.4826 * float(np.median(np.abs(finite - np.median(finite))))
    if sigma < 1e-9:
        sigma = float(np.std(finite))
    return sigma if sigma > 1e-9 else 1e-6


def _segments(indices: np.ndarray, met: np.ndarray, merge_gap: float) -> list[tuple[int, int]]:
    """Group flagged sample indices into contiguous segments separated by > ``merge_gap`` seconds."""
    if indices.size == 0:
        return []
    segments: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for idx in indices[1:]:
        idx = int(idx)
        if met[idx] - met[prev] > merge_gap:
            segments.append((start, prev))
            start = idx
        prev = idx
    segments.append((start, prev))
    return segments


def _baseline(values: np.ndarray, start: int, dt: float, window_seconds: float) -> float:
    """Median of the samples in the window preceding ``start``."""
    n = max(3, int(round(window_seconds / max(dt, 1e-6))))
    pre = values[max(0, start - n):start]
    return float(np.median(pre)) if pre.size else float(values[start])


def _describe(
    sensor_id: str, unit: str, baseline: float, peak: float, peak_met: float, start: float, end: float,
    methods: list[str], score: float, dt: float, limits: tuple[float | None, float | None] | None, breached: bool,
) -> str:
    delta = peak - baseline
    pct = f" ({delta / baseline * 100:+.1f}%)" if abs(baseline) > 1e-9 else ""
    direction = "rose" if delta > 0 else "dropped"
    duration = max(end - start, dt)
    text = (
        f"{sensor_id} {direction} {abs(delta):.3g} {unit}{pct} from a baseline of {baseline:.4g} to "
        f"{peak:.4g} {unit} at {format_met(peak_met)}; anomalous interval {format_met(start)} to "
        f"{format_met(end)} (~{duration:.1f}s). Peak deviation {score:.1f} sigma; detected by {', '.join(methods)}."
    )
    if breached and limits:
        low, high = limits
        text += f" Hard limit breached (allowed range {low} to {high} {unit})."
    return text


def _event_id(sensor_id: str, start_iso: str) -> str:
    """Deterministic anomaly ID so re-runs on the same data map to the same IDs."""
    return "ANM-" + hashlib.sha1(f"{sensor_id}|{start_iso}".encode()).hexdigest()[:8].upper()


def _subsystem_for(sensor: str) -> str:
    if "chamber" in sensor or "injector" in sensor or "thrust" in sensor:
        return "main engine combustion"
    if "tank" in sensor or "ullage" in sensor or "pressurant" in sensor:
        return "propellant tank pressurisation"
    if "pump" in sensor or "turbo" in sensor:
        return "turbomachinery"
    if "vib" in sensor or "accel" in sensor:
        return "structures / vibration"
    if "temp" in sensor or "thermo" in sensor:
        return "thermal"
    if "volt" in sensor or "current" in sensor or "bus" in sensor or "batt" in sensor:
        return "electrical power"
    if "gyro" in sensor or "attitude" in sensor or "imu" in sensor:
        return "guidance, navigation & control"
    return "unknown"


def _category_for(sensor: str) -> str:
    if "tank" in sensor or "ullage" in sensor or "pressurant" in sensor:
        return "pressurisation"
    if "chamber" in sensor or "pump" in sensor or "turbo" in sensor or "injector" in sensor:
        return "propulsion"
    if "vib" in sensor or "accel" in sensor or "strain" in sensor:
        return "structural"
    if "temp" in sensor:
        return "thermal"
    if "volt" in sensor or "current" in sensor or "bus" in sensor:
        return "electrical"
    if "gyro" in sensor or "attitude" in sensor or "imu" in sensor:
        return "gnc"
    return "unknown"


def _normalise_override(value: Any) -> str | None:
    """Return an upper-case severity name, or ``None`` for null-like LLM values."""
    if value is None:
        return None
    text = str(value).strip().upper()
    return text if text in {s.value for s in Severity} else None


def _safe_float(value: Any, default: float) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return default
