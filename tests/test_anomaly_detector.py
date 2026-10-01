"""Tests for ingestion, the anomaly detector, triage and LLM fallback routing."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.agents.anomaly_detector import AnomalyDetector, DetectorConfig, PlannedEvent
from src.agents.sensor_fault_classifier import SensorFaultClassifier
from src.agents.telemetry_analyst import AnalystConfig, TelemetryAnalyst
from src.ingestion import load_telemetry, write_sample_files
from src.ingestion.csv_parser import INJECTED_ANOMALIES, split_sensor_header
from src.models.telemetry_schema import Severity, TelemetryReading
from src.utils.helpers import LLMRouter, LLMUnavailableError, extract_json, normalize_timestamp

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE = datetime(2026, 1, 1, tzinfo=UTC)


class FakeClient:
    """Stand-in for OllamaClient / ClaudeClient."""

    def __init__(self, model: str, response: str | None = None, up: bool = True, fail: bool = False) -> None:
        self.model = model
        self.response = response
        self.up = up
        self.fail = fail
        self.calls = 0

    def available(self) -> bool:
        return self.up

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        self.calls += 1
        if self.fail:
            raise LLMUnavailableError("boom")
        return self.response or "{}"


@pytest.fixture(scope="module")
def agents_config() -> dict:
    with (PROJECT_ROOT / "config" / "agents.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def sample_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_sample_files(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="module")
def readings(sample_files: dict[str, Path]) -> list[TelemetryReading]:
    return load_telemetry(sample_files["nasa_long"])


@pytest.fixture(scope="module")
def detector(agents_config: dict) -> AnomalyDetector:
    return AnomalyDetector(DetectorConfig.from_dict(agents_config["anomaly_detector"]), use_llm=False)


def _series(values: np.ndarray, sensor: str = "test_sensor", unit: str = "psi", dt: float = 0.1) -> list[TelemetryReading]:
    return [
        TelemetryReading(timestamp=BASE + timedelta(seconds=i * dt), sensor_id=sensor, value=float(v), unit=unit)
        for i, v in enumerate(values)
    ]


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_both_csv_layouts_parse_identically(sample_files: dict[str, Path]) -> None:
    long_readings = load_telemetry(sample_files["nasa_long"])
    wide_readings = load_telemetry(sample_files["spacex_wide"])
    assert len(long_readings) == len(wide_readings) == 18000
    assert {(r.sensor_id, r.unit) for r in long_readings} == {(r.sensor_id, r.unit) for r in wide_readings}
    assert all(r.timestamp.tzinfo is not None for r in long_readings[:100])


def test_missing_values_are_imputed(readings: list[TelemetryReading]) -> None:
    imputed = [r for r in readings if r.metadata.get("imputed")]
    assert imputed, "sample data contains NaNs that should be interpolated"
    assert all(np.isfinite(r.value) for r in imputed)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-01-01T00:00:10Z", BASE + timedelta(seconds=10)),
        ("T+12.5", BASE + timedelta(seconds=12.5)),
        (3.0, BASE + timedelta(seconds=3)),
        (1767225600, BASE),
        (1767225600000, BASE),
    ],
)
def test_normalize_timestamp(raw: object, expected: datetime) -> None:
    assert normalize_timestamp(raw, BASE) == expected


@pytest.mark.parametrize(
    ("header", "expected"),
    [("Chamber Pressure (psi)", ("chamber_pressure", "psi")), ("bus_voltage_v", ("bus_voltage", "V")),
     ("engine_bay_temp_degC", ("engine_bay_temp", "degC")), ("mystery", ("mystery", ""))],
)
def test_split_sensor_header(header: str, expected: tuple[str, str]) -> None:
    assert split_sensor_header(header) == expected


def test_json_channel_document(tmp_path: Path) -> None:
    doc = {"base_time": "2026-01-01T00:00:00Z", "sensors": {"lox": {"unit": "psi", "samples": [[0, 1.0], [0.1, None], [0.2, 1.2]]}}}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    parsed = load_telemetry(path)
    assert [round(r.value, 3) for r in parsed] == [1.0, 1.1, 1.2]
    assert parsed[1].metadata.get("imputed") is True


def test_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.xlsx"
    path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        load_telemetry(path)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_detects_all_injected_anomalies(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = [e for e in detector.detect(readings) if not e.expected]
    for sensor, met in INJECTED_ANOMALIES.items():
        assert any(e.sensor_id == sensor and abs(e.mission_elapsed_s - met) <= 5 for e in events), sensor


def test_no_high_severity_false_positives(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    for event in detector.detect(readings):
        if event.severity in (Severity.CRITICAL, Severity.WARNING):
            assert any(
                event.sensor_id == s and abs(event.mission_elapsed_s - m) <= 5 for s, m in INJECTED_ANOMALIES.items()
            ), event.description


def test_limit_breach_is_critical(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    voltage = [e for e in detector.detect(readings) if e.sensor_id == "avionics_bus_voltage"]
    assert voltage and voltage[0].severity == Severity.CRITICAL
    assert "limit" in voltage[0].detection_methods


def test_clean_noise_has_no_anomalies() -> None:
    rng = np.random.default_rng(0)
    events = AnomalyDetector(use_llm=False).detect(_series(100 + rng.normal(0, 1, 2000)))
    assert not [e for e in events if e.severity != Severity.INFO]


def test_step_change_detected_with_baseline() -> None:
    rng = np.random.default_rng(1)
    values = 50 + rng.normal(0, 0.1, 1000)
    values[500:] -= 5.0
    events = AnomalyDetector(use_llm=False).detect(_series(values))
    assert events
    event = max(events, key=lambda e: e.score)
    assert abs(event.mission_elapsed_s - 50.0) < 2.0
    assert event.baseline_value == pytest.approx(50.0, abs=0.2)
    assert event.peak_value < event.baseline_value


def test_planned_event_downgraded_to_expected() -> None:
    rng = np.random.default_rng(2)
    values = 10 + rng.normal(0, 0.05, 1000)
    values[300:] += 4.0
    config = DetectorConfig(planned_events=[PlannedEvent("Throttle up", 28.0, 32.0, ["test_*"])])
    events = AnomalyDetector(config, use_llm=False).detect(_series(values))
    assert events and all(e.expected and e.severity == Severity.INFO for e in events)


def test_too_few_points_and_empty_input() -> None:
    detector = AnomalyDetector(use_llm=False)
    assert detector.detect([]) == []
    assert detector.detect(_series(np.ones(5))) == []


def test_event_ids_are_deterministic(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    assert [e.id for e in detector.detect(readings)] == [e.id for e in detector.detect(readings)]


# --------------------------------------------------------------------------- #
# Triage and LLM routing
# --------------------------------------------------------------------------- #
def test_rule_based_triage_flags_sensor_glitch(detector: AnomalyDetector, readings: list[TelemetryReading], agents_config: dict) -> None:
    analyst = TelemetryAnalyst(AnalystConfig.from_dict(agents_config["telemetry_analyst"]))
    events = detector.classify(analyst.contextualize(detector.detect(readings)))
    glitch = next(e for e in events if e.sensor_id == "fuel_tank_pressure" and not e.expected)
    assert glitch.classification["category"] == "sensor_fault"
    assert glitch.severity == Severity.WARNING  # downgraded from CRITICAL
    lox = next(e for e in events if e.sensor_id == "lox_tank_pressure")
    assert lox.correlated_with, "LOX drop should correlate with the vibration and temperature anomalies"
    assert lox.flight_phase == "Mid-burn ascent"


def test_llm_triage_used_and_falls_back_to_cloud(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = [e for e in detector.detect(readings) if not e.expected]
    payload = {"classifications": [
        {"anomaly": e.label, "category": "propulsion", "subsystem": "engine", "assessment": "test", "confidence": 0.9}
        for e in events
    ]}
    local = FakeClient("qwen-test", fail=True)
    cloud = FakeClient("claude-test", response="```json\n" + json.dumps(payload) + "\n```")
    llm_detector = AnomalyDetector(detector.config, router=LLMRouter(local, cloud))  # type: ignore[arg-type]
    classified = llm_detector.classify(events)
    assert local.calls == 1 and cloud.calls == 1
    assert all(e.classification["category"] == "propulsion" for e in classified)
    assert llm_detector.last_classifier == "cloud:claude-test"


def test_triage_falls_back_to_rules_when_no_llm(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = detector.detect(readings)
    router = LLMRouter(FakeClient("q", up=False), FakeClient("c", up=False))  # type: ignore[arg-type]
    classified = AnomalyDetector(detector.config, router=router).classify(events)
    assert all(e.classification["source"] == "rule-based" for e in classified)


def test_llm_cannot_downgrade_limit_breach_to_info(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    voltage = [e for e in detector.detect(readings) if e.sensor_id == "avionics_bus_voltage"]
    payload = {"classifications": [{"anomaly": voltage[0].label, "category": "electrical", "severity_override": "INFO"}]}
    router = LLMRouter(FakeClient("q", response=json.dumps(payload)), None)  # type: ignore[arg-type]
    classified = AnomalyDetector(detector.config, router=router).classify(voltage)
    assert classified[0].severity == Severity.WARNING


def _glitch_triage(detector: AnomalyDetector, readings: list[TelemetryReading], agents_config: dict, override: object) -> AnomalyEvent:
    """Triage the fuel-pressure spike with a fake Qwen that returns ``override``."""
    analyst = TelemetryAnalyst(AnalystConfig.from_dict(agents_config["telemetry_analyst"]))
    events = analyst.contextualize(detector.detect(readings))
    glitch = next(e for e in events if e.sensor_id == "fuel_tank_pressure" and not e.expected)
    payload = {"classifications": [{"anomaly": glitch.label, "category": "sensor_fault", "subsystem": "instrumentation",
                                    "assessment": "spike", "confidence": 0.8, "severity_override": override}]}
    router = LLMRouter(FakeClient("qwen", response=json.dumps(payload)), None)  # type: ignore[arg-type]
    return AnomalyDetector(detector.config, router=router).classify([glitch])[0]


@pytest.mark.parametrize("override", [None, "null", "", "CRITICAL"])
def test_llm_without_explicit_change_keeps_rule_downgrade(
    detector: AnomalyDetector, readings: list[TelemetryReading], agents_config: dict, override: object
) -> None:
    triaged = _glitch_triage(detector, readings, agents_config, override)
    assert triaged.severity == Severity.WARNING
    assert triaged.classification["original_severity"] == "CRITICAL"
    assert triaged.classification["severity_adjusted_by"] == "rule-based"
    assert triaged.classification["source"] == "local:qwen"


def test_llm_explicit_override_wins(detector: AnomalyDetector, readings: list[TelemetryReading], agents_config: dict) -> None:
    triaged = _glitch_triage(detector, readings, agents_config, "INFO")
    assert triaged.severity == Severity.INFO
    assert triaged.classification["severity_adjusted_by"] == "local:qwen"


def test_triage_prompt_uses_labels_not_ids(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = [e for e in detector.detect(readings) if not e.expected]
    client = FakeClient("qwen", response='{"classifications": []}')
    prompts: list[str] = []
    original = client.chat

    def capture(system: str, prompt: str, **kwargs: object) -> str:
        prompts.append(prompt)
        return original(system, prompt, **kwargs)  # type: ignore[arg-type]

    client.chat = capture  # type: ignore[method-assign]
    AnomalyDetector(detector.config, router=LLMRouter(client, None)).classify(events)  # type: ignore[arg-type]
    assert "LOX tank pressure drop (CRITICAL) at T+142.1s" in prompts[0]  # onset, not first-flagged time
    assert "ANM-" not in prompts[0]


def test_extract_json_handles_prose() -> None:
    assert extract_json('Sure! Here you go: {"a": [1, 2]} hope this helps') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")


# --------------------------------------------------------------------------- #
# Fix 6: sensor-fault classifier
# --------------------------------------------------------------------------- #
def test_consecutive_samples_recorded(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = {e.sensor_id: e for e in detector.detect(readings) if not e.expected}
    assert events["fuel_tank_pressure"].consecutive_samples == 1
    for sensor in ("lox_tank_pressure", "turbopump_vibration", "engine_bay_temp", "avionics_bus_voltage"):
        assert events[sensor].consecutive_samples > 3, sensor


def test_sensor_fault_classifier_verdicts(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    classified = SensorFaultClassifier().classify(detector.detect(readings))
    verdicts = {e.sensor_id: e.classification.get("fault_class") for e in classified if not e.expected}
    assert verdicts == {
        "lox_tank_pressure": "real_event", "turbopump_vibration": "real_event", "engine_bay_temp": "real_event",
        "avionics_bus_voltage": "real_event", "fuel_tank_pressure": "sensor_fault",
    }
    assert all("fault_class" not in e.classification for e in classified if e.expected)


def test_classifier_overrides_triage_mislabel(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = detector.detect(readings)
    fuel = next(e for e in events if e.sensor_id == "fuel_tank_pressure")
    lox = next(e for e in events if e.sensor_id == "lox_tank_pressure")
    mislabelled = [
        fuel.model_copy(update={"classification": {"category": "pressurisation"}}),  # what Qwen did
        lox.model_copy(update={"classification": {"category": "sensor_fault"}}),
    ]
    fuel_out, lox_out = SensorFaultClassifier().classify(mislabelled)
    assert fuel_out.classification["category"] == "sensor_fault"
    assert fuel_out.classification["triage_category"] == "pressurisation"
    assert fuel_out.severity == Severity.WARNING  # CRITICAL glitch capped
    assert lox_out.classification["category"] == "pressurisation"
    assert lox_out.classification["triage_category"] == "sensor_fault"


def test_two_sample_excursion_is_indeterminate() -> None:
    rng = np.random.default_rng(3)
    values = 20 + rng.normal(0, 0.05, 1000)
    values[500:502] += 3.0
    event = next(e for e in AnomalyDetector(use_llm=False).detect(_series(values)) if e.consecutive_samples == 2)
    assert SensorFaultClassifier().verdict(event)[0] == "indeterminate"


# --------------------------------------------------------------------------- #
# Onset backtracking, startup filter, severity floors
# --------------------------------------------------------------------------- #
def test_onsets_close_to_injected_fault_times(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = {e.sensor_id: e for e in detector.detect(readings) if not e.expected}
    injected = {"lox_tank_pressure": 142.0, "turbopump_vibration": 143.5, "engine_bay_temp": 145.0,
                "avionics_bus_voltage": 210.0, "fuel_tank_pressure": 230.0}
    for sensor, met in injected.items():
        assert abs(events[sensor].mission_elapsed_s - met) <= 1.0, sensor
        assert events[sensor].mission_elapsed_s <= events[sensor].detected_at_s
    temp = events["engine_bay_temp"]
    assert temp.detected_at_s - temp.mission_elapsed_s > 2.0  # slow ramp: detected late, onset backtracked
    assert "Onset backtracked" in temp.description
    assert events["fuel_tank_pressure"].mission_elapsed_s == events["fuel_tank_pressure"].detected_at_s  # spike


def test_slow_ramp_onset_precedes_later_sharp_event() -> None:
    rng = np.random.default_rng(11)
    t = np.arange(0, 300, 0.1)
    ramp = 20 + rng.normal(0, 0.4, t.size)
    ramp += 25 * np.clip((t - 130) / 4, 0, 1)                       # slow 4 s ramp from T+130
    step = 45 + rng.normal(0, 0.1, t.size)
    step[t >= 133.0] -= 4.0                                         # sharp drop at T+133
    readings = _series(ramp, sensor="bay_temp", unit="degC") + _series(step, sensor="tank_pressure")
    events = {e.sensor_id: e for e in AnomalyDetector(use_llm=False).detect(readings)}
    assert events["bay_temp"].mission_elapsed_s < events["tank_pressure"].mission_elapsed_s


def test_startup_transients_suppressed_and_reported() -> None:
    rng = np.random.default_rng(12)
    values = 50 + rng.normal(0, 0.1, 1000)
    values[20:25] += 5.0                                            # T+2.0 - T+2.4
    values[600:605] += 5.0                                          # T+60
    detector = AnomalyDetector(use_llm=False)
    events = detector.detect(_series(values))
    assert all(e.mission_elapsed_s > 5 for e in events)
    assert len(detector.last_suppressed) == 1 and detector.last_suppressed[0].startswith("test_sensor first detected at T+1.")
    assert AnomalyDetector(DetectorConfig(startup_ignore_seconds=0), use_llm=False).detect(_series(values))[0] \
        .mission_elapsed_s < 5


def test_severity_floor_raises_large_relative_changes() -> None:
    rng = np.random.default_rng(13)
    values = 100 + rng.normal(0, 2.0, 1000)
    values[500:560] += 12.0                                         # 12 % pressure change, modest in sigma terms
    floors = [("*pressure*", 5.0, Severity.WARNING)]
    plain = AnomalyDetector(DetectorConfig(), use_llm=False).detect(_series(values, sensor="tank_pressure"))
    floored = AnomalyDetector(DetectorConfig(severity_floors=floors), use_llm=False).detect(
        _series(values, sensor="tank_pressure"))
    assert plain and floored
    if plain[0].severity == Severity.INFO:
        assert floored[0].severity == Severity.WARNING and "floor" in floored[0].description
    assert all(e.severity != Severity.INFO for e in floored)
    other = AnomalyDetector(DetectorConfig(severity_floors=floors), use_llm=False).detect(_series(values, sensor="x_rate"))
    assert [e.severity for e in other] == [e.severity for e in plain]  # pattern does not match -> unchanged


def test_shipped_config_has_floors_and_startup_filter(agents_config: dict) -> None:
    config = DetectorConfig.from_dict(agents_config["anomaly_detector"])
    assert config.startup_ignore_seconds == 5.0 and config.onset_backtrack
    assert {p: m for p, m, _ in config.severity_floors} == {"*pressure*": 5, "*temp*": 10, "*vib*": 20, "*voltage*": 3}


# --------------------------------------------------------------------------- #
# Fix 15: transient vs sustained
# --------------------------------------------------------------------------- #
def test_recovery_classification_on_sample(detector: AnomalyDetector, readings: list[TelemetryReading]) -> None:
    events = {e.sensor_id: e for e in detector.detect(readings) if not e.expected}
    assert events["fuel_tank_pressure"].recovery == "transient"        # single-sample glitch
    assert events["avionics_bus_voltage"].recovery == "transient"      # 0.8 s sag
    assert events["turbopump_vibration"].recovery == "transient"       # ~1 s burst
    assert events["lox_tank_pressure"].recovery == "sustained"         # slow 12 s recovery
    assert events["engine_bay_temp"].recovery == "sustained"           # 30 s decay
    lox = events["lox_tank_pressure"]
    assert lox.recovery_text.startswith("sustained (value remained at") and "baseline was" in lox.recovery_text
    assert "Sustained: value remained at" in lox.description
    assert events["fuel_tank_pressure"].to_prompt_dict()["recovery"].startswith("transient (recovered within")


def test_step_is_sustained_and_glitch_is_transient() -> None:
    rng = np.random.default_rng(14)
    step = 28 + rng.normal(0, 0.04, 3000)
    step[2300:] -= 0.9                                              # permanent step at T+230
    glitch = 28 + rng.normal(0, 0.04, 3000)
    glitch[2300:2304] -= 1.5                                        # 0.4 s dip
    step_event = AnomalyDetector(use_llm=False).detect(_series(step, sensor="bus_voltage", unit="V"))[0]
    glitch_event = AnomalyDetector(use_llm=False).detect(_series(glitch, sensor="bus_voltage", unit="V"))[0]
    assert step_event.recovery == "sustained" and abs(step_event.settled_value - 27.1) < 0.1
    assert glitch_event.recovery == "transient" and glitch_event.recovery_s < 1.5
