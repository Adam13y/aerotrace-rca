"""Tests for the subsystem-evidence (rule 1) and coupling (rule 2) checks and the new explanation types."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.agents.evidence_rules import UNOBSERVABLE_LABEL, EvidenceRules, normalise_type
from src.agents.hypothesis_generator import HypothesisConfig, HypothesisGenerator
from src.memory.knowledge_base import KnowledgeBase
from src.memory.vector_store import VectorStore
from src.models.telemetry_schema import AnomalyEvent, Severity
from src.utils.helpers import LLMRouter

BASE = datetime(2026, 1, 1, tzinfo=UTC)
VEHICLE = ["chamber_pressure", "lox_tank_pressure", "fuel_tank_pressure", "engine_bay_temp",
           "turbopump_vibration", "avionics_bus_voltage"]


def _a(sensor: str, met: float) -> AnomalyEvent:
    ts = BASE + timedelta(seconds=met)
    return AnomalyEvent(id=f"ANM-{sensor[:5].upper()}{int(met * 10):05d}", timestamp=ts, sensor_id=sensor,
                        severity=Severity.CRITICAL, description=f"{sensor} deviated", mission_elapsed_s=met,
                        duration_s=2.0, baseline_value=50.0, peak_value=44.0)


def _pair(first: AnomalyEvent, second: AnomalyEvent) -> list[AnomalyEvent]:
    return [first.model_copy(update={"correlated_with": [second.id]}),
            second.model_copy(update={"correlated_with": [first.id]})]


class Client:
    model = "qwen"

    def __init__(self, *answers: dict) -> None:
        self.answers = [json.dumps(a) for a in answers]
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else '{"hypotheses": []}'


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path, store=VectorStore(tmp_path, prefer_chroma=False))


def _h(title: str, cites: list[AnomalyEvent], text: str, conf: float = 0.8, kind: str = "root_cause") -> dict:
    return {"type": kind, "title": title, "description": text, "confidence": conf,
            "related_anomalies": [a.label for a in cites]}


def _generator(kb: KnowledgeBase, *answers: dict, **config: object) -> tuple[HypothesisGenerator, Client]:
    client = Client(*answers)
    config.setdefault("coverage_retries", 0)
    return HypothesisGenerator(kb, router=LLMRouter(client, None), config=HypothesisConfig(**config)), client  # type: ignore[arg-type]


# --------------------------------------------------------------------------- rule 1 unit tests
def test_named_subsystems_strips_sensor_names_and_neutral_phrases() -> None:
    rules = EvidenceRules()
    assert rules.named_subsystems("Engine bay temperature rose in the engine bay", VEHICLE) == set()
    assert rules.named_subsystems("Avionics Bus Under-Voltage from Load Transient", VEHICLE) == {"electrical"}
    assert rules.named_subsystems("Hot Gas Leak in Engine Bay", VEHICLE) == {"propulsion"}
    assert rules.named_subsystems("LOX Tank Pressurisation Leak causing pump cavitation", VEHICLE) == \
        {"pressurization", "propulsion"}
    assert rules.named_subsystems("Turbopump vibration rose", VEHICLE) == set()  # sensor name, not a cause
    assert rules.named_subsystems("Thermal control failure", VEHICLE) == {"thermal"}


def test_subsystem_check_outcomes() -> None:
    rules = EvidenceRules()
    temp, lox = _a("engine_bay_temp", 165), _a("lox_tank_pressure", 142)
    problem, _ = rules.subsystem_check("Turbopump Cavitation and Rotor Instability", [temp], VEHICLE)
    assert problem and problem.startswith("names propulsion as part of the cause but cites no anomaly")
    assert rules.subsystem_check("Thermal Control Failure", [temp], VEHICLE) == (None, [])
    assert rules.subsystem_check("LOX Tank Pressurisation Leak", [lox], VEHICLE) == (None, [])  # multi-evidence sensor
    assert rules.subsystem_check("Structural failure of a tank strut", [lox], VEHICLE) == (None, ["structural"])


def test_coupling_check() -> None:
    rules = EvidenceRules()
    lox, vib, temp = _a("lox_tank_pressure", 142), _a("turbopump_vibration", 143), _a("engine_bay_temp", 148)
    volt, imu = _a("avionics_bus_voltage", 188), _a("imu_heartbeat_hz", 188.3)
    assert rules.coupling_check([lox, vib, temp]) is None                 # propulsion -> thermal
    assert rules.coupling_check([volt, imu]) is None                      # electrical -> avionics
    problem = rules.coupling_check([lox, _a("avionics_bus_voltage", 152)])
    assert problem and "no physical path propulsion -> electrical" in problem
    assert rules.coupling_check([temp, _a("fuel_tank_pressure", 151)]) is not None  # no direct thermal -> propulsion
    camera = _a("camera_frame_rate_hz", 176)                              # unmapped sensor: no couplings at all
    assert "unmapped:camera_frame_rate_hz" in rules.coupling_check([vib, camera])


def test_normalise_type() -> None:
    assert normalise_type("independent_events") == "independent_events"
    assert normalise_type("Independent coincident events") == "independent_events"
    assert normalise_type("insufficient evidence to discriminate") == "insufficient_evidence"
    assert normalise_type(None) == normalise_type("whatever") == "root_cause"


def test_config_from_yaml_section() -> None:
    rules = EvidenceRules.from_dict({"couplings": {"thermal": ["propulsion"]}, "unobservable_confidence_cap": 0.2})
    assert rules.couplings == {"thermal": ["propulsion"]} and rules.unobservable_cap == 0.2


# --------------------------------------------------------------------------- generator integration
def test_padding_mechanism_without_evidence_is_rejected(kb: KnowledgeBase) -> None:
    temp = _a("engine_bay_temp", 165)
    answer = {"hypotheses": [
        _h("Turbopump Cavitation and Rotor Instability", [temp], "Engine bay temperature rose at T+165s.", 0.75),
        _h("Thermal control failure", [temp], "Bay thermal control lost at T+165s."),
    ]}
    # The vehicle has propulsion sensors (they only show planned transients), so a propulsion cause is
    # observable but unsupported -> rejected.
    planned = [_a(s, 1).model_copy(update={"expected": True}) for s in VEHICLE if s != "engine_bay_temp"]
    generator, _ = _generator(kb, answer)
    assert [h.title for h in generator.generate([temp, *planned])] == ["Thermal control failure"]
    assert any("names propulsion as part of the cause" in r.reason for r in generator.last_rejections)
    # A vehicle with only the temperature sensor cannot observe propulsion -> capped and labelled instead.
    generator, _ = _generator(kb, answer)
    capped = {h.title: h for h in generator.generate([temp])}["Turbopump Cavitation and Rotor Instability"]
    assert capped.confidence == pytest.approx(0.3) and capped.labels == [f"{UNOBSERVABLE_LABEL}: propulsion"]


def test_unobservable_subsystem_is_capped_and_labelled(kb: KnowledgeBase) -> None:
    lox = _a("lox_tank_pressure", 142)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Structural failure of internal tank strut", [lox], "LOX tank pressure fell at T+142s.", 0.75)]})
    hypothesis = generator.generate([lox])[0]
    assert hypothesis.confidence == pytest.approx(0.3)
    assert hypothesis.labels == [f"{UNOBSERVABLE_LABEL}: structural"]
    assert any("no sensor for it" in f.reason for f in generator.last_flags)


def test_independent_events_may_link_uncoupled_subsystems(kb: KnowledgeBase) -> None:
    lox, volt = _pair(_a("lox_tank_pressure", 150), _a("avionics_bus_voltage", 152.5))
    generator, _ = _generator(kb, {"hypotheses": [
        _h("LOX leak caused the bus sag", [lox, volt], "LOX tank pressure fell at T+150s, then the voltage sagged."),
        _h("Independent coincident events", [lox, volt], "LOX tank pressure drop and avionics bus voltage sag are unrelated.", 0.7,
           kind="independent_events"),
        _h("LOX regulator fault", [lox], "LOX tank pressurisation regulator failed at T+150s.", 0.6),
        _h("Battery cell load transient", [volt], "Bus voltage sag at T+152.5s from a battery cell.", 0.6),
    ]})
    titles = {h.title: h for h in generator.generate([lox, volt])}
    assert "LOX leak caused the bus sag" not in titles
    assert any("propulsion -> electrical" in r.reason for r in generator.last_rejections)
    assert titles["Independent coincident events"].hypothesis_type == "independent_events"
    assert {"LOX regulator fault", "Battery cell load transient"} <= set(titles)
    assert "Independent coincident events" in generator.last_competing  # counts as a valid alternative


def test_insufficient_evidence_counts_as_alternative(kb: KnowledgeBase) -> None:
    temp = _a("engine_bay_temp", 165)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Thermal control failure", [temp], "Bay thermal control lost at T+165s."),
        _h("Cannot discriminate", [temp], "Thermal control loss vs external heating; one sensor cannot tell.", 0.4,
           kind="insufficient_evidence"),
    ]})
    generator.generate([temp])
    assert generator.last_competing == ["Thermal control failure", "Cannot discriminate"]


def test_independent_events_does_not_count_as_explaining_an_anomaly(kb: KnowledgeBase) -> None:
    lox, volt = _pair(_a("lox_tank_pressure", 150), _a("avionics_bus_voltage", 152.5))
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Independent coincident events", [lox, volt], "LOX tank pressure and bus voltage events are unrelated.", 0.7,
           kind="independent_events"),
        _h("LOX regulator fault", [lox], "LOX tank pressurisation regulator failed at T+150s.", 0.6),
    ]}, {"hypotheses": [_h("Battery cell transient", [volt], "Bus voltage sag at T+152.5s from a battery cell.")]},
        coverage_retries=1)
    titles = [h.title for h in generator.generate([lox, volt])]
    assert "Battery cell transient" in titles  # the coverage retry still ran for the voltage anomaly
    assert [r.label for r in generator.last_retries if r.kind == "coverage"] == [volt.label]


def test_prompt_shows_sensors_couplings_and_types(kb: KnowledgeBase) -> None:
    lox = _a("lox_tank_pressure", 142)
    generator, client = _generator(kb, {"hypotheses": []})
    generator.generate([lox])
    prompt = client.prompts[0]
    assert "## Vehicle sensors by subsystem" in prompt and "- pressurization: lox_tank_pressure" in prompt
    assert "- propulsion -> thermal" in prompt
    assert '"independent_events"' in prompt and '"insufficient_evidence"' in prompt
    assert "does NOT mean they are causally related" in prompt
