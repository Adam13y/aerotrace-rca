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
    assert any("names turbopump cavitation as the cause" in r.reason for r in generator.last_rejections)
    # A vehicle with only the temperature sensor cannot observe the turbopump -> capped and labelled instead.
    generator, _ = _generator(kb, answer)
    capped = {h.title: h for h in generator.generate([temp])}["Turbopump Cavitation and Rotor Instability"]
    assert capped.confidence == pytest.approx(0.3)
    assert capped.labels == [f"{UNOBSERVABLE_LABEL}: turbopump_cavitation"]


def test_unobservable_subsystem_is_capped_and_labelled(kb: KnowledgeBase) -> None:
    lox = _a("lox_tank_pressure", 142)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Structural failure of internal tank strut", [lox], "LOX tank pressure fell at T+142s.", 0.75)]})
    hypothesis = generator.generate([lox])[0]
    assert hypothesis.confidence == pytest.approx(0.3)
    assert hypothesis.labels == [f"{UNOBSERVABLE_LABEL}: structural"]
    assert any("no dedicated sensor" in f.reason for f in generator.last_flags)


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


# --------------------------------------------------------------------------- component-level rule 1
def test_component_requirements() -> None:
    rules = EvidenceRules()
    lox, vib = _a("lox_tank_pressure", 142), _a("turbopump_vibration", 143)
    chamber = _a("chamber_pressure", 200)
    vehicle = [*VEHICLE, "camera_frame_rate_hz"]
    # cavitation needs turbopump vibration AND a tank pressure
    assert rules.component_check("Propellant pump cavitation", [lox, vib], vehicle) == (None, [], True)
    problem, _, _ = rules.component_check("Propellant pump cavitation", [lox], vehicle)
    assert problem and "requires cited anomalies on turbopump_vibration AND (lox_tank_pressure OR fuel_tank_pressure)" in problem
    problem, _, _ = rules.component_check("Pump cavitation", [chamber, vib], vehicle)
    assert problem and "lox_tank_pressure" in problem
    assert rules.component_check("Combustion instability in the main chamber", [chamber], vehicle)[0] is None
    # camera: glob requirement, inferred when the vehicle has no camera sensor
    camera = _a("camera_frame_rate_hz", 176)
    assert rules.component_check("Camera power supply failure", [camera], vehicle) == (None, [], True)
    assert rules.component_check("Camera failure", [vib], VEHICLE)[1] == [("local_component_inferred: camera_failure", 0.5)]
    # no listed component -> caller falls back to the subsystem check
    assert rules.component_check("Gas generator malfunction", [vib], vehicle) == (None, [], False)


def test_only_title_and_primary_cause_are_checked() -> None:
    text = EvidenceRules.primary_cause_text(
        "Thermal control failure",
        "Bay thermal control was lost at T+165s. Unlike a turbopump cavitation event, no pressure change was seen.")
    assert text == "Thermal control failure. Bay thermal control was lost at T+165s."
    assert EvidenceRules().named_components(text, VEHICLE) == {"thermal_failure"}


def test_passing_mention_no_longer_rejects(kb: KnowledgeBase) -> None:
    vib = _a("turbopump_vibration", 173.8)
    camera = _a("camera_frame_rate_hz", 175.9)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Turbopump bearing wear", [vib], "Bearing wear raised turbopump vibration at T+173.8s. A battery or "
                                           "electrical cause was considered but the bus voltage stayed nominal."),
        _h("Camera power supply failure", [camera], "The camera's local power supply failed at T+175.9s. "
                                                    "The main electrical bus was unaffected."),
    ]})
    planned = [_a(s, 1).model_copy(update={"expected": True}) for s in VEHICLE if s != "turbopump_vibration"]
    titles = {h.title: h for h in generator.generate([vib, camera, *planned])}
    assert set(titles) == {"Turbopump bearing wear", "Camera power supply failure"}
    assert not generator.last_rejections


# --------------------------------------------------------------------------- Fix 16: cross-subsystem mechanisms
def test_cross_subsystem_mechanism_needs_both_subsystems() -> None:
    rules = EvidenceRules()
    temp, vib, lox = _a("engine_bay_temp", 165), _a("turbopump_vibration", 164), _a("lox_tank_pressure", 163)
    problem, _, _ = rules.component_check("Plume Recirculation Overheating Engine Compartment", [temp], VEHICLE)
    assert problem and problem.startswith("cross-subsystem hypothesis: names thermal failure (thermal)")
    assert "propulsion mechanism ('plume')" in problem
    # with propulsion evidence cited, the same claim passes
    assert rules.component_check("Plume Recirculation Overheating", [lox, vib, temp], VEHICLE)[0] is None
    # "hot gas" counts for both thermal and propulsion
    problem, _, _ = rules.component_check("Hot gas leak overheating the bay", [temp], VEHICLE)
    assert problem and "propulsion mechanism ('hot gas')" in problem
    # same-subsystem mechanisms are fine
    assert rules.component_check("Overheating from insulation debonding", [temp], VEHICLE)[0] is None
    # sensor names (also without their unit word) are not mechanisms
    imu = _a("imu_heartbeat_hz", 188.3)
    volt = _a("avionics_bus_voltage", 188)
    assert rules.component_check("Bus under-voltage. The IMU heartbeat dropped.", [volt, imu],
                                 [*VEHICLE, "imu_heartbeat_hz"])[0] is None


def test_cross_subsystem_unobservable_is_capped() -> None:
    temp = _a("engine_bay_temp", 165)
    problem, caps, _ = EvidenceRules().component_check("Plume overheating", [temp], ["engine_bay_temp"])
    assert problem is None and caps == [("unobservable - no sensor coverage: propulsion (cross-subsystem)", 0.3)]


# --------------------------------------------------------------------------- Fix 14: staged follow-ups
def test_independence_instruction_then_single_cause_then_placeholder(kb: KnowledgeBase) -> None:
    lox, volt = _pair(_a("lox_tank_pressure", 150), _a("avionics_bus_voltage", 152.5))
    linked = {"hypotheses": [
        _h("LOX leak", [lox], "LOX tank pressurisation regulator failed at T+150s."),
        _h("Leak caused bus sag", [lox, volt], "LOX tank pressure fell at T+150s; the avionics bus voltage then sagged."),
    ]}
    still_linked = {"hypotheses": [_h("Bus sag from leak", [lox, volt], "LOX tank pressure fell at T+150s; bus voltage sagged.")]}
    generator, client = _generator(kb, linked, still_linked, still_linked, coverage_retries=2)
    generated = generator.generate([lox, volt])
    first_follow_up, second_follow_up = client.prompts[1], client.prompts[2]
    assert "IMPORTANT: The anomalies [" + volt.label + "] and [" + lox.label + "] are INDEPENDENT events" in first_follow_up
    assert "You MUST provide a cause for [" + volt.label + "] that does NOT reference [" + lox.label + "]" in first_follow_up
    assert second_follow_up.startswith("Give exactly ONE root cause for this single anomaly:")
    assert "only the electrical subsystem" in second_follow_up
    assert [r.kind for r in generator.last_retries] == ["coverage", "single_cause"]
    placeholder = next(h for h in generated if h.hypothesis_type == "unresolved")
    assert placeholder.title == f"Unresolved: {volt.label} - insufficient evidence to determine root cause"
    assert placeholder.labels == ["manual review required"] and placeholder.related_anomalies == [volt.id]
    assert generator.last_unresolved == [volt.id]


def test_single_cause_answer_resolves_anomaly(kb: KnowledgeBase) -> None:
    lox, volt = _pair(_a("lox_tank_pressure", 150), _a("avionics_bus_voltage", 152.5))
    generator, _ = _generator(kb, {"hypotheses": [
        _h("LOX leak", [lox], "LOX tank pressurisation regulator failed at T+150s.")]},
        {"hypotheses": []},
        {"hypotheses": [_h("Battery cell transient", [volt], "Avionics bus voltage sagged at T+152.5s from a battery cell.")]},
        coverage_retries=2)
    titles = [h.title for h in generator.generate([lox, volt])]
    assert "Battery cell transient" in titles and not generator.last_unresolved


def test_single_cause_answer_citing_other_anomalies_is_not_accepted(kb: KnowledgeBase) -> None:
    temp, volt = _a("engine_bay_temp", 100), _a("avionics_bus_voltage", 102)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Thermal control failure", [temp], "Bay thermal control lost at T+100s.")]},
        {"hypotheses": []},
        {"hypotheses": [_h("Heat-induced bus sag", [temp, volt],
                           "Engine bay temperature rose at T+100s and heated the avionics bus power supply, "
                           "which sagged at T+102s.")]},
        coverage_retries=2)
    generated = generator.generate([temp, volt])
    assert generator.last_unresolved == [volt.id]
    assert any("single-cause answer cites other anomalies" in r.reason for r in generator.last_rejections)
    assert all(h.hypothesis_type != "root_cause" or volt.id not in h.related_anomalies for h in generated)


# --------------------------------------------------------------------------- validator ordering / fallback / padding
def _vehicle_analysis() -> object:
    """Minimal dataset overview so the generator knows every sensor the vehicle has."""
    from types import SimpleNamespace
    sensors = [SimpleNamespace(sensor_id=s, unit="", sample_rate_hz=10.0) for s in VEHICLE]
    return SimpleNamespace(sensors=sensors, duration_s=300.0, total_readings=18000, data_quality_notes=[])
def test_rewording_dropped_before_coverage_so_its_anomaly_gets_a_follow_up(kb: KnowledgeBase) -> None:
    lox, fuel = _pair(_a("lox_tank_pressure", 150), _a("fuel_tank_pressure", 151))
    generator, _ = _generator(kb, {"hypotheses": [
        _h("LOX Tank Pressurisation Leak", [lox], "LOX tank pressure fell at T+150s from a pressurisation leak.", 0.8),
        _h("LOX Tank Pressurization System Leak", [lox, fuel],
           "LOX tank pressure at T+150s and fuel tank pressure at T+151s fell from a pressurization leak.", 0.6),
    ]}, {"hypotheses": [_h("Fuel regulator droop", [fuel], "Fuel tank pressure fell at T+151s from regulator droop.")]},
        coverage_retries=1)
    titles = [h.title for h in generator.generate([lox, fuel])]
    assert "LOX Tank Pressurization System Leak" not in titles     # the rewording is dropped ...
    assert [r.label for r in generator.last_retries] == [fuel.label]  # ... before coverage is checked
    assert "Fuel regulator droop" in titles


def test_all_rejected_runs_follow_ups_instead_of_unvalidated_fallback(kb: KnowledgeBase) -> None:
    temp = _a("engine_bay_temp", 165)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Turbopump hot gas leak", [temp], "Engine bay temperature rose at T+165s from a turbopump hot gas leak."),
    ]}, {"hypotheses": [_h("Bay thermal control failure", [temp], "Engine bay temperature rose at T+165s; thermal "
                                                                   "control failed.")]}, coverage_retries=1)
    generated = generator.generate([temp], _vehicle_analysis())  # type: ignore[arg-type]
    assert [h.title for h in generated] == ["Bay thermal control failure"]
    assert generator.last_source != "rule-based" and generator.last_retries


def test_rule_based_fallback_is_validated(kb: KnowledgeBase, monkeypatch: pytest.MonkeyPatch) -> None:
    from src.models.telemetry_schema import Hypothesis
    temp = _a("engine_bay_temp", 165)
    bad = Hypothesis(title="Turbopump hot gas leak", description="Engine bay temperature rose at T+165s from a "
                     "turbopump hot gas leak.", confidence=0.6, related_anomalies=[temp.id], source="rule-based")
    generator = HypothesisGenerator(kb, use_llm=False)
    monkeypatch.setattr(generator, "_rule_based", lambda physical, kb_matches: [bad])
    generated = generator.generate([temp], _vehicle_analysis())  # type: ignore[arg-type]
    assert [h.hypothesis_type for h in generated] == ["unresolved"]   # rejected by rule 1, not passed through
    assert generator.last_rejections and generator.last_rejections[0].title == "Turbopump hot gas leak"


def test_insufficient_evidence_cannot_pad_with_unrelated_causes(kb: KnowledgeBase) -> None:
    volt = _a("avionics_bus_voltage", 230)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("Battery cell loss", [volt], "Bus voltage stepped down at T+230s after losing a battery cell.", 0.7),
        _h("Turbopump Bearing Instability", [volt], "Bus voltage dip at T+230s; could be turbopump bearing wear.",
           0.3, kind="insufficient_evidence"),
    ]})
    titles = [h.title for h in generator.generate([volt], _vehicle_analysis())]  # type: ignore[arg-type]
    assert titles == ["Battery cell loss"]
    assert any(r.title == "Turbopump Bearing Instability" for r in generator.last_rejections)


# --------------------------------------------------------------------------- local unit faults
def test_local_unit_fault_not_rejected_for_nominal_parent_sensor() -> None:
    rules = EvidenceRules()
    vehicle = [*VEHICLE, "imu_heartbeat_hz"]
    imu, temp = _a("imu_heartbeat_hz", 95), _a("engine_bay_temp", 97)
    for text in ("IMU Power Supply Transient Failure", "Electrical fault in IMU power supply", "IMU internal brownout"):
        problem, _ = rules.subsystem_check(text, [imu], vehicle)
        assert problem is None, text
        assert rules.component_check(text, [imu], vehicle)[0] is None, text
    assert rules.local_units_named("IMU power supply failure", [imu], vehicle) == {"imu": {"electrical", "avionics"}}
    # The unit's own sensor must be cited: blaming the IMU's power supply for a bay temperature rise still fails.
    assert rules.subsystem_check("IMU power supply failure", [temp], vehicle)[0] is not None
    # A bus-level cause without naming a local unit is still rejected when the bus voltage is nominal.
    assert rules.subsystem_check("Avionics power supply failure", [imu], vehicle)[0] is not None
    # Other subsystems are not exempted: an IMU fault blamed on a turbopump still needs propulsion evidence.
    assert rules.subsystem_check("Turbopump bearing shock damaged the IMU", [imu], vehicle)[0] is not None


def test_local_unit_label_on_accepted_hypothesis(kb: KnowledgeBase) -> None:
    imu = _a("imu_heartbeat_hz", 95)
    generator, _ = _generator(kb, {"hypotheses": [
        _h("IMU Power Supply Transient Failure", [imu], "The IMU internal power supply dropped out at T+95s; "
           "IMU heartbeat hz fell to zero while the bus stayed nominal.", 0.7),
    ]})
    analysis = _vehicle_analysis()
    analysis.sensors.append(type(analysis.sensors[0])(sensor_id="imu_heartbeat_hz", unit="Hz", sample_rate_hz=10.0))
    generated = generator.generate([imu], analysis)  # type: ignore[arg-type]
    assert [h.title for h in generated] == ["IMU Power Supply Transient Failure"]
    assert generated[0].labels == ["local unit fault: imu"]


def test_shipped_config_lists_local_units() -> None:
    import yaml
    data = yaml.safe_load(Path("config/agents.yaml").read_text(encoding="utf-8"))
    rules = EvidenceRules.from_dict(data["hypothesis_generator"]["evidence_rules"])
    assert {"imu", "camera", "star_tracker", "battery_cell"} <= set(rules.local_units)
    assert rules.local_units["imu"]["local_subsystems"] == ["electrical", "avionics"]
