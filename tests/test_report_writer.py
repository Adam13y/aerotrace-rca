"""Tests for the report writer, hypothesis generator and supervisor pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.settings import Settings
from src.agents.hypothesis_generator import HypothesisGenerator, citation_problem
from src.agents.report_writer import ReportConfig, ReportWriter
from src.agents.supervisor import Supervisor
from src.ingestion import write_sample_files
from src.memory.knowledge_base import KnowledgeBase
from src.memory.vector_store import VectorStore
from src.models.telemetry_schema import AnomalyEvent, Hypothesis, RCAReport, Severity, TelemetryReading
from src.utils.helpers import LLMRouter, resolve_anomaly_reference

BASE = datetime(2026, 1, 1, tzinfo=UTC)


class FakeClient:
    """Stand-in LLM client returning a canned response."""

    def __init__(self, model: str, response: str, up: bool = True) -> None:
        self.model = model
        self.response = response
        self.up = up

    def available(self) -> bool:
        return self.up

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        return self.response


def _anomaly(sensor: str, met: float, severity: Severity, **kwargs: object) -> AnomalyEvent:
    ts = BASE + timedelta(seconds=met)
    reading = TelemetryReading(timestamp=ts, sensor_id=sensor, value=1.0, unit="psi")
    return AnomalyEvent(
        id=f"ANM-{sensor[:4].upper()}{int(met):04d}",
        timestamp=ts,
        end_timestamp=ts + timedelta(seconds=2),
        sensor_id=sensor,
        severity=severity,
        description=f"{sensor} deviated at T+{met}s",
        readings=[reading],
        unit="psi",
        mission_elapsed_s=met,
        duration_s=2.0,
        baseline_value=50.0,
        peak_value=44.0,
        score=18.0,
        detection_methods=["zscore"],
        **kwargs,
    )


@pytest.fixture()
def anomalies() -> list[AnomalyEvent]:
    lox = _anomaly("lox_tank_pressure", 142, Severity.CRITICAL, classification={"category": "pressurisation"})
    vib = _anomaly("turbopump_vibration", 143, Severity.CRITICAL, classification={"category": "propulsion"})
    lox = lox.model_copy(update={"correlated_with": [vib.id], "flight_phase": "Mid-burn ascent"})
    vib = vib.model_copy(update={"correlated_with": [lox.id]})
    expected = _anomaly("chamber_pressure", 0, Severity.INFO, expected=True)
    return [expected, lox, vib]


@pytest.fixture()
def hypotheses(anomalies: list[AnomalyEvent]) -> list[Hypothesis]:
    ids = [a.id for a in anomalies if not a.expected]
    return [
        Hypothesis(
            title="Pressurisation regulator failure", description="Regulator stuck closed.", confidence=0.72,
            supporting_evidence=["LOX pressure drop precedes vibration"], counter_evidence=["No pressurant data"],
            related_anomalies=ids, subsystem="pressurisation", recommended_actions=["Inspect regulator"],
        ),
        Hypothesis(title="Sensor glitch", description="DAQ error.", confidence=0.2, related_anomalies=ids[:1]),
    ]


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path, store=VectorStore(tmp_path, prefer_chroma=False))


# --------------------------------------------------------------------------- #
# Report writer
# --------------------------------------------------------------------------- #
def test_report_contains_required_sections(tmp_path: Path, anomalies: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> None:
    writer = ReportWriter(tmp_path, config=ReportConfig(mission_name="Test Flight", include_plots=False), use_llm=False)
    report = writer.build_report(anomalies, hypotheses)
    markdown = writer.render_markdown(report)
    for heading in ("Executive Summary", "Conclusion", "Data Overview", "Event Timeline", "Detailed Findings",
                    "Root Cause Hypotheses", "Recommendations", "Expected Transients", "Methodology"):
        assert heading in markdown, heading
    assert "ANM-" not in markdown and "ANM-" not in writer.render_summary(report)
    assert "LOX tank pressure drop (CRITICAL) at T+142.0s" in markdown
    assert "Pressurisation regulator failure" in report.conclusion
    assert report.recommendations[0].startswith("[P1]")
    assert report.severity_counts == {"CRITICAL": 2, "WARNING": 0, "INFO": 1}


def test_write_creates_timestamped_files(tmp_path: Path, anomalies: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> None:
    writer = ReportWriter(tmp_path, config=ReportConfig(include_plots=False), use_llm=False)
    report = writer.build_report(anomalies, hypotheses)
    stamp = datetime(2026, 3, 4, 5, 6, 7, tzinfo=UTC)
    paths = writer.write(report, formats=("markdown", "json"), timestamp=stamp)
    assert paths["report"].name == "rca_report_20260304_050607.md"
    assert paths["summary"].name == "rca_summary_20260304_050607.md"
    assert "Quick-Look" in paths["summary"].read_text(encoding="utf-8")
    restored = RCAReport.model_validate_json(paths["json"].read_text(encoding="utf-8"))
    assert restored.hypotheses[0].title == hypotheses[0].title


def test_plot_written_when_readings_supplied(tmp_path: Path, anomalies: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> None:
    pytest.importorskip("matplotlib")
    readings = [
        TelemetryReading(timestamp=BASE + timedelta(seconds=i * 0.5), sensor_id="lox_tank_pressure", value=50 + (i % 3), unit="psi")
        for i in range(400)
    ]
    writer = ReportWriter(tmp_path, use_llm=False)
    paths = writer.write(writer.build_report(anomalies, hypotheses), readings=readings)
    assert paths["plot"].exists()
    assert paths["plot"].name in paths["report"].read_text(encoding="utf-8")


def test_report_with_no_anomalies(tmp_path: Path) -> None:
    writer = ReportWriter(tmp_path, use_llm=False)
    report = writer.build_report([], [])
    assert "No actionable anomalies" in report.conclusion
    assert "No anomalies detected." in writer.render_markdown(report)


def test_llm_executive_summary_used(tmp_path: Path, anomalies: list[AnomalyEvent], hypotheses: list[Hypothesis]) -> None:
    router = LLMRouter(None, FakeClient("claude-test", "Board summary: regulator failure suspected."))  # type: ignore[arg-type]
    report = ReportWriter(tmp_path, router=router).build_report(anomalies, hypotheses)
    assert report.summary.startswith("Board summary")
    assert report.metadata["summary_source"] == "llm"


# --------------------------------------------------------------------------- #
# Hypothesis generator
# --------------------------------------------------------------------------- #
def test_rule_based_hypotheses_ranked(kb: KnowledgeBase, anomalies: list[AnomalyEvent]) -> None:
    generated = HypothesisGenerator(kb, use_llm=False).generate(anomalies)
    assert generated
    assert [h.confidence for h in generated] == sorted((h.confidence for h in generated), reverse=True)
    assert all(h.source == "rule-based" for h in generated)
    assert all(set(h.related_anomalies) <= {a.id for a in anomalies if not a.expected} for h in generated)


def test_anomaly_label_format(anomalies: list[AnomalyEvent]) -> None:
    assert anomalies[1].label == "LOX tank pressure drop (CRITICAL) at T+142.0s"
    assert '"label"' in anomalies[1].model_dump_json()  # label exported next to the ID


def test_resolve_reference_tolerates_near_copies(anomalies: list[AnomalyEvent]) -> None:
    lox = anomalies[1]
    assert resolve_anomaly_reference(lox.label.upper(), anomalies) is lox
    assert resolve_anomaly_reference("LOX tank pressure drop at T+142.0s", anomalies) is lox
    assert resolve_anomaly_reference(lox.id, anomalies) is lox
    assert resolve_anomaly_reference("LOX tank pressure drop (CRITICAL) at T+150.0s", anomalies) is None
    # Sensor-name references without a time resolve only when unambiguous and the severity matches.
    assert resolve_anomaly_reference("lox_tank_pressure CRITICAL anomaly", anomalies) is lox
    assert resolve_anomaly_reference("lox_tank_pressure WARNING anomaly", anomalies) is None
    second_lox = _anomaly("lox_tank_pressure", 200, Severity.CRITICAL)
    assert resolve_anomaly_reference("lox_tank_pressure CRITICAL anomaly", [*anomalies, second_lox]) is None


def test_llm_hypotheses_parsed_and_validated(kb: KnowledgeBase, anomalies: list[AnomalyEvent]) -> None:
    lox, vib = anomalies[1], anomalies[2]
    payload = {"hypotheses": [
        {"title": "COPV leak", "description": "LOX ullage pressure fell at T+142s, starving the turbopump inlet "
         "and causing cavitation-induced vibration.", "confidence": 85, "related_anomalies": [lox.label, vib.label],
         "supporting_evidence": "single string", "counter_evidence": [], "recommended_actions": ["x"]},
        {"title": "Strut failure", "description": "d", "confidence": 0.4},
        {"title": "Phantom", "description": "LOX issue", "confidence": 0.5, "related_anomalies": ["Ghost sensor drop at T+1.0s"]},
        {"title": "Bearing wear", "description": "Turbopump bearing degradation.", "confidence": 0.3,
         "related_anomalies": [vib.label, lox.label]},
    ]}
    router = LLMRouter(None, FakeClient("claude-test", json.dumps(payload)))  # type: ignore[arg-type]
    generator = HypothesisGenerator(kb, router=router)
    generated = generator.generate(anomalies)
    assert [h.title for h in generated] == ["COPV leak"]
    assert generated[0].confidence == pytest.approx(0.85)  # percentage normalised
    assert generated[0].related_anomalies == [lox.id, vib.id]  # labels mapped back to internal IDs
    assert generated[0].supporting_evidence == ["single string"]
    assert generator.last_source == "cloud:claude-test"
    reasons = {r.title: r.reason for r in generator.last_rejections}
    assert reasons["Strut failure"] == "cites no anomalies"
    assert "not a detected anomaly" in reasons["Phantom"]
    assert "never mentions the lox tank pressure sensor" in reasons["Bearing wear"]


def test_validation_rejects_mislabelled_pressurisation_hypothesis(kb: KnowledgeBase, anomalies: list[AnomalyEvent]) -> None:
    """Regression: the hypothesis described the LOX drop but cited the temperature and fuel anomalies."""
    temp = _anomaly("engine_bay_temp", 148, Severity.WARNING)
    fuel = _anomaly("fuel_tank_pressure", 230, Severity.WARNING)
    events = [*anomalies, temp, fuel]
    payload = {"hypotheses": [{
        "title": "Tank Pressurisation System Leak", "confidence": 0.45,
        "description": "A rapid drop in LOX or fuel tank pressure followed by a slow recovery suggests a leak.",
        "related_anomalies": [temp.label, fuel.label],
        "supporting_evidence": ["lox tank pressure dropped 5.89 psi at T+143.8s"],
    }]}
    router = LLMRouter(FakeClient("qwen", json.dumps(payload)), None)  # type: ignore[arg-type]
    generator = HypothesisGenerator(kb, router=router)
    generated = generator.generate(events)
    assert all(h.source == "rule-based" for h in generated)  # only hypothesis rejected -> fallback
    assert len(generator.last_rejections) == 1
    assert "Engine bay temperature" in generator.last_rejections[0].reason


def test_citation_time_must_be_consistent(anomalies: list[AnomalyEvent]) -> None:
    vib = anomalies[2]
    assert citation_problem(vib, "Turbopump vibration burst at T+143.5s") is None
    assert citation_problem(vib, "Turbopump vibration rose") is None
    assert "none near" in citation_problem(vib, "Turbopump vibration rose at T+210s")
    assert "never mentions" in citation_problem(vib, "Bus voltage sag at T+143s")


def test_invalid_llm_output_falls_back(kb: KnowledgeBase, anomalies: list[AnomalyEvent]) -> None:
    router = LLMRouter(FakeClient("q", "not json at all"), FakeClient("c", "still not json"))  # type: ignore[arg-type]
    generated = HypothesisGenerator(kb, router=router).generate(anomalies)
    assert generated and all(h.source == "rule-based" for h in generated)


def test_prompt_includes_context(kb: KnowledgeBase, anomalies: list[AnomalyEvent]) -> None:
    generator = HypothesisGenerator(kb, use_llm=False)
    actionable = [a for a in anomalies if not a.expected]
    prompt = generator.build_prompt(anomalies, actionable, {a.id: kb.context_for(a) for a in actionable}, None, "Test mission")
    assert "Test mission" in prompt and "Group 1" in prompt and actionable[0].label in prompt
    assert "Knowledge-base matches" in prompt
    assert "ANM-" not in prompt


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def test_supervisor_end_to_end_offline(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", report_dir=tmp_path / "data" / "reports")
    settings.ensure_directories()
    sample = write_sample_files(settings.raw_dir)["spacex_wide"]
    kb = KnowledgeBase(tmp_path, store=VectorStore(tmp_path / "kb", prefer_chroma=False))
    result = Supervisor(settings=settings, use_llm=False, knowledge_base=kb).run(sample, output_formats=("markdown", "json"))
    assert not result.errors
    assert result.report is not None and result.hypotheses
    for kind in ("report", "summary", "json", "plot", "decision_log"):
        assert result.output_paths[kind].exists(), kind
    log = json.loads(result.output_paths["decision_log"].read_text(encoding="utf-8"))
    assert any("offline" in d["decision"] for d in log["decisions"])
    assert kb.store.count() > 10  # seeded failure modes + stored anomalies


def test_supervisor_analyze_mode(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", report_dir=tmp_path / "data" / "reports")
    settings.ensure_directories()
    sample = write_sample_files(settings.raw_dir)["nasa_long"]
    kb = KnowledgeBase(tmp_path, store=VectorStore(tmp_path / "kb", prefer_chroma=False))
    result = Supervisor(settings=settings, use_llm=False, knowledge_base=kb).run(sample, mode="analyze")
    assert result.anomalies and not result.hypotheses and result.report is None
