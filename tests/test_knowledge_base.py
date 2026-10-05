"""Tests for past-anomaly retrieval: a dataset must never retrieve its own earlier output."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from config.settings import Settings
from src.agents.supervisor import Supervisor
from src.ingestion import load_telemetry, write_sample_files
from src.ingestion.csv_parser import generate_sample_telemetry, to_long_format
from src.memory.knowledge_base import KnowledgeBase
from src.memory.vector_store import VectorStore
from src.models.telemetry_schema import AnomalyEvent, Hypothesis, Severity
from src.utils.helpers import dataset_fingerprint

BASE = datetime(2026, 1, 1, tzinfo=UTC)


def _anomaly(sensor: str, met: float, description: str) -> AnomalyEvent:
    ts = BASE + timedelta(seconds=met)
    return AnomalyEvent(
        id=f"ANM-{sensor[:4].upper()}{int(met):04d}", timestamp=ts, sensor_id=sensor, severity=Severity.CRITICAL,
        description=description, mission_elapsed_s=met, duration_s=2.0, baseline_value=50.0, peak_value=44.0,
    )


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path, store=VectorStore(tmp_path, prefer_chroma=False))


@pytest.fixture(scope="module")
def sample_files(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return write_sample_files(tmp_path_factory.mktemp("raw"))


# --------------------------------------------------------------------------- #
# Dataset fingerprint
# --------------------------------------------------------------------------- #
def test_fingerprint_is_format_independent(sample_files: dict[str, Path], tmp_path: Path) -> None:
    """The same flight as NASA-style CSV, wide-format CSV and JSON is one dataset."""
    channels = tmp_path / "flight.json"
    wide = generate_sample_telemetry()
    channels.write_text(json.dumps({"base_time": wide["timestamp"].iloc[0], "sensors": {
        sensor: {"unit": unit, "samples": [[t, v] for t, v in zip(rows["met_s"], rows["value"])]}
        for (sensor, unit), rows in to_long_format(wide).assign(met_s=list(wide["met_s"]) * 6)
        .groupby(["sensor_id", "unit"])
    }}), encoding="utf-8")
    fingerprints = {
        dataset_fingerprint(load_telemetry(sample_files["nasa_long"])),
        dataset_fingerprint(load_telemetry(sample_files["wide"])),
        dataset_fingerprint(load_telemetry(channels)),
    }
    assert len(fingerprints) == 1


def test_fingerprint_differs_between_flights(sample_files: dict[str, Path], tmp_path: Path) -> None:
    other = tmp_path / "other.csv"
    to_long_format(generate_sample_telemetry(seed=7)).to_csv(other, index=False)
    assert dataset_fingerprint(load_telemetry(sample_files["nasa_long"])) != dataset_fingerprint(load_telemetry(other))


# --------------------------------------------------------------------------- #
# Retrieval exclusion
# --------------------------------------------------------------------------- #
def test_same_dataset_records_excluded_other_datasets_kept(kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142, "lox tank pressure dropped sharply")
    vib = _anomaly("turbopump_vibration", 143, "lox tank pressure related turbopump vibration rose")
    other_flight = _anomaly("lox_tank_pressure", 90, "lox tank pressure dropped sharply on another flight")
    kb.record_run([lox, vib], [], "flight_a.csv", dataset_id="aaaa")
    kb.record_run([other_flight], [], "flight_b.csv", dataset_id="bbbb")

    hits = kb.similar_to(lox, k=5, kind="anomaly", exclude_dataset="aaaa", exclude_source="flight_a.csv")
    assert [h.metadata["dataset_id"] for h in hits] == ["bbbb"]
    assert kb.excluded_ids == {f"aaaa:{vib.id}", f"aaaa:{lox.id}"}

    # Without exclusion the same-dataset records come back.
    unfiltered = kb.similar_to(lox, k=5, kind="anomaly")
    assert {h.metadata["dataset_id"] for h in unfiltered} == {"aaaa", "bbbb"}


def test_same_flight_under_another_file_name_is_excluded(kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142, "lox tank pressure dropped")
    vib = _anomaly("turbopump_vibration", 143, "lox tank pressure turbopump vibration")
    kb.record_run([vib], [], "flight.json", dataset_id="aaaa")
    assert kb.similar_to(lox, k=5, kind="anomaly", exclude_dataset="aaaa", exclude_source="flight_nasa.csv") == []


def test_legacy_records_without_fingerprint_matched_by_file_name(kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142, "lox tank pressure dropped")
    vib = _anomaly("turbopump_vibration", 143, "lox tank pressure turbopump vibration")
    kb.record_run([vib], [], "flight_a.csv")  # pre-fix record: no dataset_id
    assert kb.similar_to(lox, k=5, kind="anomaly", exclude_dataset="aaaa", exclude_source="flight_a.csv") == []
    assert len(kb.similar_to(lox, k=5, kind="anomaly", exclude_dataset="aaaa", exclude_source="other.csv")) == 1


def test_records_namespaced_by_dataset(kb: KnowledgeBase) -> None:
    """Two flights with coinciding anomaly IDs no longer overwrite each other."""
    anomaly = _anomaly("lox_tank_pressure", 142, "lox tank pressure dropped")
    kb.record_run([anomaly], [Hypothesis(title="A", description="", confidence=0.5, related_anomalies=[anomaly.id])],
                  "a.csv", dataset_id="aaaa")
    kb.record_run([anomaly], [Hypothesis(title="B", description="", confidence=0.5, related_anomalies=[anomaly.id])],
                  "b.csv", dataset_id="bbbb")
    probe = _anomaly("lox_tank_pressure", 10, "lox tank pressure dropped")
    hits = kb.similar_to(probe, k=5, kind="anomaly")
    assert {h.metadata["leading_hypothesis"] for h in hits} == {"A", "B"}
    assert all(h.metadata["anomaly_id"] == anomaly.id for h in hits)


def test_failure_modes_unaffected_by_exclusion(kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142, "tank pressure drop slow recovery regulator")
    kinds = [h.metadata["kind"] for h in kb.context_for(lox, k=3, exclude_dataset="aaaa", exclude_source="x.csv")]
    assert kinds.count("failure_mode") == 3


# --------------------------------------------------------------------------- #
# End-to-end: re-analysing a dataset must not see its own previous output
# --------------------------------------------------------------------------- #
def test_rerun_does_not_retrieve_own_output(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path / "data", report_dir=tmp_path / "data" / "reports")
    settings.ensure_directories()
    paths = write_sample_files(settings.raw_dir)
    kb = KnowledgeBase(tmp_path, store=VectorStore(tmp_path / "kb", prefer_chroma=False))
    supervisor = Supervisor(settings=settings, use_llm=False, knowledge_base=kb)

    first = supervisor.run(paths["nasa_long"], output_formats=("json",))
    assert kb.excluded_ids == set()  # nothing stored yet
    stored_after_first = kb.store.count()

    second = supervisor.run(paths["nasa_long"], output_formats=("json",))
    assert kb.excluded_ids, "second run must skip the records saved by the first"
    third = supervisor.run(paths["wide"], output_formats=("json",))  # same flight, other format
    assert kb.excluded_ids

    signature = lambda r: [(h.title, h.confidence, h.related_anomalies, h.description) for h in r.hypotheses]  # noqa: E731
    assert signature(first) == signature(second) == signature(third)
    assert kb.store.count() == stored_after_first  # same dataset -> same namespaced records, upserted
    log = json.loads(second.output_paths["decision_log"].read_text(encoding="utf-8"))
    assert any(d["decision"].startswith("Excluded") and "past-anomaly" in d["decision"] for d in log["decisions"])
