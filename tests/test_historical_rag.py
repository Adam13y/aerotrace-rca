"""Tests for knowledge-base RAG: documents, chunking, retrieval, category priority and prompt injection."""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.agents.hypothesis_generator import HypothesisConfig, HypothesisGenerator
from src.memory.historical_rag import (
    KB_CATEGORIES,
    HistoricalCase,
    HistoricalRAG,
    anomaly_categories,
    anomaly_query,
    chunk_text,
    estimate_tokens,
    format_historical_context,
    load_documents,
)
from src.memory.knowledge_base import KnowledgeBase
from src.memory.vector_store import HashingEmbedder, VectorStore
from src.models.telemetry_schema import AnomalyEvent, Severity
from src.utils.helpers import LLMRouter, LLMUnavailableError, OllamaClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = PROJECT_ROOT / "data" / "historical_failures"
BASE = datetime(2026, 1, 1, tzinfo=UTC)
SECTIONS = ("FAILURE MODE", "CAUSAL CHAIN", "TELEMETRY SIGNATURES", "ROOT CAUSE", "LESSONS LEARNED")


class FakeEmbedder:
    """Deterministic offline embedder with the same interface as OllamaEmbedder."""

    def __init__(self, name: str = "fake:hashing") -> None:
        self.name = name
        self._hash = HashingEmbedder(256)

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._hash.embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._hash.embed(text)


class RecordingClient:
    """Fake LLM that records prompts and returns a canned answer."""

    def __init__(self, response: dict) -> None:
        self.model = "qwen"
        self.response = json.dumps(response)
        self.prompts: list[str] = []

    def available(self) -> bool:
        return True

    def chat(self, system: str, prompt: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        self.prompts.append(prompt)
        return self.response


def _anomaly(sensor: str, met: float, description: str, category: str = "") -> AnomalyEvent:
    ts = BASE + timedelta(seconds=met)
    return AnomalyEvent(
        id=f"ANM-{sensor[:4].upper()}{int(met):04d}", timestamp=ts, end_timestamp=ts + timedelta(seconds=2),
        sensor_id=sensor, severity=Severity.CRITICAL, description=description, mission_elapsed_s=met, duration_s=2.0,
        baseline_value=50.0, peak_value=44.0, score=18.0, detection_methods=["zscore"],
        classification={"category": category} if category else {},
    )


def _write(path: Path, header: dict[str, str], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = "\n".join(f"{k}: {v}" for k, v in header.items())
    path.write_text(f"{head}\n\n{body}\n" if header else f"{body}\n", encoding="utf-8")


@pytest.fixture(scope="module")
def docs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A small, controlled knowledge base laid out like the real one."""
    root = tmp_path_factory.mktemp("historical_failures")
    _write(root / "pressurization" / "press-01.txt",
           {"Title": "Ullage pressure loss", "Category": "pressurization", "Failure mode": "Ullage pressure loss",
            "Source type": "synthetic"},
           "FAILURE MODE tank ullage pressure loss from a regulator fault. CAUSAL CHAIN regulator sticks closed, "
           "tank pressure decays, pump inlet pressure falls below margin.")
    _write(root / "propulsion" / "prop-01.txt",
           {"Title": "Turbopump cavitation", "Category": "propulsion", "Failure mode": "Turbopump cavitation",
            "Source type": "synthetic"},
           "FAILURE MODE turbopump cavitation when tank pressure and pump inlet pressure fall. CAUSAL CHAIN low inlet "
           "pressure, vapour bubbles, turbopump vibration rise.")
    _write(root / "electrical" / "elec-01.txt",
           {"Title": "Battery cell failure", "Category": "electrical", "Failure mode": "Battery cell failure",
            "Source type": "synthetic"},
           "FAILURE MODE battery cell short causes a bus voltage sag. CAUSAL CHAIN cell voltage collapses, bus "
           "undervoltage, avionics brownout.")
    _write(root / "historical" / "real-01.txt",
           {"Title": "Real feed system incident", "Category": "propulsion", "Failure mode": "Feed blockage",
            "Source type": "historical", "Date": "2023"},
           "FAILURE MODE oxidizer filter blockage lowered turbopump inlet pressure; tank pressure dropped and engines "
           "shut down. " + "detail " * 400)
    _write(root / "notes" / "custom.txt", {}, "Free-form engineering note about thermal protection.")
    (root / ".hidden").mkdir()
    _write(root / ".hidden" / "skip.txt", {}, "must not be loaded")
    return root


@pytest.fixture(scope="module")
def rag(docs: Path, tmp_path_factory: pytest.TempPathFactory) -> HistoricalRAG:
    store = HistoricalRAG(tmp_path_factory.mktemp("vector_store"), FakeEmbedder(), category_boost=0.05)
    store.build(docs)
    return store


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path, store=VectorStore(tmp_path, prefer_chroma=False))


# --------------------------------------------------------------------------- #
# Loading and chunking
# --------------------------------------------------------------------------- #
def test_recursive_load_with_metadata(docs: Path) -> None:
    by_file = {d.source_file: d for d in load_documents(docs)}
    assert set(by_file) == {
        "pressurization/press-01.txt", "propulsion/prop-01.txt", "electrical/elec-01.txt",
        "historical/real-01.txt", "notes/custom.txt",
    }
    real = by_file["historical/real-01.txt"]
    assert (real.category, real.source_type, real.failure_mode) == ("propulsion", "historical", "Feed blockage")
    custom = by_file["notes/custom.txt"]
    assert (custom.category, custom.source_type) == ("uncategorized", "custom")


def test_chunk_text_respects_token_budget_and_overlap() -> None:
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, chunk_tokens=500, overlap_tokens=50)
    assert len(chunks) == 3
    assert all(estimate_tokens(c) <= 500 for c in chunks)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-37:] == second[:37]  # ~50 tokens of overlap
    assert chunk_text("", 500, 50) == []
    with pytest.raises(ValueError):
        chunk_text(text, 100, 100)


# --------------------------------------------------------------------------- #
# Vector store
# --------------------------------------------------------------------------- #
def test_build_stores_metadata_and_splits_long_cases(rag: HistoricalRAG) -> None:
    stats = rag.stats()
    assert stats["cases"] == 5 and stats["chunks"] == 6  # the long historical case is split in two
    assert stats["by_source_type"] == {"synthetic": 3, "historical": 1, "custom": 1}
    assert stats["by_category"]["propulsion"] == 2


def test_search_returns_one_entry_per_case(rag: HistoricalRAG) -> None:
    cases = rag.search("oxidizer filter blockage turbopump inlet pressure engines shut down", k=5)
    assert len({c.source_file for c in cases}) == len(cases) == 5
    assert [c.score for c in cases] == sorted((c.score for c in cases), reverse=True)  # no preference -> by similarity
    real = next(c for c in cases if c.source_file == "historical/real-01.txt")
    assert real.source_type == "historical" and real.category == "propulsion"
    assert "filter blockage" in real.text  # best chunk of the two-chunk case is the informative one


def test_same_category_cases_are_prioritised(rag: HistoricalRAG) -> None:
    query = "tank pressure fell and turbopump inlet pressure dropped"
    plain = rag.search(query, k=5)
    assert plain[0].category != "electrical"  # not the most similar on its own

    boosted = rag.search(query, k=5, prefer_categories={"electrical"})
    electrical = next(c for c in boosted if c.category == "electrical")
    assert electrical.category_match and electrical.rank_score == pytest.approx(electrical.score + 0.05)
    assert electrical.score == pytest.approx(next(c for c in plain if c.category == "electrical").score)  # raw kept
    assert all(not c.category_match for c in boosted if c.category != "electrical")

    strong = HistoricalRAG(rag.store_dir, FakeEmbedder(), category_boost=1.0)
    assert strong.search(query, k=5, prefer_categories={"electrical"})[0].category == "electrical"


def test_anomaly_categories_mapping() -> None:
    events = [_anomaly("lox_tank_pressure", 1, "", "pressurisation"), _anomaly("bus", 2, "", "electrical"),
              _anomaly("fuel", 3, "", "sensor_fault"), _anomaly("imu", 4, "", "gnc")]
    assert anomaly_categories(events) == {"pressurization", "electrical", "avionics"}


def test_rebuild_is_idempotent(rag: HistoricalRAG, docs: Path) -> None:
    assert rag.build(docs) == {"documents": 5, "chunks": 6}
    assert rag.status()[0]


def test_embedder_mismatch_is_reported(rag: HistoricalRAG) -> None:
    other = HistoricalRAG(rag.store_dir, FakeEmbedder("fake:other"))
    ok, reason = other.status()
    assert not ok and "built with fake:hashing" in reason
    with pytest.raises(LLMUnavailableError):
        other.search("anything")


def test_missing_store_reports_build_instruction(tmp_path: Path) -> None:
    store = HistoricalRAG(tmp_path / "empty", FakeEmbedder())
    ok, reason = store.status()
    assert not ok and "build_knowledge_base.py" in reason
    assert store.stats()["cases"] == 0


# --------------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------------- #
def test_anomaly_query_is_time_ordered() -> None:
    later = _anomaly("avionics_bus_voltage", 209.9, "bus voltage dropped")
    earlier = _anomaly("lox_tank_pressure", 142.4, "lox tank pressure dropped")
    query = anomaly_query([later, earlier])
    assert query.index("LOX tank pressure") < query.index("Avionics bus voltage")


def test_format_labels_real_and_synthetic_cases() -> None:
    real = HistoricalCase("Apollo 13", "historical/apollo13.txt", "word " * 400, 0.81234, "electrical", "", "historical")
    synthetic = HistoricalCase("Battery cell failure", "electrical/e1.txt", "text", 0.7, "electrical", "", "synthetic")
    text = format_historical_context([real, synthetic], max_words=50)
    assert text.startswith("### Case 1: Apollo 13\n(REAL historical incident; category: electrical; similarity 0.81")
    assert "synthetic failure-mode summary (generic engineering reference, not a real event)" in text
    assert "..." in text.split("### Case 2")[0]
    assert format_historical_context([]) == "- none retrieved"


def test_top_five_cases_injected_into_qwen_prompt(rag: HistoricalRAG, kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142.4, "lox tank pressure dropped; low turbopump inlet pressure", "pressurisation")
    vib = _anomaly("turbopump_vibration", 143.2, "turbopump vibration rose sharply", "propulsion")
    lox = lox.model_copy(update={"correlated_with": [vib.id]})
    vib = vib.model_copy(update={"correlated_with": [lox.id]})
    client = RecordingClient({"hypotheses": [{
        "title": "Feed-system blockage", "confidence": 0.7,
        "description": "LOX tank pressure fell at T+142.4s, starving the turbopump which vibrated at T+143.2s.",
        "related_anomalies": [lox.label, vib.label],
    }]})
    generator = HypothesisGenerator(kb, router=LLMRouter(client, None), historical_rag=rag)  # type: ignore[arg-type]
    assert HypothesisConfig().historical_cases == 5
    generated = generator.generate([lox, vib])
    assert len(generator.last_historical_cases) == 5
    prompt = client.prompts[0]
    assert "## Historical Context" in prompt
    for i, case in enumerate(generator.last_historical_cases, 1):
        assert f"### Case {i}: {case.title}" in prompt
    matched = {c.category for c in generator.last_historical_cases if c.category_match}
    assert matched <= {"pressurization", "propulsion"} and matched
    assert generated[0].title == "Feed-system blockage"


def test_rag_failure_does_not_break_generation(tmp_path: Path, kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142.4, "lox tank pressure dropped")
    client = RecordingClient({"hypotheses": [{
        "title": "Leak", "confidence": 0.5, "description": "LOX tank pressure leak at T+142.4s.",
        "related_anomalies": [lox.label]}]})
    empty = HistoricalRAG(tmp_path / "no_store", FakeEmbedder())
    generator = HypothesisGenerator(kb, router=LLMRouter(client, None), historical_rag=empty)  # type: ignore[arg-type]
    assert generator.generate([lox])[0].title == "Leak"
    assert generator.last_historical_cases == []
    assert "build_knowledge_base.py" in (generator.last_historical_error or "")
    assert "- none retrieved" in client.prompts[0]


# --------------------------------------------------------------------------- #
# The real, generated knowledge base (data/historical_failures)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def real_documents() -> list:
    return load_documents(KB_DIR)


def test_real_kb_has_100_cases_in_expected_layout(real_documents: list) -> None:
    synthetic = [d for d in real_documents if d.source_type == "synthetic"]
    historical = [d for d in real_documents if d.source_type == "historical"]
    assert len(synthetic) == 80 and len(historical) == 20
    assert Counter(d.category for d in synthetic) == {c: 10 for c in KB_CATEGORIES}
    assert all(d.source_file.startswith(f"{d.category}/") for d in synthetic)
    assert all(d.source_file.startswith("historical/") for d in historical)
    assert all(d.category in KB_CATEGORIES for d in historical)


def test_real_kb_case_lengths_and_sections(real_documents: list) -> None:
    for doc in real_documents:
        words = len(doc.text.split())
        low, high = (400, 500) if doc.source_type == "historical" else (250, 350)
        assert low <= words <= high, (doc.source_file, words)
        for section in SECTIONS:
            assert section in doc.text, (doc.source_file, section)
        if doc.source_type == "synthetic":
            assert "HISTORICAL REFERENCE" in doc.text, doc.source_file
        else:
            assert doc.metadata.get("sources") and doc.metadata.get("date"), doc.source_file


def test_real_kb_builds_and_retrieves_offline(real_documents: list, tmp_path: Path) -> None:
    store = HistoricalRAG(tmp_path / "vs", FakeEmbedder())
    stats = store.build(KB_DIR)
    assert stats["documents"] == 100 and stats["chunks"] >= 100
    cases = store.search("bus undervoltage battery cell short circuit voltage sag", k=5, prefer_categories={"electrical"})
    assert len(cases) == 5 and cases[0].category == "electrical"


# --------------------------------------------------------------------------- #
# Context-window guard
# --------------------------------------------------------------------------- #
def test_truncation_warning_when_prompt_fills_context(caplog: pytest.LogCaptureFixture) -> None:
    class Stub:
        def chat(self, **kwargs: object) -> dict:
            return {"message": {"content": "{}"}, "prompt_eval_count": 4096}

    client = OllamaClient("http://localhost:1", "qwen", num_ctx=4096)
    client._client = Stub()
    with caplog.at_level(logging.WARNING, logger="aerotrace"):
        logging.getLogger("aerotrace").propagate = True
        try:
            client.chat("sys", "x" * 20000)
        finally:
            logging.getLogger("aerotrace").propagate = False
    assert client.last_prompt_tokens == 4096
    assert any("TRUNCATED" in r.getMessage() for r in caplog.records)
