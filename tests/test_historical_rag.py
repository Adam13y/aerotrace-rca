"""Tests for historical-failure RAG: chunking, the vector store and prompt injection."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from src.agents.hypothesis_generator import HypothesisGenerator
from src.memory.historical_rag import (
    HistoricalCase,
    HistoricalRAG,
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
DOCS = PROJECT_ROOT / "data" / "historical_failures"
BASE = datetime(2026, 1, 1, tzinfo=UTC)


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


def _anomaly(sensor: str, met: float, severity: Severity, description: str) -> AnomalyEvent:
    ts = BASE + timedelta(seconds=met)
    return AnomalyEvent(
        id=f"ANM-{sensor[:4].upper()}{int(met):04d}", timestamp=ts, end_timestamp=ts + timedelta(seconds=2),
        sensor_id=sensor, severity=severity, description=description, mission_elapsed_s=met, duration_s=2.0,
        baseline_value=50.0, peak_value=44.0, score=18.0, detection_methods=["zscore"],
    )


@pytest.fixture(scope="module")
def rag(tmp_path_factory: pytest.TempPathFactory) -> HistoricalRAG:
    store = HistoricalRAG(tmp_path_factory.mktemp("vector_store"), FakeEmbedder())
    store.build(DOCS)
    return store


@pytest.fixture()
def kb(tmp_path: Path) -> KnowledgeBase:
    return KnowledgeBase(tmp_path, store=VectorStore(tmp_path, prefer_chroma=False))


# --------------------------------------------------------------------------- #
# Documents and chunking
# --------------------------------------------------------------------------- #
def test_historical_documents_present_and_sized() -> None:
    documents = load_documents(DOCS)
    assert {d.source_file for d in documents} == {
        "starship_ift2_engine_out.txt", "apollo13_oxygen_tank.txt", "falcon9_strut_failure.txt",
        "challenger_oring.txt", "columbia_foam_strike.txt",
    }
    for document in documents:
        words = len(document.text.split())
        assert 200 <= words <= 300, (document.source_file, words)
        for section in ("FAILURE MODE", "CAUSAL CHAIN", "TELEMETRY SIGNATURES", "ROOT CAUSE"):
            assert section in document.text, (document.source_file, section)
        assert document.title and document.metadata.get("sources")


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
def test_build_and_search_returns_one_entry_per_case(rag: HistoricalRAG) -> None:
    cases = rag.search("liquid oxygen filter blockage turbopump inlet pressure engine shutdown boostback", k=3)
    assert len(cases) == 3 and len({c.source_file for c in cases}) == 3
    assert cases[0].source_file == "starship_ift2_engine_out.txt"
    assert [c.score for c in cases] == sorted((c.score for c in cases), reverse=True)


def test_search_matches_electrical_signature(rag: HistoricalRAG) -> None:
    cases = rag.search("bus undervoltage fan switched on current transient oxygen tank pressure rise", k=1)
    assert cases[0].source_file == "apollo13_oxygen_tank.txt"


def test_rebuild_is_idempotent(rag: HistoricalRAG) -> None:
    assert rag.build(DOCS) == {"documents": 5, "chunks": 5}
    assert rag.status()[0]


def test_embedder_mismatch_is_reported(rag: HistoricalRAG) -> None:
    other = HistoricalRAG(rag.store_dir, FakeEmbedder("fake:other"))
    ok, reason = other.status()
    assert not ok and "built with fake:hashing" in reason
    with pytest.raises(LLMUnavailableError):
        other.search("anything")


def test_missing_store_reports_build_instruction(tmp_path: Path) -> None:
    ok, reason = HistoricalRAG(tmp_path / "empty", FakeEmbedder()).status()
    assert not ok and "build_knowledge_base.py" in reason


# --------------------------------------------------------------------------- #
# Prompt injection
# --------------------------------------------------------------------------- #
def test_anomaly_query_is_time_ordered() -> None:
    later = _anomaly("avionics_bus_voltage", 209.9, Severity.CRITICAL, "bus voltage dropped")
    earlier = _anomaly("lox_tank_pressure", 142.4, Severity.CRITICAL, "lox tank pressure dropped")
    query = anomaly_query([later, earlier])
    assert query.index("LOX tank pressure") < query.index("Avionics bus voltage")


def test_format_historical_context() -> None:
    case = HistoricalCase("Apollo 13", "apollo13_oxygen_tank.txt", "word " * 400, 0.81234)
    text = format_historical_context([case], max_words=50)
    assert text.startswith("### Case 1: Apollo 13\n(similarity 0.81, source: apollo13_oxygen_tank.txt)")
    assert text.endswith("...")
    assert format_historical_context([]) == "- none retrieved"


def test_top_three_cases_injected_into_qwen_prompt(rag: HistoricalRAG, kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142.4, Severity.CRITICAL,
                   "lox tank pressure dropped; low turbopump inlet pressure")
    vib = _anomaly("turbopump_vibration", 143.2, Severity.CRITICAL, "turbopump vibration rose sharply")
    lox = lox.model_copy(update={"correlated_with": [vib.id]})
    vib = vib.model_copy(update={"correlated_with": [lox.id]})
    client = RecordingClient({"hypotheses": [{
        "title": "Feed-system blockage", "confidence": 0.7,
        "description": "LOX tank pressure fell at T+142.4s, starving the turbopump which vibrated at T+143.2s.",
        "related_anomalies": [lox.label, vib.label],
    }]})
    generator = HypothesisGenerator(kb, router=LLMRouter(client, None), historical_rag=rag)  # type: ignore[arg-type]
    generated = generator.generate([lox, vib])
    assert len(generator.last_historical_cases) == 3
    prompt = client.prompts[0]
    assert "## Historical Context" in prompt
    for i, case in enumerate(generator.last_historical_cases, 1):
        assert f"### Case {i}: {case.title}" in prompt
    assert generated[0].title == "Feed-system blockage"


def test_rag_failure_does_not_break_generation(tmp_path: Path, kb: KnowledgeBase) -> None:
    lox = _anomaly("lox_tank_pressure", 142.4, Severity.CRITICAL, "lox tank pressure dropped")
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
