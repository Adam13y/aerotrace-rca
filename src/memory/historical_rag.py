"""Retrieval-augmented generation over the aerospace failure knowledge base.

Text files under ``data/historical_failures/`` (scanned recursively: one sub-folder per failure
category plus ``historical/`` for real incidents) are split into ~500-token chunks, embedded
locally with ``nomic-embed-text`` through Ollama and stored in a persistent ChromaDB collection
under ``data/vector_store/`` together with their category, failure mode and source type
(``synthetic`` engineering summary or ``historical`` real incident).

At analysis time the detected anomalies are turned into a query; the most similar *cases*
(files, not chunks) are returned, with cases from the anomalies' own failure categories ranked
ahead of comparably similar cases from other categories.

Generate the documents with ``python scripts/acquire_aerospace_data.py`` and build the store
with ``python scripts/build_knowledge_base.py``.
"""

from __future__ import annotations

import fnmatch
import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.models.telemetry_schema import AnomalyEvent, sensor_display_name
from src.utils.helpers import LLMUnavailableError, list_ollama_models, ollama_model_installed
from src.utils.logger import get_logger

logger = get_logger("memory.historical_rag")

COLLECTION_NAME = "historical_failures"
#: Rough English tokens-per-word ratio used for chunk sizing (no tokenizer dependency).
TOKENS_PER_WORD = 4 / 3
#: Knowledge-base failure categories (sub-folders of ``data/historical_failures``).
KB_CATEGORIES = (
    "propulsion", "pressurization", "structural", "thermal",
    "electrical", "avionics", "separation", "aerodynamic",
)
#: Triage categories (anomaly classification) -> knowledge-base categories.
TRIAGE_TO_KB_CATEGORY = {
    "propulsion": "propulsion",
    "pressurisation": "pressurization",
    "pressurization": "pressurization",
    "structural": "structural",
    "thermal": "thermal",
    "electrical": "electrical",
    "gnc": "avionics",
    "avionics": "avionics",
    "separation": "separation",
    "aerodynamic": "aerodynamic",
}


#: Default sensor -> knowledge-base category mapping (overridden by ``historical_rag.sensor_categories``).
DEFAULT_SENSOR_CATEGORIES: dict[str, str] = {
    "turbopump_vibration": "propulsion",
    "chamber_pressure": "propulsion",
    "fuel_tank_pressure": "propulsion",
    "lox_tank_pressure": "propulsion",
    "engine_bay_temp": "thermal",
    "avionics_bus_voltage": "electrical",
}
#: Fallback glob patterns for sensors not listed explicitly (first match wins).
DEFAULT_SENSOR_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("*turbopump*", "propulsion"), ("*chamber*", "propulsion"), ("*injector*", "propulsion"),
    ("*tank_pressure*", "propulsion"), ("*pump_inlet*", "propulsion"),
    ("*pressurant*", "pressurization"), ("*copv*", "pressurization"), ("*helium*", "pressurization"),
    ("*ullage*", "pressurization"), ("*regulator*", "pressurization"),
    ("*voltage*", "electrical"), ("*current*", "electrical"), ("*battery*", "electrical"),
    ("*temp*", "thermal"), ("*thermocouple*", "thermal"), ("*heat_flux*", "thermal"),
    ("*strain*", "structural"), ("*accel*", "structural"),
    ("*imu*", "avionics"), ("*gyro*", "avionics"), ("*gps*", "avionics"), ("*attitude*", "avionics"),
    ("*star_tracker*", "avionics"),
    ("*fin*", "aerodynamic"), ("*dynamic_pressure*", "aerodynamic"), ("*angle_of_attack*", "aerodynamic"),
    ("*separation*", "separation"), ("*pyro*", "separation"),
)


def sensor_category(
    sensor_id: str,
    mapping: dict[str, str] | None = None,
    patterns: Sequence[tuple[str, str]] | None = None,
) -> str | None:
    """Knowledge-base category of a sensor: exact mapping first, then glob patterns, else ``None``."""
    mapping = DEFAULT_SENSOR_CATEGORIES if mapping is None else mapping
    patterns = DEFAULT_SENSOR_CATEGORY_PATTERNS if patterns is None else patterns
    key = sensor_id.lower()
    if key in mapping:
        return mapping[key]
    return next((category for pattern, category in patterns if fnmatch.fnmatch(key, pattern)), None)


class Embedder(Protocol):
    """Anything that embeds documents and queries into the same vector space."""

    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts to be stored."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query."""
        ...


class OllamaEmbedder:
    """Local embeddings via Ollama (default model: ``nomic-embed-text``).

    nomic-embed-text expects task prefixes; they are added automatically.
    """

    def __init__(self, base_url: str, model: str = "nomic-embed-text", timeout: float = 120.0) -> None:
        """Create the embedder.

        Args:
            base_url: Ollama server URL.
            model: Embedding model tag.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url
        self.model = model
        self.name = f"ollama:{model}"
        self.timeout = timeout
        self._client: Any = None
        self._prefixes = ("search_document: ", "search_query: ") if model.startswith("nomic-embed-text") else ("", "")

    def available(self) -> bool:
        """Whether Ollama is running and the embedding model is installed."""
        try:
            return ollama_model_installed(self.model, list_ollama_models(self.base_url))
        except LLMUnavailableError:
            return False

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if self._client is None:
            try:
                import ollama
            except ImportError as exc:
                raise LLMUnavailableError("The 'ollama' package is not installed") from exc
            self._client = ollama.Client(host=self.base_url, timeout=self.timeout)
        try:
            response = self._client.embed(model=self.model, input=list(texts))
        except Exception as exc:  # ollama raises httpx / ResponseError variants
            raise LLMUnavailableError(f"Ollama embedding request failed: {exc}") from exc
        return [list(map(float, vector)) for vector in response["embeddings"]]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed documents with the model's document prefix."""
        return self._embed([self._prefixes[0] + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the model's query prefix."""
        return self._embed([self._prefixes[1] + text])[0]


@dataclass(slots=True)
class HistoricalDocument:
    """A knowledge-base report loaded from disk."""

    source_file: str
    title: str
    text: str
    category: str = "uncategorized"
    failure_mode: str = ""
    source_type: str = "custom"
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class HistoricalCase:
    """A retrieved case: the best-matching chunk of one report."""

    title: str
    source_file: str
    text: str
    score: float
    category: str = "uncategorized"
    failure_mode: str = ""
    source_type: str = "custom"
    category_match: bool = False
    rank_score: float = 0.0


def estimate_tokens(text: str) -> int:
    """Approximate token count from the word count."""
    return math.ceil(len(text.split()) * TOKENS_PER_WORD)


def chunk_text(text: str, chunk_tokens: int = 500, overlap_tokens: int = 50) -> list[str]:
    """Split ``text`` into chunks of about ``chunk_tokens`` tokens with ``overlap_tokens`` overlap.

    Raises:
        ValueError: If the overlap is not smaller than the chunk size.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")
    words = text.split()
    if not words:
        return []
    size = max(1, int(chunk_tokens / TOKENS_PER_WORD))
    step = max(1, size - int(overlap_tokens / TOKENS_PER_WORD))
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def load_documents(directory: str | Path) -> list[HistoricalDocument]:
    """Recursively load ``*.txt`` reports.

    ``Key: value`` lines before the first blank line are metadata. ``Category``/``Source type``
    fall back to the folder layout: ``historical/`` -> source type ``historical``; a category
    folder -> that category with source type ``synthetic``; anything else -> ``custom``.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Historical failures directory not found: {directory}")
    documents: list[HistoricalDocument] = []
    for path in sorted(directory.rglob("*.txt")):
        if any(part.startswith(".") for part in path.relative_to(directory).parts):
            continue
        raw = path.read_text(encoding="utf-8").strip()
        header, _, body = raw.partition("\n\n")
        metadata = {
            key.strip().lower(): value.strip()
            for key, sep, value in (line.partition(":") for line in header.splitlines())
            if sep
        }
        if "title" not in metadata:  # no header block: the whole file is the body
            metadata, body = {}, raw
        folder = path.parent.name.lower() if path.parent != directory else ""
        category = metadata.get("category") or (folder if folder in KB_CATEGORIES else "uncategorized")
        source_type = metadata.get("source type") or (
            "historical" if folder == "historical" else "synthetic" if folder in KB_CATEGORIES else "custom"
        )
        documents.append(
            HistoricalDocument(
                source_file=path.relative_to(directory).as_posix(),
                title=metadata.get("title", path.stem.replace("_", " ")),
                text=body.strip(),
                category=category.lower(),
                failure_mode=metadata.get("failure mode", ""),
                source_type=source_type.lower(),
                metadata=metadata,
            )
        )
    return documents


def anomaly_categories(anomalies: Iterable[AnomalyEvent]) -> set[str]:
    """Knowledge-base categories implied by the anomalies' triage classifications."""
    return {
        TRIAGE_TO_KB_CATEGORY[c]
        for a in anomalies
        if (c := str(a.classification.get("category", "")).lower()) in TRIAGE_TO_KB_CATEGORY
    }


class HistoricalRAG:
    """ChromaDB-backed retrieval of knowledge-base cases."""

    def __init__(
        self,
        store_dir: str | Path,
        embedder: Embedder,
        chunk_tokens: int = 500,
        overlap_tokens: int = 50,
        category_boost: float = 0.05,
    ) -> None:
        """Open the vector store.

        Args:
            store_dir: ChromaDB persistence directory (``data/vector_store``).
            embedder: Embedding backend; must match the one used to build the store.
            chunk_tokens: Target chunk size in tokens.
            overlap_tokens: Overlap between consecutive chunks.
            category_boost: Similarity bonus for cases in the anomalies' failure categories.
        """
        self.store_dir = Path(store_dir)
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.category_boost = category_boost
        self._client: Any = None

    def _chroma(self) -> Any:
        if self._client is None:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self.store_dir.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(self.store_dir), settings=ChromaSettings(anonymized_telemetry=False)
            )
        return self._client

    def _collection(self) -> Any | None:
        try:
            return self._chroma().get_collection(COLLECTION_NAME, embedding_function=None)
        except Exception:  # chromadb raises NotFoundError / ValueError depending on version
            return None

    def build(self, docs_dir: str | Path, progress: Callable[[str], None] | None = None) -> dict[str, int]:
        """(Re)build the collection from every report under ``docs_dir`` (recursively).

        Returns:
            ``{"documents": n, "chunks": m}``.

        Raises:
            FileNotFoundError: If ``docs_dir`` is missing.
            ValueError: If it contains no ``.txt`` reports.
        """
        documents = load_documents(docs_dir)
        if not documents:
            raise ValueError(f"No .txt reports found in {docs_dir}")
        client = self._chroma()
        if self._collection() is not None:
            client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedder.name, "chunk_tokens": self.chunk_tokens},
        )
        total = 0
        for document in documents:
            chunks = chunk_text(document.text, self.chunk_tokens, self.overlap_tokens)
            # Prefix each chunk with the title so every chunk carries its case context.
            texts = [f"{document.title}\n\n{chunk}" for chunk in chunks]
            collection.add(
                ids=[f"{document.source_file}#{i}" for i in range(len(texts))],
                documents=texts,
                embeddings=self.embedder.embed_documents(texts),
                metadatas=[
                    {
                        "source_file": document.source_file,
                        "title": document.title,
                        "category": document.category,
                        "failure_mode": document.failure_mode,
                        "source_type": document.source_type,
                        "chunk": i,
                        "tokens": estimate_tokens(chunk),
                    }
                    for i, chunk in enumerate(chunks)
                ],
            )
            total += len(texts)
            if progress:
                progress(f"{document.source_file}: {len(texts)} chunk(s), ~{estimate_tokens(document.text)} tokens")
        logger.info("Built knowledge base: %d documents, %d chunks", len(documents), total)
        return {"documents": len(documents), "chunks": total}

    def status(self) -> tuple[bool, str]:
        """Whether retrieval can run, with a human-readable reason."""
        collection = self._collection()
        if collection is None or collection.count() == 0:
            return False, f"vector store empty or missing at {self.store_dir} (run scripts/build_knowledge_base.py)"
        built_with = (collection.metadata or {}).get("embedding_model")
        if built_with and built_with != self.embedder.name:
            return False, f"store was built with {built_with} but the configured embedder is {self.embedder.name}"
        available = getattr(self.embedder, "available", None)
        if callable(available) and not available():
            return False, f"embedding model {self.embedder.name} is not available"
        return True, f"{collection.count()} chunks indexed with {self.embedder.name}"

    def stats(self) -> dict[str, Any]:
        """Counts of indexed chunks and cases, by category and source type (empty if not built)."""
        collection = self._collection()
        if collection is None:
            return {"chunks": 0, "cases": 0, "by_category": {}, "by_source_type": {}, "embedding_model": None}
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        cases = {m.get("source_file"): m for m in metadatas}
        return {
            "chunks": len(metadatas),
            "cases": len(cases),
            "by_category": dict(Counter(m.get("category", "?") for m in cases.values())),
            "by_source_type": dict(Counter(m.get("source_type", "?") for m in cases.values())),
            "embedding_model": (collection.metadata or {}).get("embedding_model"),
        }

    def search(self, query: str, k: int = 5, prefer_categories: Iterable[str] | None = None) -> list[HistoricalCase]:
        """Return the ``k`` best cases (best chunk per report).

        Cases whose category is in ``prefer_categories`` get ``category_boost`` added to their
        cosine similarity for ranking, so same-category cases come first unless a case from
        another category is clearly more similar. ``score`` stays the raw similarity.

        Raises:
            LLMUnavailableError: If the store or embedder is unusable.
        """
        ok, reason = self.status()
        if not ok:
            raise LLMUnavailableError(reason)
        preferred = set(prefer_categories or ())
        collection = self._collection()
        result = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(collection.count(), max(k * 10, 50)),
            include=["documents", "metadatas", "distances"],
        )
        best: dict[str, HistoricalCase] = {}
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            source = str(meta.get("source_file", "?"))
            score = 1.0 - float(distance)
            if source in best and score <= best[source].score:
                continue
            category = str(meta.get("category", "uncategorized"))
            match = category in preferred
            body = text.split("\n\n", 1)[1] if "\n\n" in text else text
            best[source] = HistoricalCase(
                title=str(meta.get("title", source)),
                source_file=source,
                text=body,
                score=score,
                category=category,
                failure_mode=str(meta.get("failure_mode", "")),
                source_type=str(meta.get("source_type", "custom")),
                category_match=match,
                rank_score=score + (self.category_boost if match else 0.0),
            )
        return sorted(best.values(), key=lambda c: (c.rank_score, c.score), reverse=True)[:k]

    def search_for_anomalies(
        self,
        anomalies: Sequence[AnomalyEvent],
        k: int = 5,
        prefer_categories: Iterable[str] | None = None,
    ) -> list[HistoricalCase]:
        """Retrieve cases similar to a set of anomalies.

        Args:
            anomalies: Anomalies to describe in the query.
            k: Number of cases.
            prefer_categories: Categories that receive the similarity bonus; defaults to the
                categories implied by the anomalies' triage labels.
        """
        preferred = anomaly_categories(anomalies) if prefer_categories is None else prefer_categories
        return self.search(anomaly_query(anomalies), k, prefer_categories=preferred)


def describe_change(anomaly: AnomalyEvent) -> str:
    """Plain physical description of one anomaly, e.g. ``LOX tank pressure drop of 12%``."""
    text = f"{sensor_display_name(anomaly.sensor_id)} {anomaly.change_kind}"
    baseline, peak = anomaly.baseline_value, anomaly.peak_value
    if baseline is not None and peak is not None and abs(baseline) > 1e-9:
        percent = abs(peak - baseline) / abs(baseline) * 100.0
        text += f" of {percent:.0f}%" if percent >= 1 else f" of {percent:.1f}%"
    return text


def anomaly_query(anomalies: Sequence[AnomalyEvent]) -> str:
    """Build the retrieval query as a plain physical failure signature, in time order.

    Example: ``Failure signature: LOX tank pressure drop of 12%, then Turbopump vibration rise
    of 289% 0.8 s later, then Engine bay temperature rise of 37% 5.3 s later.``

    Only physical content goes into the query: detector details (z-scores, sigma values,
    detection methods), triage labels and absolute timestamps are left out because they are
    not failure vocabulary and dilute the similarity search. Time deltas are measured from the
    previous anomaly's start.
    """
    parts: list[str] = []
    previous: float | None = None
    for anomaly in sorted(anomalies, key=lambda a: a.timestamp):
        part = describe_change(anomaly)
        start = anomaly.mission_elapsed_s
        if previous is not None and start is not None:
            part += f" {start - previous:.1f} s later"
        parts.append(part)
        previous = start if start is not None else previous
    return "Failure signature: " + ", then ".join(parts) + "." if parts else "Failure signature: none."


def format_historical_context(cases: Sequence[HistoricalCase], max_words: int = 150) -> str:
    """Render retrieved cases for the hypothesis prompt, labelling real vs. synthetic cases."""
    if not cases:
        return "- none retrieved"
    kinds = {
        "historical": "REAL historical incident",
        "synthetic": "synthetic failure-mode summary (generic engineering reference, not a real event)",
    }
    blocks = []
    for i, case in enumerate(cases, 1):
        words = re.sub(r"\s+", " ", case.text).split()
        excerpt = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        blocks.append(
            f"### Case {i}: {case.title}\n"
            f"({kinds.get(case.source_type, 'reference case')}; category: {case.category}; "
            f"similarity {case.score:.2f}; source: {case.source_file})\n{excerpt}"
        )
    return "\n\n".join(blocks)
