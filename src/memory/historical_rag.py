"""Retrieval-augmented generation over historical aerospace failure reports.

Text files in ``data/historical_failures/`` are split into ~500-token chunks, embedded
locally with ``nomic-embed-text`` through Ollama and stored in a persistent ChromaDB
collection under ``data/vector_store/``. At analysis time the detected anomalies are
turned into a query and the most similar historical *cases* (files, not chunks) are
returned for injection into the hypothesis prompt.

Build the store with ``python scripts/build_knowledge_base.py``.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from src.models.telemetry_schema import AnomalyEvent
from src.utils.helpers import LLMUnavailableError, list_ollama_models, ollama_model_installed
from src.utils.logger import get_logger

logger = get_logger("memory.historical_rag")

COLLECTION_NAME = "historical_failures"
#: Rough English tokens-per-word ratio used for chunk sizing (no tokenizer dependency).
TOKENS_PER_WORD = 4 / 3


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

    def __init__(self, base_url: str, model: str = "nomic-embed-text", timeout: float = 60.0) -> None:
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
    """A historical failure report loaded from disk."""

    source_file: str
    title: str
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class HistoricalCase:
    """A retrieved historical case: the best-matching chunk of one report."""

    title: str
    source_file: str
    text: str
    score: float


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
    """Load ``*.txt`` reports. ``Key: value`` lines before the first blank line are metadata.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Historical failures directory not found: {directory}")
    documents: list[HistoricalDocument] = []
    for path in sorted(directory.glob("*.txt")):
        raw = path.read_text(encoding="utf-8").strip()
        header, _, body = raw.partition("\n\n")
        metadata = {
            key.strip().lower(): value.strip()
            for key, sep, value in (line.partition(":") for line in header.splitlines())
            if sep
        }
        if "title" not in metadata:  # no header block: the whole file is the body
            metadata, body = {}, raw
        documents.append(
            HistoricalDocument(path.name, metadata.get("title", path.stem.replace("_", " ")), body.strip(), metadata)
        )
    return documents


class HistoricalRAG:
    """ChromaDB-backed retrieval of historical failure cases."""

    def __init__(
        self,
        store_dir: str | Path,
        embedder: Embedder,
        chunk_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> None:
        """Open the vector store.

        Args:
            store_dir: ChromaDB persistence directory (``data/vector_store``).
            embedder: Embedding backend; must match the one used to build the store.
            chunk_tokens: Target chunk size in tokens.
            overlap_tokens: Overlap between consecutive chunks.
        """
        self.store_dir = Path(store_dir)
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
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
        """(Re)build the collection from all reports in ``docs_dir``.

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
                    {"source_file": document.source_file, "title": document.title, "chunk": i,
                     "tokens": estimate_tokens(chunk)}
                    for i, chunk in enumerate(chunks)
                ],
            )
            total += len(texts)
            if progress:
                progress(f"{document.source_file}: {len(texts)} chunk(s), ~{estimate_tokens(document.text)} tokens")
        logger.info("Built historical knowledge base: %d documents, %d chunks", len(documents), total)
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

    def search(self, query: str, k: int = 3) -> list[HistoricalCase]:
        """Return the ``k`` most similar historical cases (best chunk per report).

        Raises:
            LLMUnavailableError: If the store or embedder is unusable.
        """
        ok, reason = self.status()
        if not ok:
            raise LLMUnavailableError(reason)
        collection = self._collection()
        result = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=min(collection.count(), max(k * 4, k)),
            include=["documents", "metadatas", "distances"],
        )
        best: dict[str, HistoricalCase] = {}
        for text, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            source = str(meta.get("source_file", "?"))
            score = 1.0 - float(distance)
            if source not in best or score > best[source].score:
                body = text.split("\n\n", 1)[1] if "\n\n" in text else text
                best[source] = HistoricalCase(str(meta.get("title", source)), source, body, score)
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]

    def search_for_anomalies(self, anomalies: Sequence[AnomalyEvent], k: int = 3) -> list[HistoricalCase]:
        """Retrieve cases similar to a set of anomalies (see :func:`anomaly_query`)."""
        return self.search(anomaly_query(anomalies), k)


def anomaly_query(anomalies: Sequence[AnomalyEvent]) -> str:
    """Build a retrieval query describing anomalies in time order (signature + classification)."""
    lines = []
    for anomaly in sorted(anomalies, key=lambda a: a.timestamp):
        category = anomaly.classification.get("category", "")
        subsystem = anomaly.classification.get("subsystem", "")
        lines.append(f"{anomaly.label}; {category} {subsystem}; {anomaly.description}")
    return "Rocket flight telemetry anomaly sequence:\n" + "\n".join(lines)


def format_historical_context(cases: Sequence[HistoricalCase], max_words: int = 220) -> str:
    """Render retrieved cases for the hypothesis prompt."""
    if not cases:
        return "- none retrieved"
    blocks = []
    for i, case in enumerate(cases, 1):
        words = re.sub(r"\s+", " ", case.text).split()
        excerpt = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
        blocks.append(f"### Case {i}: {case.title}\n(similarity {case.score:.2f}, source: {case.source_file})\n{excerpt}")
    return "\n\n".join(blocks)
