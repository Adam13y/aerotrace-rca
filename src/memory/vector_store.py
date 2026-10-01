"""Vector store used by the knowledge base.

Embeddings come from a deterministic hashed bag-of-words embedder (word unigrams,
bigrams and character trigrams). It is fully offline, fast, and good enough for
matching engineering anomaly descriptions that share vocabulary. Storage uses a
persistent ChromaDB collection when available, falling back to a JSON-backed
in-process store with identical behaviour.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger("memory.vector_store")

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that the this to was were with".split()
)


class HashingEmbedder:
    """Deterministic feature-hashing text embedder."""

    def __init__(self, dimensions: int = 512) -> None:
        """Create an embedder producing ``dimensions``-length unit vectors."""
        self.dimensions = dimensions

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = [t for t in _TOKEN_RE.findall(text.lower().replace("_", " ")) if t not in _STOPWORDS]
        features: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        features += [(f"b:{a}_{b}", 0.7) for a, b in zip(tokens, tokens[1:])]
        for token in tokens:
            padded = f"#{token}#"
            features += [(f"c:{padded[i:i + 3]}", 0.25) for i in range(len(padded) - 2)]
        return features

    def embed(self, text: str) -> list[float]:
        """Embed a single text into an L2-normalised vector."""
        vector = np.zeros(self.dimensions, dtype=np.float64)
        for feature, weight in self._features(text):
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[bucket] += sign * weight
        norm = np.linalg.norm(vector)
        return (vector / norm if norm > 0 else vector).tolist()

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts."""
        return [self.embed(text) for text in texts]


@dataclass(slots=True)
class SearchResult:
    """A single similarity-search hit."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """Persistent similarity store backed by ChromaDB or a JSON file."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection: str = "aerotrace_knowledge",
        embedder: HashingEmbedder | None = None,
        prefer_chroma: bool = True,
    ) -> None:
        """Open (or create) a store.

        Args:
            persist_dir: Directory used for persistence.
            collection: Collection name.
            embedder: Embedder; defaults to :class:`HashingEmbedder`.
            prefer_chroma: Use ChromaDB when importable; otherwise the JSON backend.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection
        self.embedder = embedder or HashingEmbedder()
        self._collection: Any = None
        self._records: dict[str, dict[str, Any]] = {}
        self._json_path = self.persist_dir / f"{collection}.json"
        self.backend = "json"
        if prefer_chroma:
            self._init_chroma()
        if self.backend == "json":
            self._load_json()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError:
            logger.info("chromadb not installed - using JSON vector store")
            return
        try:
            client = chromadb.PersistentClient(
                path=str(self.persist_dir / "chroma"),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=None,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chroma"
        except Exception as exc:  # chromadb raises a variety of backend errors
            logger.warning("ChromaDB unavailable (%s) - using JSON vector store", exc)
            self._collection = None

    def _load_json(self) -> None:
        if self._json_path.exists():
            try:
                self._records = json.loads(self._json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Could not read %s (%s); starting empty", self._json_path, exc)
                self._records = {}

    def _save_json(self) -> None:
        tmp = self._json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._records), encoding="utf-8")
        tmp.replace(self._json_path)

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """Chroma accepts only scalar metadata; serialise anything else to JSON."""
        cleaned: dict[str, str | int | float | bool] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is not None:
                cleaned[key] = json.dumps(value, default=str)
        return cleaned

    def count(self) -> int:
        """Number of stored documents."""
        return int(self._collection.count()) if self.backend == "chroma" else len(self._records)

    def upsert(self, ids: list[str], texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> None:
        """Insert or replace documents.

        Raises:
            ValueError: If argument lengths differ.
        """
        if not ids:
            return
        metadatas = metadatas or [{} for _ in ids]
        if not (len(ids) == len(texts) == len(metadatas)):
            raise ValueError("ids, texts and metadatas must have equal length")
        embeddings = self.embedder.embed_many(texts)
        cleaned = [self._clean_metadata(m) or {"_": ""} for m in metadatas]
        if self.backend == "chroma":
            self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=cleaned)
            return
        for doc_id, text, embedding, metadata in zip(ids, texts, embeddings, cleaned):
            self._records[doc_id] = {"text": text, "embedding": embedding, "metadata": metadata}
        self._save_json()

    def query(
        self, text: str, k: int = 3, min_score: float = 0.0, where: dict[str, str] | None = None
    ) -> list[SearchResult]:
        """Return up to ``k`` most similar documents with cosine similarity >= ``min_score``.

        Args:
            text: Query text.
            k: Maximum number of results.
            min_score: Minimum cosine similarity.
            where: Optional exact-match metadata filter, e.g. ``{"kind": "failure_mode"}``.
        """
        if k <= 0 or self.count() == 0:
            return []
        embedding = self.embedder.embed(text)
        if self.backend == "chroma":
            result = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(k, self.count()),
                where=dict(where) if where else None,
                include=["documents", "metadatas", "distances"],
            )
            hits = [
                SearchResult(id=doc_id, text=doc or "", score=1.0 - float(dist), metadata=dict(meta or {}))
                for doc_id, doc, meta, dist in zip(
                    result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
                )
            ]
        else:
            ids = [
                i for i, record in self._records.items()
                if not where or all(record["metadata"].get(key) == value for key, value in where.items())
            ]
            if not ids:
                return []
            matrix = np.array([self._records[i]["embedding"] for i in ids])
            scores = matrix @ np.asarray(embedding)
            order = np.argsort(-scores)[:k]
            hits = [
                SearchResult(
                    id=ids[i], text=self._records[ids[i]]["text"], score=float(scores[i]),
                    metadata=dict(self._records[ids[i]]["metadata"]),
                )
                for i in order
            ]
        return [hit for hit in hits if hit.score >= min_score]
