"""Chunk index: chunk sidecar + BM25 (always local) and a pluggable dense backend."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from app.config import Settings, settings
from app.core.interfaces import EmbeddingProvider, VectorStoreBackend
from app.models import Chunk
from app.rag.bm25 import BM25Index
from app.rag.embeddings import build_embedder

logger = logging.getLogger(__name__)
INDEX_FORMAT = 3


class IndexStaleError(RuntimeError):
    """The persisted index was built with a different embedding model or format."""


@dataclass
class LoadedIndex:
    manifest: dict[str, Any]
    chunks: list[Chunk]
    by_id: dict[str, int]
    bm25: BM25Index
    matrix: np.ndarray | None
    mtime: float


class NumpyDenseBackend:
    name = "numpy"

    def __init__(self, indexes_dir: Path) -> None:
        self.indexes_dir = indexes_dir

    def path(self, repo_id: str) -> Path:
        return self.indexes_dir / f"{repo_id}.npy"

    def build(self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        matrix = np.asarray(embeddings, dtype=np.float32) if embeddings else np.zeros((0, 1), dtype=np.float32)
        tmp = self.indexes_dir / f"{repo_id}.tmp.npy"
        np.save(tmp, matrix)
        os.replace(tmp, self.path(repo_id))

    def load(self, repo_id: str) -> np.ndarray | None:
        path = self.path(repo_id)
        return np.load(path) if path.exists() else None

    def search(self, repo_id: str, query_vector: list[float], matrix: np.ndarray | None, k: int, allowed: np.ndarray | None) -> list[tuple[int, float]]:
        if matrix is None or matrix.size == 0 or matrix.shape[1] != len(query_vector):
            return []
        scores = matrix @ np.asarray(query_vector, dtype=np.float32)
        if allowed is not None:
            scores = np.where(allowed, scores, -np.inf)
        k = min(k, scores.shape[0])
        if k <= 0:
            return []
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top if np.isfinite(scores[i])]

    def delete(self, repo_id: str) -> None:
        if self.path(repo_id).exists():
            self.path(repo_id).unlink()


class ChromaDenseBackend:
    name = "chroma"

    def __init__(self, indexes_dir: Path) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed.") from exc
        self.client = chromadb.PersistentClient(path=str(indexes_dir / "chroma"))

    def collection_name(self, repo_id: str) -> str:
        digest = hashlib.sha1(repo_id.encode()).hexdigest()[:12]
        prefix = "".join(char if char.isalnum() else "_" for char in repo_id)[:40].strip("_") or "repo"
        return f"repo_{prefix}_{digest}"

    def build(self, repo_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        self.delete(repo_id)
        collection = self.client.get_or_create_collection(name=self.collection_name(repo_id), metadata={"hnsw:space": "cosine"})
        batch_size = 256
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=embeddings[start : start + batch_size],
                metadatas=[{"file_path": c.file_path, "role": c.role, "language": c.language} for c in batch],
            )

    def load(self, repo_id: str) -> None:
        return None

    def search_ids(self, repo_id: str, query_vector: list[float], k: int, where: dict[str, Any] | None) -> list[tuple[str, float]]:
        try:
            collection = self.client.get_collection(name=self.collection_name(repo_id))
        except Exception as exc:
            raise IndexStaleError(f"Chroma collection missing for {repo_id}") from exc
        count = collection.count()
        if count == 0:
            return []
        result = collection.query(query_embeddings=[query_vector], n_results=min(k, count), where=where or None, include=["distances"])
        ids = result.get("ids", [[]])[0]
        distances = result.get("distances", [[]])[0]
        return [(chunk_id, 1.0 - float(distance)) for chunk_id, distance in zip(ids, distances, strict=False)]

    def delete(self, repo_id: str) -> None:
        try:
            self.client.delete_collection(name=self.collection_name(repo_id))
        except Exception as exc:  # collection absent is the normal case
            logger.debug("Chroma delete for %s: %s", repo_id, exc)


class VectorStore(VectorStoreBackend):
    """Facade: every repo gets a local chunk sidecar + BM25; dense vectors live in numpy or Chroma."""

    def __init__(self, indexes_dir: Path, embedder: EmbeddingProvider | None = None, app_settings: Settings | None = None, cache_size: int = 6) -> None:
        self.settings = app_settings or settings
        self.indexes_dir = indexes_dir
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or build_embedder(self.settings.embedding_provider, self.settings.embedding_model, self.settings.embedding_batch_size)
        self.dense: NumpyDenseBackend | ChromaDenseBackend
        if self.settings.vector_store_provider.lower() == "chroma":
            try:
                self.dense = ChromaDenseBackend(indexes_dir)
            except RuntimeError as exc:
                logger.warning("Chroma unavailable (%s); using local numpy vectors.", exc)
                self.dense = NumpyDenseBackend(indexes_dir)
        else:
            self.dense = NumpyDenseBackend(indexes_dir)
        self._cache: OrderedDict[str, LoadedIndex] = OrderedDict()
        self._lock = threading.Lock()
        self.cache_size = cache_size
        self.resolve_id: Callable[[str], str] = lambda repo_id: repo_id

    @property
    def backend_name(self) -> str:
        return self.dense.name

    def _sidecar(self, repo_id: str) -> Path:
        repo_id = self.resolve_id(repo_id)
        return self.indexes_dir / f"{repo_id}.chunks.json"

    # --- build / lifecycle ------------------------------------------------------------
    def build(self, repo_id: str, chunks: list[Chunk]) -> None:
        repo_id = self.resolve_id(repo_id)
        embeddings = self.embedder.embed_documents([chunk.index_text for chunk in chunks]) if chunks else []
        self.dense.build(repo_id, chunks, embeddings)
        manifest = {
            "format": INDEX_FORMAT,
            "embedding_model": self.embedder.model_id,
            "dense_backend": self.dense.name,
            "chunk_count": len(chunks),
            "dimensions": len(embeddings[0]) if embeddings else 0,
        }
        path = self._sidecar(repo_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"manifest": manifest, "chunks": [chunk.to_dict() for chunk in chunks]}), encoding="utf-8")
        os.replace(tmp, path)
        legacy = self.indexes_dir / f"{repo_id}.json"
        if legacy.exists():
            legacy.unlink()
        with self._lock:
            self._cache.pop(repo_id, None)

    def manifest(self, repo_id: str) -> dict[str, Any] | None:
        path = self._sidecar(repo_id)
        if not path.exists():
            return None
        try:
            with path.open(encoding="utf-8") as handle:
                return json.load(handle).get("manifest")
        except (OSError, json.JSONDecodeError):
            return None

    def compatibility_problem(self, repo_id: str) -> str | None:
        manifest = self.manifest(repo_id)
        if manifest is None:
            return "index missing"
        if manifest.get("format") != INDEX_FORMAT:
            return f"index format {manifest.get('format')} != {INDEX_FORMAT}"
        if manifest.get("embedding_model") != self.embedder.model_id:
            return f"index built with {manifest.get('embedding_model')}, current embedder is {self.embedder.model_id}"
        if manifest.get("dense_backend") != self.dense.name:
            return f"index dense backend {manifest.get('dense_backend')} != {self.dense.name}"
        key = self.resolve_id(repo_id)
        try:
            count = int(manifest["chunk_count"])
            dimensions = int(manifest["dimensions"])
            if isinstance(self.dense, NumpyDenseBackend):
                path = self.dense.path(key)
                if not path.is_file():
                    return "dense matrix missing"
                matrix = np.load(path, mmap_mode="r", allow_pickle=False)
                expected = (count, dimensions if count else 1)
                if matrix.shape != expected or matrix.dtype != np.float32:
                    return "dense matrix shape or dtype differs from manifest"
            else:
                collection = self.dense.client.get_collection(name=self.dense.collection_name(key))
                if collection.count() != count:
                    return "dense collection count differs from manifest"
                if count:
                    embeddings = collection.get(limit=1, include=["embeddings"]).get("embeddings")
                    if embeddings is None or len(embeddings) != 1 or len(embeddings[0]) != dimensions:
                        return "dense collection dimensions differ from manifest"
        except Exception as exc:
            logger.warning("Dense index integrity check failed for %s: %s", repo_id, type(exc).__name__)
            return "dense artifact missing or unreadable"
        return None

    def exists(self, repo_id: str) -> bool:
        problem = self.compatibility_problem(repo_id)
        if problem and problem != "index missing":
            logger.info("Index for %s is stale: %s", repo_id, problem)
        return problem is None

    def delete(self, repo_id: str) -> None:
        repo_id = self.resolve_id(repo_id)
        self.dense.delete(repo_id)
        for path in (self._sidecar(repo_id), self.indexes_dir / f"{repo_id}.json"):
            if path.exists():
                path.unlink()
        with self._lock:
            self._cache.pop(repo_id, None)

    def load(self, repo_id: str) -> LoadedIndex:
        repo_id = self.resolve_id(repo_id)
        problem = self.compatibility_problem(repo_id)
        if problem:
            raise IndexStaleError(f"Index for {repo_id} must be rebuilt: {problem}")
        path = self._sidecar(repo_id)
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(repo_id)
            if cached and cached.mtime == mtime:
                self._cache.move_to_end(repo_id)
                return cached
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(item) for item in payload.get("chunks", [])]
        if len(chunks) != payload["manifest"].get("chunk_count") or len({c.id for c in chunks}) != len(chunks):
            raise IndexStaleError(f"Chunk sidecar for {repo_id} differs from its manifest")
        loaded = LoadedIndex(
            manifest=payload["manifest"],
            chunks=chunks,
            by_id={chunk.id: index for index, chunk in enumerate(chunks)},
            bm25=BM25Index([chunk.index_text for chunk in chunks]),
            matrix=self.dense.load(repo_id),
            mtime=mtime,
        )
        with self._lock:
            self._cache[repo_id] = loaded
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return loaded

    # --- retrieval channels ------------------------------------------------------------
    def _allowed(self, index: LoadedIndex, filters: dict[str, Any] | None) -> set[int] | None:
        if not filters:
            return None
        allowed = set()
        for position, chunk in enumerate(index.chunks):
            if all((getattr(chunk, key, None) if hasattr(chunk, key) else chunk.metadata.get(key)) == value for key, value in filters.items()):
                allowed.add(position)
        return allowed

    def _scored(self, index: LoadedIndex, pairs: list[tuple[int, float]]) -> list[Chunk]:
        results = []
        for position, score in pairs:
            chunk = Chunk.from_dict(index.chunks[position].to_dict())
            chunk.score = round(float(score), 6)
            results.append(chunk)
        return results

    def search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        """Dense (embedding) search."""
        repo_id = self.resolve_id(repo_id)
        index = self.load(repo_id)
        vector = self.embedder.embed_query(query)
        allowed = self._allowed(index, filters)
        if isinstance(self.dense, ChromaDenseBackend):
            where = {"file_path": filters["file_path"]} if filters and set(filters) == {"file_path"} else None
            pairs = [(index.by_id[cid], score) for cid, score in self.dense.search_ids(repo_id, vector, top_k * 2, where) if cid in index.by_id]
            if allowed is not None:
                pairs = [pair for pair in pairs if pair[0] in allowed]
            return self._scored(index, pairs[:top_k])
        mask = None
        if allowed is not None:
            mask = np.zeros(len(index.chunks), dtype=bool)
            mask[list(allowed)] = True
        return self._scored(index, self.dense.search(repo_id, vector, index.matrix, top_k, mask))

    def lexical_search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        index = self.load(repo_id)
        return self._scored(index, index.bm25.top(query, top_k, self._allowed(index, filters)))

    def chunks_for_files(self, repo_id: str, file_paths: set[str], limit_per_file: int = 2, query: str | None = None) -> list[Chunk]:
        """Best chunks within the given files: ranked by BM25 against the query when given."""
        if not file_paths:
            return []
        index = self.load(repo_id)
        positions_by_file: dict[str, list[int]] = {}
        for position, chunk in enumerate(index.chunks):
            if chunk.file_path in file_paths:
                positions_by_file.setdefault(chunk.file_path, []).append(position)
        scores = index.bm25.scores(query) if query else {}
        selected: list[tuple[int, float]] = []
        for positions in positions_by_file.values():
            ranked = sorted(positions, key=lambda p: (-scores.get(p, 0.0), index.chunks[p].start_line))
            selected.extend((p, scores.get(p, 0.0)) for p in ranked[:limit_per_file])
        return self._scored(index, selected)

    def chunks_for_ranges(self, repo_id: str, ranges: list[tuple[str, int, int]]) -> list[Chunk]:
        """Chunks overlapping the given (file, start, end) line ranges, e.g. symbol definitions."""
        if not ranges:
            return []
        index = self.load(repo_id)
        wanted: dict[str, list[tuple[int, int]]] = {}
        for path, start, end in ranges:
            wanted.setdefault(path, []).append((start, end))
        selected = []
        for position, chunk in enumerate(index.chunks):
            for start, end in wanted.get(chunk.file_path, []):
                if chunk.start_line <= end and chunk.end_line >= start:
                    overlap = min(chunk.end_line, end) - max(chunk.start_line, start) + 1
                    selected.append((position, overlap / max(1, end - start + 1)))
                    break
        return self._scored(index, selected)

    def all_chunks(self, repo_id: str) -> list[Chunk]:
        return list(self.load(repo_id).chunks)
