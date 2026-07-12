from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.interfaces import EmbeddingProvider, VectorStoreBackend
from app.models import Chunk
from app.rag.embeddings import LocalHashingEmbedder, build_embedder, cosine


class LocalJsonVectorStore(VectorStoreBackend):
    """Persistent local fallback vector store.

    Chroma is the preferred provider for real semantic retrieval. This fallback
    preserves offline testability and is used automatically when optional vector
    DB dependencies are unavailable.
    """

    def __init__(self, indexes_dir: Path, embedder: EmbeddingProvider | None = None) -> None:
        self.indexes_dir = indexes_dir
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or LocalHashingEmbedder()

    def build(self, repo_id: str, chunks: list[Chunk]) -> None:
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed_documents(texts)
        rows = [
            {"chunk": chunk.to_dict(), "embedding": embedding}
            for chunk, embedding in zip(chunks, embeddings)
        ]
        manifest = {
            "provider": "local-json",
            "embedding_model": self.embedder.model_id,
            "chunk_count": len(chunks),
        }
        path = self._path(repo_id)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({"manifest": manifest, "rows": rows}), encoding="utf-8")
        os.replace(tmp_path, path)

    def search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        rows = self._load_rows(repo_id)
        if not rows:
            return []
        query_embedding = self.embedder.embed_query(query)
        scored: list[Chunk] = []
        for row in rows:
            chunk = Chunk.from_dict(row["chunk"])
            if filters and not self._matches(chunk, filters):
                continue
            chunk.score = cosine(query_embedding, row["embedding"])
            scored.append(chunk)
        scored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return scored[:top_k]

    def chunks_for_files(self, repo_id: str, file_paths: set[str], limit_per_file: int = 2) -> list[Chunk]:
        if not file_paths:
            return []
        rows = self._load_rows(repo_id)
        selected: list[Chunk] = []
        counts = {path: 0 for path in file_paths}
        for row in rows:
            chunk = Chunk.from_dict(row["chunk"])
            if chunk.file_path in file_paths and counts[chunk.file_path] < limit_per_file:
                counts[chunk.file_path] += 1
                chunk.score = 0.0
                selected.append(chunk)
        return selected

    def exists(self, repo_id: str) -> bool:
        return self._path(repo_id).exists()

    def delete(self, repo_id: str) -> None:
        path = self._path(repo_id)
        if path.exists():
            path.unlink()

    def _path(self, repo_id: str) -> Path:
        return self.indexes_dir / f"{repo_id}.json"

    def _load_rows(self, repo_id: str) -> list[dict]:
        path = self._path(repo_id)
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Vector index for {repo_id} is corrupted and should be rebuilt.") from exc
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise RuntimeError(f"Vector index for {repo_id} has an invalid format.")
        return rows

    def _matches(self, chunk: Chunk, filters: dict[str, Any]) -> bool:
        for key, expected in filters.items():
            actual = getattr(chunk, key, chunk.metadata.get(key))
            if actual != expected:
                return False
        return True


class ChromaVectorStore(VectorStoreBackend):
    """Chroma-backed vector store for persistent semantic retrieval."""

    def __init__(self, indexes_dir: Path, embedder: EmbeddingProvider) -> None:
        try:
            import chromadb  # type: ignore
        except ImportError as exc:
            raise RuntimeError("chromadb is not installed.") from exc
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(indexes_dir / "chroma"))

    def build(self, repo_id: str, chunks: list[Chunk]) -> None:
        self.delete(repo_id)
        collection = self.client.get_or_create_collection(name=self._collection(repo_id))
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        batch_size = 128
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            collection.add(
                ids=[chunk.id for chunk in batch],
                embeddings=embeddings[start : start + batch_size],
                documents=[chunk.text for chunk in batch],
                metadatas=[
                    {
                        "repo_id": chunk.repo_id,
                        "file_path": chunk.file_path,
                        "language": chunk.language,
                        "role": chunk.role,
                        "start_line": chunk.start_line,
                        "end_line": chunk.end_line,
                        "symbol": chunk.symbol or "",
                    }
                    for chunk in batch
                ],
            )

    def search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        collection = self.client.get_or_create_collection(name=self._collection(repo_id))
        where = filters or {}
        result = collection.query(
            query_embeddings=[self.embedder.embed_query(query)],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        chunks: list[Chunk] = []
        for doc, meta, distance in zip(result.get("documents", [[]])[0], result.get("metadatas", [[]])[0], result.get("distances", [[]])[0]):
            chunks.append(
                Chunk(
                    id="",
                    repo_id=repo_id,
                    file_path=meta["file_path"],
                    language=meta.get("language", ""),
                    role=meta.get("role", ""),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                    text=doc,
                    score=1.0 / (1.0 + float(distance)),
                    symbol=meta.get("symbol") or None,
                )
            )
        return chunks

    def chunks_for_files(self, repo_id: str, file_paths: set[str], limit_per_file: int = 2) -> list[Chunk]:
        chunks: list[Chunk] = []
        for file_path in file_paths:
            chunks.extend(self.search(repo_id, file_path, top_k=limit_per_file, filters={"file_path": file_path}))
        return chunks

    def delete(self, repo_id: str) -> None:
        try:
            self.client.delete_collection(name=self._collection(repo_id))
        except Exception:
            pass

    def exists(self, repo_id: str) -> bool:
        try:
            self.client.get_collection(name=self._collection(repo_id))
            return True
        except Exception:
            return False

    def _collection(self, repo_id: str) -> str:
        return "repo_" + "".join(char if char.isalnum() else "_" for char in repo_id)[:50]


class VectorStore(VectorStoreBackend):
    """Facade selecting Chroma when available, otherwise local persistent fallback."""

    def __init__(self, indexes_dir: Path, embedder: EmbeddingProvider | None = None) -> None:
        self.embedder = embedder or build_embedder(settings.embedding_provider, settings.embedding_model, settings.embedding_batch_size)
        if settings.vector_store_provider.lower() == "chroma":
            try:
                self.backend: VectorStoreBackend = ChromaVectorStore(indexes_dir, self.embedder)
            except RuntimeError:
                self.backend = LocalJsonVectorStore(indexes_dir, self.embedder)
        else:
            self.backend = LocalJsonVectorStore(indexes_dir, self.embedder)

    def build(self, repo_id: str, chunks: list[Chunk]) -> None:
        self.backend.build(repo_id, chunks)

    def search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        return self.backend.search(repo_id, query, top_k, filters)

    def chunks_for_files(self, repo_id: str, file_paths: set[str], limit_per_file: int = 2) -> list[Chunk]:
        return self.backend.chunks_for_files(repo_id, file_paths, limit_per_file)

    def delete(self, repo_id: str) -> None:
        self.backend.delete(repo_id)

    def exists(self, repo_id: str) -> bool:
        return self.backend.exists(repo_id)
