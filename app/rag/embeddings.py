from __future__ import annotations

import hashlib
import logging
import math
import threading
from collections import Counter, OrderedDict
from importlib.metadata import PackageNotFoundError, version

from app.core.interfaces import EmbeddingProvider
from app.rag.model_identity import model_revision
from app.rag.tokenize import code_tokens

logger = logging.getLogger(__name__)


class LocalHashingEmbedder(EmbeddingProvider):
    """Deterministic offline embedding: signed feature hashing of code-aware tokens.

    It is lexical, not semantic - RepoMind pairs it with BM25, symbol and graph
    channels. Use EMBEDDING_PROVIDER=sentence-transformers for semantic vectors.
    """

    def __init__(self, dimensions: int = 512) -> None:
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"local-hashing-v3-{self.dimensions}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token, count in Counter(code_tokens(text)).items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Semantic embedding provider with lazy optional dependency loading and a bounded cache."""

    def __init__(self, model_name: str, batch_size: int = 32, cache_size: int = 2048, device: str | None = None) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.device = device or self._detect_device()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_lock = threading.Lock()
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        try:
            self.model = SentenceTransformer(model_name, device=self.device, trust_remote_code=False)
        except Exception as exc:  # model download / load failure
            raise RuntimeError(f"could not load embedding model {model_name}: {exc}") from exc
        module = self.model._first_module()
        self.revision = model_revision(model_name, getattr(getattr(module, "auto_model", None), "config", None))
        try:
            self.provider_version = version("sentence-transformers")
        except PackageNotFoundError:
            self.provider_version = "unknown"

    @property
    def model_id(self) -> str:
        # Device is intentionally excluded: vectors are compatible across devices.
        return f"st:{self.model_name}@{self.revision or 'unversioned'}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            vectors.extend(vector.tolist() for vector in encoded)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        with self._cache_lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached
        vector = self.embed_documents([text])[0]
        with self._cache_lock:
            self._cache[text] = vector
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return vector

    def _detect_device(self) -> str:
        try:
            import torch  # type: ignore
        except ImportError:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def build_embedder(provider: str, model_name: str, batch_size: int) -> EmbeddingProvider:
    """`local` -> hashing; `sentence-transformers`/`semantic` -> required; `auto` -> semantic if available."""
    choice = (provider or "local").lower()
    if choice in {"semantic", "sentence-transformers", "sentence_transformers", "auto"}:
        try:
            return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
        except RuntimeError as exc:
            logger.warning("Semantic embeddings unavailable (%s); using local hashing embedder.", exc)
    return LocalHashingEmbedder()
