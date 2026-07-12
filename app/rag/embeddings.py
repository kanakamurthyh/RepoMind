from __future__ import annotations

import hashlib
from importlib.metadata import PackageNotFoundError, version
import math
import re

from app.core.interfaces import EmbeddingProvider

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[./-][A-Za-z0-9_.-]+")


class LocalHashingEmbedder(EmbeddingProvider):
    """Small deterministic embedding fallback for offline demos and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    @property
    def model_id(self) -> str:
        return f"local-hashing-{self.dimensions}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in TOKEN_RE.findall(text.lower()):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class SentenceTransformerEmbedder(EmbeddingProvider):
    """Semantic embedding provider with lazy optional dependency loading."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._detect_device()
        self._cache: dict[str, list[float]] = {}
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed; using local hashing fallback.") from exc
        self.model = SentenceTransformer(model_name, device=self.device)
        try:
            self.provider_version = version("sentence-transformers")
        except PackageNotFoundError:
            self.provider_version = "unknown"

    @property
    def model_id(self) -> str:
        return f"{self.model_name}|sentence-transformers:{self.provider_version}|device:{self.device}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        missing = [text for text in texts if text not in self._cache]
        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            encoded = self.model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            for text, vector in zip(batch, encoded):
                self._cache[text] = vector.tolist()
        return [self._cache[text] for text in texts]

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
    if provider.lower() in {"semantic", "sentence-transformers", "sentence_transformers"}:
        try:
            return SentenceTransformerEmbedder(model_name, batch_size=batch_size)
        except RuntimeError:
            return LocalHashingEmbedder()
    return LocalHashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))
