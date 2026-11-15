from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.models import Chunk, FileRecord


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class VectorStoreBackend(ABC):
    @abstractmethod
    def build(self, repo_id: str, chunks: list[Chunk]) -> None:
        raise NotImplementedError

    @abstractmethod
    def search(self, repo_id: str, query: str, top_k: int = 8, filters: dict[str, Any] | None = None) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def chunks_for_files(self, repo_id: str, file_paths: set[str], limit_per_file: int = 2, query: str | None = None) -> list[Chunk]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, repo_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, repo_id: str) -> bool:
        raise NotImplementedError


class SourceParser(ABC):
    @abstractmethod
    def parse(self, record: FileRecord) -> dict[str, list[Any]]:
        raise NotImplementedError


class LLMProvider(ABC):
    name: str = "llm"

    @abstractmethod
    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False, max_tokens: int = 1400) -> Any:
        """Return an LLMResponse (text, provider, model, usage)."""
        raise NotImplementedError

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text
