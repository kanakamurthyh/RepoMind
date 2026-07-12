from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class FileRecord:
    repo_id: str
    path: str
    abs_path: str
    language: str
    role: str
    size_bytes: int
    line_count: int
    text: str
    truncated: bool = False
    warnings: list[str] = field(default_factory=list)
    kind: str = "text"
    mime_type: str = "text/plain"
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Chunk:
    id: str
    repo_id: str
    file_path: str
    language: str
    role: str
    start_line: int
    end_line: int
    text: str
    score: float | None = None
    symbol: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(**data)


@dataclass(slots=True)
class CodeEntity:
    kind: str
    name: str
    file_path: str
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Weakness:
    severity: str
    title: str
    why_it_matters: str
    suggested_fix: str
    category: str = "maintainability"
    confidence: str = "medium"
    file_path: str | None = None
    line_number: int | None = None
    evidence: str | None = None
    id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["issue"] = self.title
        return data

    @property
    def issue(self) -> str:
        return self.title


@dataclass(slots=True)
class AnalysisResult:
    repo_id: str
    repo_url: str
    summary: dict[str, Any]
    weaknesses: list[Weakness]
    graph: dict[str, Any]
    chunks: list[Chunk]
    statuses: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_id": self.repo_id,
            "repo_url": self.repo_url,
            "summary": self.summary,
            "weaknesses": [w.to_dict() for w in self.weaknesses],
            "graph": self.graph,
            "chunks": [c.to_dict() for c in self.chunks],
            "statuses": self.statuses,
        }
