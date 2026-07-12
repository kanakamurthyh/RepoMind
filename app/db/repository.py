from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models import AnalysisResult, Chunk, Weakness

logger = logging.getLogger(__name__)


class MetadataRepository:
    """Persist analysis metadata in SQLAlchemy DB, with JSON fallback."""

    def __init__(self, database_url: str, metadata_dir: Path, enabled: bool = True) -> None:
        self.database_url = database_url
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self._sqlalchemy_ready = False
        if enabled:
            try:
                self._init_sqlalchemy()
                self._sqlalchemy_ready = True
            except Exception as exc:
                logger.warning("Metadata DB unavailable; falling back to JSON metadata: %s", exc)

    def exists(self, repo_id: str) -> bool:
        if self._sqlalchemy_ready:
            with self.Session() as session:  # type: ignore[attr-defined]
                return session.get(self.AnalysisRun, repo_id) is not None  # type: ignore[attr-defined]
        return self._path(repo_id).exists()

    def save(self, result: AnalysisResult) -> None:
        if self._sqlalchemy_ready:
            self._save_db(result)
            return
        self._path(result.repo_id).write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    def load(self, repo_id: str, statuses: list[str] | None = None) -> AnalysisResult:
        if self._sqlalchemy_ready:
            loaded = self._load_db(repo_id, statuses)
            if loaded:
                return loaded
        path = self._path(repo_id)
        if not path.exists():
            raise FileNotFoundError(f"No analysis metadata found for {repo_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AnalysisResult(
            repo_id=data["repo_id"],
            repo_url=data["repo_url"],
            summary=data["summary"],
            weaknesses=[self._weakness_from_dict(item) for item in data["weaknesses"]],
            graph=data["graph"],
            chunks=[Chunk.from_dict(item) for item in data.get("chunks", [])],
            statuses=statuses or data.get("statuses", []),
        )

    def delete(self, repo_id: str) -> None:
        if self._sqlalchemy_ready:
            with self.Session() as session:  # type: ignore[attr-defined]
                for row_cls in (self.ChatSessionRow, self.GraphEdgeRow, self.SymbolRow, self.FileRow):  # type: ignore[attr-defined]
                    session.query(row_cls).filter_by(repo_id=repo_id).delete()
                run = session.get(self.AnalysisRun, repo_id)  # type: ignore[attr-defined]
                if run is not None:
                    session.delete(run)
                repo = session.get(self.RepositoryRow, repo_id)  # type: ignore[attr-defined]
                if repo is not None:
                    session.delete(repo)
                session.commit()
        path = self._path(repo_id)
        if path.exists():
            path.unlink()

    def record_chat(self, repo_id: str, question: str, answer: str, citations: list[dict[str, Any]]) -> None:
        payload = {"question": question, "answer": answer, "citations": citations}
        if self._sqlalchemy_ready:
            with self.Session() as session:  # type: ignore[attr-defined]
                session.add(
                    self.ChatSessionRow(  # type: ignore[attr-defined]
                        id=uuid4().hex,
                        repo_id=repo_id,
                        history=payload,
                    )
                )
                session.commit()
            return
        path = self.metadata_dir / f"{repo_id}.chats.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(payload)
        path.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    def _init_sqlalchemy(self) -> None:
        from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, create_engine, func, inspect, text
        from sqlalchemy.orm import declarative_base, relationship, sessionmaker

        Base = declarative_base()

        class RepositoryRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "repositories"
            repo_id = Column(String, primary_key=True)
            repo_url = Column(String, nullable=False)
            created_at = Column(DateTime(timezone=True), server_default=func.now())
            updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

        class AnalysisRun(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "analysis_runs"
            repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), primary_key=True)
            repo_url = Column(String, nullable=False)
            status = Column(String, default="completed")
            summary = Column(JSON, nullable=False)
            graph = Column(JSON, nullable=False)
            statuses = Column(JSON, nullable=False)
            chunks = relationship("ChunkRow", cascade="all, delete-orphan")
            weaknesses = relationship("WeaknessRow", cascade="all, delete-orphan")

        class FileRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "files"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
            path = Column(String, index=True)
            language = Column(String)
            role = Column(String)
            metadata_json = Column(JSON, default={})

        class ChunkRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "chunks"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("analysis_runs.repo_id", ondelete="CASCADE"), index=True)
            file_path = Column(String, index=True)
            language = Column(String)
            role = Column(String)
            start_line = Column(Integer)
            end_line = Column(Integer)
            text = Column(Text)
            symbol = Column(String, nullable=True)
            metadata_json = Column(JSON, default={})

        class WeaknessRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "weaknesses"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("analysis_runs.repo_id", ondelete="CASCADE"), index=True)
            severity = Column(String)
            category = Column(String)
            confidence = Column(String, default="medium")
            title = Column(String)
            file_path = Column(String, nullable=True)
            line_number = Column(Integer, nullable=True)
            evidence = Column(Text, nullable=True)
            why_it_matters = Column(Text)
            suggested_fix = Column(Text)

        class SymbolRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "symbols"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
            kind = Column(String)
            name = Column(String)
            file_path = Column(String, index=True, nullable=True)
            start_line = Column(Integer, nullable=True)
            end_line = Column(Integer, nullable=True)
            metadata_json = Column(JSON, default={})

        class GraphEdgeRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "graph_edges"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
            source = Column(String, index=True)
            target = Column(String, index=True)
            relation = Column(String, index=True)
            metadata_json = Column(JSON, default={})

        class ChatSessionRow(Base):  # type: ignore[valid-type,misc]
            __tablename__ = "chat_sessions"
            id = Column(String, primary_key=True)
            repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
            history = Column(JSON, nullable=False)
            created_at = Column(DateTime(timezone=True), server_default=func.now())

        self.Base = Base
        self.RepositoryRow = RepositoryRow
        self.AnalysisRun = AnalysisRun
        self.FileRow = FileRow
        self.ChunkRow = ChunkRow
        self.WeaknessRow = WeaknessRow
        self.SymbolRow = SymbolRow
        self.GraphEdgeRow = GraphEdgeRow
        self.ChatSessionRow = ChatSessionRow
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, future=True, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        inspector = inspect(self.engine)
        weakness_columns = {column["name"] for column in inspector.get_columns("weaknesses")}
        if "confidence" not in weakness_columns:
            with self.engine.begin() as connection:
                connection.execute(text("ALTER TABLE weaknesses ADD COLUMN confidence VARCHAR DEFAULT 'medium'"))
        self.Session = sessionmaker(self.engine, future=True)

    def _save_db(self, result: AnalysisResult) -> None:
        with self.Session() as session:  # type: ignore[attr-defined]
            repo = session.get(self.RepositoryRow, result.repo_id)  # type: ignore[attr-defined]
            if repo is None:
                session.add(self.RepositoryRow(repo_id=result.repo_id, repo_url=result.repo_url))  # type: ignore[attr-defined]
            else:
                repo.repo_url = result.repo_url
            existing = session.get(self.AnalysisRun, result.repo_id)  # type: ignore[attr-defined]
            if existing is not None:
                session.delete(existing)
                session.flush()
            for row_cls in (self.GraphEdgeRow, self.SymbolRow, self.FileRow):  # type: ignore[attr-defined]
                session.query(row_cls).filter_by(repo_id=result.repo_id).delete()
            run = self.AnalysisRun(  # type: ignore[attr-defined]
                repo_id=result.repo_id,
                repo_url=result.repo_url,
                status="completed",
                summary=result.summary,
                graph=result.graph,
                statuses=result.statuses,
            )
            run.chunks = [
                self.ChunkRow(  # type: ignore[attr-defined]
                    id=chunk.id,
                    repo_id=result.repo_id,
                    file_path=chunk.file_path,
                    language=chunk.language,
                    role=chunk.role,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    text=chunk.text,
                    symbol=chunk.symbol,
                    metadata_json=chunk.metadata,
                )
                for chunk in result.chunks
            ]
            run.weaknesses = [
                self.WeaknessRow(  # type: ignore[attr-defined]
                    id=weakness.id,
                    repo_id=result.repo_id,
                    severity=weakness.severity,
                    category=weakness.category,
                    confidence=weakness.confidence,
                    title=weakness.title,
                    file_path=weakness.file_path,
                    line_number=weakness.line_number,
                    evidence=weakness.evidence,
                    why_it_matters=weakness.why_it_matters,
                    suggested_fix=weakness.suggested_fix,
                )
                for weakness in result.weaknesses
            ]
            session.add(run)
            session.add_all(self._file_rows(result))
            session.add_all(self._symbol_rows(result))
            session.add_all(self._edge_rows(result))
            session.commit()

    def _file_rows(self, result: AnalysisResult) -> list[Any]:
        files: dict[str, dict[str, Any]] = {}
        for chunk in result.chunks:
            files.setdefault(
                chunk.file_path,
                {
                    "language": chunk.language,
                    "role": chunk.role,
                    "metadata": {"source": "chunks"},
                },
            )
        for node in result.graph.get("nodes", []):
            if node.get("kind") == "file":
                path = node.get("label") or node.get("file_path")
                if path:
                    files.setdefault(
                        str(path),
                        {
                            "language": node.get("language", ""),
                            "role": node.get("role", ""),
                            "metadata": {"source": "graph"},
                        },
                    )
        return [
            self.FileRow(  # type: ignore[attr-defined]
                id=self._stable_id(result.repo_id, "file", path),
                repo_id=result.repo_id,
                path=path,
                language=data["language"],
                role=data["role"],
                metadata_json=data["metadata"],
            )
            for path, data in files.items()
        ]

    def _symbol_rows(self, result: AnalysisResult) -> list[Any]:
        rows = []
        for node in result.graph.get("nodes", []):
            if node.get("kind") not in {"function", "class", "route", "entrypoint"}:
                continue
            rows.append(
                self.SymbolRow(  # type: ignore[attr-defined]
                    id=self._stable_id(result.repo_id, "symbol", node.get("id", ""), node.get("label", "")),
                    repo_id=result.repo_id,
                    kind=node.get("kind", ""),
                    name=node.get("label", ""),
                    file_path=node.get("file_path"),
                    start_line=node.get("start_line"),
                    end_line=node.get("end_line"),
                    metadata_json={
                        key: value
                        for key, value in node.items()
                        if key not in {"id", "kind", "label", "file_path", "start_line", "end_line"}
                    },
                )
            )
        return rows

    def _edge_rows(self, result: AnalysisResult) -> list[Any]:
        rows = []
        for edge in result.graph.get("edges", []):
            rows.append(
                self.GraphEdgeRow(  # type: ignore[attr-defined]
                    id=self._stable_id(result.repo_id, "edge", edge.get("source", ""), edge.get("target", ""), edge.get("relation", "")),
                    repo_id=result.repo_id,
                    source=edge.get("source", ""),
                    target=edge.get("target", ""),
                    relation=edge.get("relation", ""),
                    metadata_json={key: value for key, value in edge.items() if key not in {"source", "target", "relation"}},
                )
            )
        return rows

    def _load_db(self, repo_id: str, statuses: list[str] | None) -> AnalysisResult | None:
        with self.Session() as session:  # type: ignore[attr-defined]
            run = session.get(self.AnalysisRun, repo_id)  # type: ignore[attr-defined]
            if run is None:
                return None
            return AnalysisResult(
                repo_id=run.repo_id,
                repo_url=run.repo_url,
                summary=run.summary,
                weaknesses=[
                    Weakness(
                        id=row.id,
                        severity=row.severity,
                        category=row.category,
                        confidence=row.confidence or "medium",
                        title=row.title,
                        file_path=row.file_path,
                        line_number=row.line_number,
                        evidence=row.evidence,
                        why_it_matters=row.why_it_matters,
                        suggested_fix=row.suggested_fix,
                    )
                    for row in run.weaknesses
                ],
                graph=run.graph,
                chunks=[
                    Chunk(
                        id=row.id,
                        repo_id=run.repo_id,
                        file_path=row.file_path,
                        language=row.language,
                        role=row.role,
                        start_line=row.start_line,
                        end_line=row.end_line,
                        text=row.text,
                        symbol=row.symbol,
                        metadata=row.metadata_json or {},
                    )
                    for row in run.chunks
                ],
                statuses=statuses or run.statuses,
            )

    def _path(self, repo_id: str) -> Path:
        return self.metadata_dir / f"{repo_id}.json"

    def _weakness_from_dict(self, data: dict[str, Any]) -> Weakness:
        if "title" not in data and "issue" in data:
            data = {**data, "title": data["issue"]}
        allowed = {"id", "severity", "category", "confidence", "title", "file_path", "line_number", "evidence", "why_it_matters", "suggested_fix"}
        return Weakness(**{key: value for key, value in data.items() if key in allowed})

    def _stable_id(self, *parts: Any) -> str:
        return hashlib.sha1(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]
