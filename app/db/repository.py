from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    LATE_COLUMNS,
    AnalysisJobRow,
    AnalysisRun,
    Base,
    ChatSessionRow,
    ChunkRow,
    FileRow,
    GraphEdgeRow,
    RepositoryRow,
    SymbolRow,
    WeaknessRow,
)
from app.models import AnalysisResult, Chunk, Weakness
from app.utils.redaction import redact_data

logger = logging.getLogger(__name__)
DB_URL_CREDENTIALS_RE = re.compile(r"(\w[\w+.-]*://[^\s:/@]+):[^\s@]+@")
WEAKNESS_FIELDS = {"id", "severity", "category", "confidence", "title", "file_path", "line_number", "evidence", "why_it_matters", "suggested_fix"}


def _now() -> str:
    return datetime.now(UTC).isoformat()


class MetadataStoreUnavailableError(RuntimeError):
    """The configured metadata DB could not be initialised and JSON fallback is not allowed."""


def describe_db_error(exc: BaseException, limit: int = 300) -> str:
    """Short, credential-free description of a DB error for logs and /ready."""
    first_line = (str(exc).strip().splitlines() or [""])[0]
    first_line = DB_URL_CREDENTIALS_RE.sub(r"\1:***@", first_line)
    return f"{type(exc).__name__}: {first_line}"[:limit] if first_line else type(exc).__name__


class MetadataRepository:
    """Persist analysis metadata in a SQLAlchemy DB, with an optional JSON-file fallback.

    If the DB cannot be initialised and ``allow_json_fallback`` is true, the repository
    degrades to JSON files and says so loudly (ERROR log, ``backend == "json"``,
    ``fallback_reason``). If the fallback is not allowed it raises
    :class:`MetadataStoreUnavailableError` instead.
    """

    def __init__(self, database_url: str, metadata_dir: Path, enabled: bool = True, allow_json_fallback: bool = True) -> None:
        self.database_url = database_url
        self.metadata_dir = metadata_dir
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled
        self.allow_json_fallback = allow_json_fallback
        self.fallback_reason: str | None = None if enabled else "disabled by USE_METADATA_DB=false"
        self._sqlalchemy_ready = False
        self._json_jobs: dict[str, dict[str, Any]] = {}
        self._json_lock = threading.Lock()
        self._jobs_path = self.metadata_dir / "queue.jobs.json"
        if self._jobs_path.exists():
            try:
                self._json_jobs = json.loads(self._jobs_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                logger.error("Could not load persisted JSON jobs: %s", exc)
                raise MetadataStoreUnavailableError("JSON job storage is unreadable") from exc
        # Kept as attributes for callers/tests that introspect rows.
        self.RepositoryRow, self.AnalysisRun, self.FileRow, self.ChunkRow = RepositoryRow, AnalysisRun, FileRow, ChunkRow
        self.WeaknessRow, self.SymbolRow, self.GraphEdgeRow, self.ChatSessionRow = WeaknessRow, SymbolRow, GraphEdgeRow, ChatSessionRow
        if enabled:
            try:
                self._init_sqlalchemy()
                self._sqlalchemy_ready = True
            except Exception as exc:
                reason = describe_db_error(exc)
                if not allow_json_fallback:
                    logger.error("Metadata DB unavailable (%s) and METADATA_DB_FALLBACK does not allow JSON files; refusing to start.", reason)
                    raise MetadataStoreUnavailableError(f"Metadata database is unavailable ({type(exc).__name__}).") from exc
                self.fallback_reason = reason
                logger.error(
                    "Metadata DB unavailable (%s); FALLING BACK to JSON metadata files in %s. Analyses saved now will not be in the "
                    "configured database. Set METADATA_DB_FALLBACK=false to fail fast instead.",
                    reason, self.metadata_dir,
                )

    @property
    def backend(self) -> str:
        return "sqlalchemy" if self._sqlalchemy_ready else "json"

    def storage_info(self) -> dict[str, Any]:
        """JSON-friendly description of the active storage backend (for /ready)."""
        return {
            "backend": self.backend,
            "configured_backend": "sqlalchemy" if self.enabled else "json",
            "degraded": self.enabled and not self._sqlalchemy_ready,
            "fallback_reason": self.fallback_reason,
            "json_fallback_allowed": self.allow_json_fallback,
        }

    def _init_sqlalchemy(self) -> None:
        if self.database_url.startswith("sqlite:///"):
            Path(self.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        connect_args = {"check_same_thread": False} if self.database_url.startswith("sqlite") else {}
        self.engine = create_engine(self.database_url, future=True, connect_args=connect_args, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)
        self._ensure_late_columns()
        self.Session = sessionmaker(self.engine, future=True, expire_on_commit=False)
        self._finalizer = weakref.finalize(self, self.engine.dispose)

    def close(self) -> None:
        if hasattr(self, "engine"):
            self.engine.dispose()

    def _ensure_late_columns(self) -> None:
        inspector = inspect(self.engine)
        for table, columns in LATE_COLUMNS.items():
            existing = {column["name"] for column in inspector.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    # Schema is owned by Alembic; this only patches DBs created before a column existed.
                    logger.warning("Adding missing column %s.%s at runtime; run `alembic upgrade head` to manage the schema.", table, name)
                    with self.engine.begin() as connection:
                        connection.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))

    # --- analyses -------------------------------------------------------------------------
    def exists(self, repo_id: str) -> bool:
        if self._sqlalchemy_ready:
            with self.Session() as session:
                return session.get(AnalysisRun, repo_id) is not None
        return self._path(repo_id).exists()

    def save(self, result: AnalysisResult) -> None:
        if self._sqlalchemy_ready:
            self._save_db(result)
            return
        self._path(result.repo_id).write_text(json.dumps(result.to_dict(), indent=1), encoding="utf-8")

    def load(self, repo_id: str, statuses: list[str] | None = None, include_chunks: bool = False) -> AnalysisResult:
        if self._sqlalchemy_ready:
            loaded = self._load_db(repo_id, statuses, include_chunks)
            if loaded:
                return loaded
        path = self._path(repo_id)
        if not path.exists():
            raise FileNotFoundError(f"No analysis found for {repo_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return AnalysisResult(
            repo_id=data["repo_id"],
            repo_url=data["repo_url"],
            summary=data["summary"],
            weaknesses=[self._weakness_from_dict(item) for item in data["weaknesses"]],
            graph=data["graph"],
            chunks=[Chunk.from_dict(item) for item in data.get("chunks", [])] if include_chunks else [],
            statuses=statuses or data.get("statuses", []),
            commit_sha=data.get("commit_sha"),
        )

    def list_repos(self) -> list[dict[str, Any]]:
        if self._sqlalchemy_ready:
            with self.Session() as session:
                rows = session.query(AnalysisRun.repo_id, AnalysisRun.repo_url, AnalysisRun.commit_sha, AnalysisRun.created_at).all()
                return [{"repo_id": r[0], "repo_url": r[1], "commit_sha": r[2], "analyzed_at": r[3].isoformat() if r[3] else None} for r in rows]
        repos = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            if path.name.endswith(".chats.json") or path.name.endswith(".jobs.json"):
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                repos.append({"repo_id": data["repo_id"], "repo_url": data["repo_url"], "commit_sha": data.get("commit_sha"), "analyzed_at": None})
            except (OSError, json.JSONDecodeError, KeyError) as exc:
                logger.warning("Skipping unreadable metadata file %s: %s", path, exc)
        return repos

    def delete(self, repo_id: str) -> None:
        if self._sqlalchemy_ready:
            with self.Session() as session:
                for row_cls in (ChatSessionRow, GraphEdgeRow, SymbolRow, FileRow, ChunkRow, WeaknessRow):
                    session.query(row_cls).filter_by(repo_id=repo_id).delete()
                session.query(AnalysisRun).filter_by(repo_id=repo_id).delete()
                session.query(RepositoryRow).filter_by(repo_id=repo_id).delete()
                session.commit()
        for path in (self._path(repo_id), self.metadata_dir / f"{repo_id}.chats.json"):
            if path.exists():
                path.unlink()

    def record_chat(self, repo_id: str, question: str, answer: str, citations: list[dict[str, Any]], meta: dict[str, Any] | None = None) -> None:
        payload = redact_data({"question": question, "answer": answer, "citations": citations, "meta": meta or {}, "at": _now()})
        if self._sqlalchemy_ready:
            with self.Session() as session:
                if session.get(RepositoryRow, repo_id) is None:
                    return
                session.add(ChatSessionRow(id=uuid4().hex, repo_id=repo_id, history=payload))
                session.commit()
            return
        path = self.metadata_dir / f"{repo_id}.chats.json"
        with self._json_lock:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            existing.append(payload)
            path.write_text(json.dumps(existing[-500:], indent=1), encoding="utf-8")

    def _save_db(self, result: AnalysisResult) -> None:
        with self.Session() as session:
            repo = session.get(RepositoryRow, result.repo_id)
            if repo is None:
                session.add(RepositoryRow(repo_id=result.repo_id, repo_url=result.repo_url))
            else:
                repo.repo_url = result.repo_url
            for row_cls in (ChunkRow, WeaknessRow, GraphEdgeRow, SymbolRow, FileRow):
                session.query(row_cls).filter_by(repo_id=result.repo_id).delete()
            session.query(AnalysisRun).filter_by(repo_id=result.repo_id).delete()
            session.flush()
            session.add(
                AnalysisRun(
                    repo_id=result.repo_id,
                    repo_url=result.repo_url,
                    status="completed",
                    commit_sha=result.commit_sha,
                    summary=result.summary,
                    graph=result.graph,
                    statuses=result.statuses,
                )
            )
            session.flush()
            session.add_all(
                ChunkRow(
                    id=chunk.id, repo_id=result.repo_id, file_path=chunk.file_path, language=chunk.language, role=chunk.role,
                    start_line=chunk.start_line, end_line=chunk.end_line, text=chunk.text, symbol=chunk.symbol, metadata_json=chunk.metadata,
                )
                for chunk in result.chunks
            )
            seen_weaknesses: set[str] = set()
            for weakness in result.weaknesses:
                local_id = weakness.id or self._stable_id("weakness", weakness.title, weakness.file_path, weakness.line_number)
                weakness_id = f"{result.repo_id}:{local_id}"
                if weakness_id in seen_weaknesses:
                    continue
                seen_weaknesses.add(weakness_id)
                session.add(
                    WeaknessRow(
                        id=weakness_id, repo_id=result.repo_id, severity=weakness.severity, category=weakness.category,
                        confidence=weakness.confidence, title=weakness.title, file_path=weakness.file_path, line_number=weakness.line_number,
                        evidence=weakness.evidence, why_it_matters=weakness.why_it_matters, suggested_fix=weakness.suggested_fix,
                    )
                )
            session.add_all(self._file_rows(result))
            session.add_all(self._symbol_rows(result))
            session.add_all(self._edge_rows(result))
            session.commit()

    def _file_rows(self, result: AnalysisResult) -> list[FileRow]:
        files: dict[str, dict[str, Any]] = {}
        for node in result.graph.get("nodes", []):
            if node.get("kind") == "file":
                path = node.get("label") or node.get("file_path")
                if path:
                    files.setdefault(str(path), {"language": node.get("language", ""), "role": node.get("role", ""), "metadata": {"lines": node.get("lines")}})
        for chunk in result.chunks:
            files.setdefault(chunk.file_path, {"language": chunk.language, "role": chunk.role, "metadata": {}})
        return [
            FileRow(id=self._stable_id(result.repo_id, "file", path), repo_id=result.repo_id, path=path, language=data["language"], role=data["role"], metadata_json=data["metadata"])
            for path, data in files.items()
        ]

    def _symbol_rows(self, result: AnalysisResult) -> list[SymbolRow]:
        rows: dict[str, SymbolRow] = {}
        for node in result.graph.get("nodes", []):
            if node.get("kind") not in {"function", "class", "route", "entrypoint"}:
                continue
            row_id = self._stable_id(result.repo_id, "symbol", node.get("id", ""))
            rows[row_id] = SymbolRow(
                id=row_id, repo_id=result.repo_id, kind=node.get("kind", ""), name=node.get("label", ""), file_path=node.get("file_path"),
                start_line=node.get("start_line"), end_line=node.get("end_line"),
                metadata_json={k: v for k, v in node.items() if k not in {"id", "kind", "label", "file_path", "start_line", "end_line"}},
            )
        return list(rows.values())

    def _edge_rows(self, result: AnalysisResult) -> list[GraphEdgeRow]:
        rows: dict[str, GraphEdgeRow] = {}
        for edge in result.graph.get("edges", []):
            row_id = self._stable_id(result.repo_id, "edge", edge.get("source", ""), edge.get("target", ""), edge.get("relation", ""))
            rows[row_id] = GraphEdgeRow(
                id=row_id, repo_id=result.repo_id, source=edge.get("source", ""), target=edge.get("target", ""), relation=edge.get("relation", ""),
                metadata_json={k: v for k, v in edge.items() if k not in {"source", "target", "relation"}},
            )
        return list(rows.values())

    def _load_db(self, repo_id: str, statuses: list[str] | None, include_chunks: bool) -> AnalysisResult | None:
        with self.Session() as session:
            run = session.get(AnalysisRun, repo_id)
            if run is None:
                return None
            chunks: list[Chunk] = []
            if include_chunks:
                chunks = [
                    Chunk(
                        id=row.id, repo_id=run.repo_id, file_path=row.file_path, language=row.language, role=row.role,
                        start_line=row.start_line, end_line=row.end_line, text=row.text, symbol=row.symbol, metadata=row.metadata_json or {},
                    )
                    for row in session.query(ChunkRow).filter_by(repo_id=repo_id).order_by(ChunkRow.file_path, ChunkRow.start_line)
                ]
            return AnalysisResult(
                repo_id=run.repo_id,
                repo_url=run.repo_url,
                summary=run.summary,
                weaknesses=[
                    Weakness(
                        id=row.id.split(":", 1)[1] if row.id.startswith(f"{run.repo_id}:") else row.id,
                        severity=row.severity, category=row.category, confidence=row.confidence or "medium", title=row.title,
                        file_path=row.file_path, line_number=row.line_number, evidence=row.evidence,
                        why_it_matters=row.why_it_matters, suggested_fix=row.suggested_fix,
                    )
                    for row in sorted(run.weaknesses, key=lambda r: ({"critical": -1, "high": 0, "medium": 1, "low": 2}.get(r.severity, 9), r.category or "", r.title or ""))
                ],
                graph=run.graph,
                chunks=chunks,
                statuses=statuses or run.statuses,
                commit_sha=run.commit_sha,
            )

    # --- jobs ----------------------------------------------------------------------------
    def create_job(self, repo_id: str, repo_url: str) -> dict[str, Any]:
        job = {"job_id": uuid4().hex, "repo_id": repo_id, "repo_url": repo_url, "state": "queued", "stage": "queued", "progress": 0, "statuses": ["queued"], "error": None, "error_code": None, "created_at": _now(), "updated_at": _now()}
        if self._sqlalchemy_ready:
            with self.Session() as session:
                session.add(AnalysisJobRow(**{k: v for k, v in job.items() if k not in {"created_at", "updated_at"}}))
                session.commit()
        else:
            with self._json_lock:
                self._json_jobs[job["job_id"]] = job
                self._save_json_jobs()
        return job

    def update_job(self, job_id: str, **updates: Any) -> None:
        allowed = {"state", "stage", "progress", "statuses", "error", "error_code"}
        clean = {k: v for k, v in updates.items() if k in allowed}
        if self._sqlalchemy_ready:
            with self.Session() as session:
                row = session.get(AnalysisJobRow, job_id)
                if row is None:
                    return
                for key, value in clean.items():
                    setattr(row, key, value)
                row.updated_at = datetime.now(UTC)
                session.commit()
        else:
            with self._json_lock:
                if job_id in self._json_jobs:
                    self._json_jobs[job_id].update(clean, updated_at=_now())
                    self._save_json_jobs()

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        if self._sqlalchemy_ready:
            with self.Session() as session:
                row = session.get(AnalysisJobRow, job_id)
                return self._job_dict(row) if row else None
        return dict(self._json_jobs[job_id]) if job_id in self._json_jobs else None

    def latest_job(self, repo_id: str) -> dict[str, Any] | None:
        if self._sqlalchemy_ready:
            with self.Session() as session:
                row = session.query(AnalysisJobRow).filter_by(repo_id=repo_id).order_by(AnalysisJobRow.created_at.desc(), AnalysisJobRow.job_id).first()
                return self._job_dict(row) if row else None
        jobs = [job for job in self._json_jobs.values() if job["repo_id"] == repo_id]
        return dict(sorted(jobs, key=lambda job: job["created_at"])[-1]) if jobs else None

    def mark_interrupted_jobs(self) -> int:
        """Jobs left running by a previous process can never finish; mark them failed on startup."""
        if not self._sqlalchemy_ready:
            with self._json_lock:
                pending = [job for job in self._json_jobs.values() if job["state"] in {"queued", "running"}]
                for job in pending:
                    job.update(state="failed", error="Server restarted before the job finished.", error_code="interrupted")
                self._save_json_jobs()
                return len(pending)
        with self.Session() as session:
            rows = session.query(AnalysisJobRow).filter(AnalysisJobRow.state.in_(["queued", "running"])).all()
            for row in rows:
                row.state, row.error, row.error_code = "failed", "Server restarted before the job finished.", "interrupted"
            session.commit()
            return len(rows)

    def _save_json_jobs(self) -> None:
        temporary = self._jobs_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(redact_data(self._json_jobs)), encoding="utf-8")
        temporary.replace(self._jobs_path)

    def _job_dict(self, row: AnalysisJobRow) -> dict[str, Any]:
        return {
            "job_id": row.job_id, "repo_id": row.repo_id, "repo_url": row.repo_url, "state": row.state, "stage": row.stage,
            "progress": row.progress or 0, "statuses": row.statuses or [], "error": row.error, "error_code": row.error_code,
            "created_at": row.created_at.isoformat() if row.created_at else None, "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        }

    # --- helpers --------------------------------------------------------------------------
    def _path(self, repo_id: str) -> Path:
        return self.metadata_dir / f"{repo_id}.json"

    def _weakness_from_dict(self, data: dict[str, Any]) -> Weakness:
        if "title" not in data and "issue" in data:
            data = {**data, "title": data["issue"]}
        return Weakness(**{key: value for key, value in data.items() if key in WEAKNESS_FIELDS})

    def _stable_id(self, *parts: Any) -> str:
        return hashlib.sha1(":".join(str(part) for part in parts).encode("utf-8")).hexdigest()[:24]
