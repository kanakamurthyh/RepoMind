"""Single source of truth for the relational schema (used by the app and by Alembic)."""

from __future__ import annotations

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class RepositoryRow(Base):
    __tablename__ = "repositories"
    repo_id = Column(String, primary_key=True)
    repo_url = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), primary_key=True)
    repo_url = Column(String, nullable=False)
    status = Column(String, default="completed")
    commit_sha = Column(String, nullable=True)
    summary = Column(JSON, nullable=False)
    graph = Column(JSON, nullable=False)
    statuses = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=True)
    chunks = relationship("ChunkRow", cascade="all, delete-orphan", lazy="noload")
    weaknesses = relationship("WeaknessRow", cascade="all, delete-orphan", lazy="selectin")


class FileRow(Base):
    __tablename__ = "files"
    id = Column(String, primary_key=True)
    repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
    path = Column(String, index=True)
    language = Column(String)
    role = Column(String)
    metadata_json = Column(JSON, default=dict)


class ChunkRow(Base):
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
    metadata_json = Column(JSON, default=dict)


class WeaknessRow(Base):
    __tablename__ = "weaknesses"
    # Stored as "<repo_id>:<weakness id>" so identical findings in different repos never collide.
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


class SymbolRow(Base):
    __tablename__ = "symbols"
    id = Column(String, primary_key=True)
    repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
    kind = Column(String)
    name = Column(String)
    file_path = Column(String, index=True, nullable=True)
    start_line = Column(Integer, nullable=True)
    end_line = Column(Integer, nullable=True)
    metadata_json = Column(JSON, default=dict)


class GraphEdgeRow(Base):
    __tablename__ = "graph_edges"
    id = Column(String, primary_key=True)
    repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
    source = Column(String, index=True)
    target = Column(String, index=True)
    relation = Column(String, index=True)
    metadata_json = Column(JSON, default=dict)


class ChatSessionRow(Base):
    __tablename__ = "chat_sessions"
    id = Column(String, primary_key=True)
    repo_id = Column(String, ForeignKey("repositories.repo_id", ondelete="CASCADE"), index=True)
    history = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AnalysisJobRow(Base):
    __tablename__ = "analysis_jobs"
    job_id = Column(String, primary_key=True)
    repo_id = Column(String, index=True, nullable=False)
    repo_url = Column(String, nullable=False)
    state = Column(String, nullable=False, default="queued")
    stage = Column(String, nullable=True)
    progress = Column(Integer, default=0)
    statuses = Column(JSON, default=list)
    error = Column(Text, nullable=True)
    error_code = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# Columns added after the initial schema; ensured on SQLite dev databases created by older versions.
LATE_COLUMNS = {
    "weaknesses": {"confidence": "VARCHAR DEFAULT 'medium'"},
    "analysis_runs": {"commit_sha": "VARCHAR", "created_at": "TIMESTAMP"},
}
