"""initial metadata schema

Revision ID: 0001
Revises:
Create Date: 2026-07-10
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "repositories",
        sa.Column("repo_id", sa.String(), primary_key=True),
        sa.Column("repo_url", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_table(
        "analysis_runs",
        sa.Column("repo_id", sa.String(), sa.ForeignKey("repositories.repo_id", ondelete="CASCADE"), primary_key=True),
        sa.Column("repo_url", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("summary", sa.JSON(), nullable=False),
        sa.Column("graph", sa.JSON(), nullable=False),
        sa.Column("statuses", sa.JSON(), nullable=False),
        if_not_exists=True,
    )
    op.create_table(
        "files",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("repositories.repo_id", ondelete="CASCADE")),
        sa.Column("path", sa.String()),
        sa.Column("language", sa.String()),
        sa.Column("role", sa.String()),
        sa.Column("metadata_json", sa.JSON()),
        if_not_exists=True,
    )
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("analysis_runs.repo_id", ondelete="CASCADE")),
        sa.Column("file_path", sa.String()),
        sa.Column("language", sa.String()),
        sa.Column("role", sa.String()),
        sa.Column("start_line", sa.Integer()),
        sa.Column("end_line", sa.Integer()),
        sa.Column("text", sa.Text()),
        sa.Column("symbol", sa.String(), nullable=True),
        sa.Column("metadata_json", sa.JSON()),
        if_not_exists=True,
    )
    op.create_table(
        "weaknesses",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("analysis_runs.repo_id", ondelete="CASCADE")),
        sa.Column("severity", sa.String()),
        sa.Column("category", sa.String()),
        sa.Column("confidence", sa.String(), server_default="medium"),
        sa.Column("title", sa.String()),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("line_number", sa.Integer(), nullable=True),
        sa.Column("evidence", sa.Text(), nullable=True),
        sa.Column("why_it_matters", sa.Text()),
        sa.Column("suggested_fix", sa.Text()),
        if_not_exists=True,
    )
    op.create_table(
        "symbols",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("repositories.repo_id", ondelete="CASCADE")),
        sa.Column("kind", sa.String()),
        sa.Column("name", sa.String()),
        sa.Column("file_path", sa.String(), nullable=True),
        sa.Column("start_line", sa.Integer(), nullable=True),
        sa.Column("end_line", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.JSON()),
        if_not_exists=True,
    )
    op.create_table(
        "graph_edges",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("repositories.repo_id", ondelete="CASCADE")),
        sa.Column("source", sa.String()),
        sa.Column("target", sa.String()),
        sa.Column("relation", sa.String()),
        sa.Column("metadata_json", sa.JSON()),
        if_not_exists=True,
    )
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), sa.ForeignKey("repositories.repo_id", ondelete="CASCADE")),
        sa.Column("history", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_files_repo_id", "files", ["repo_id"], if_not_exists=True)
    op.create_index("ix_files_path", "files", ["path"], if_not_exists=True)
    op.create_index("ix_chunks_repo_id", "chunks", ["repo_id"], if_not_exists=True)
    op.create_index("ix_chunks_file_path", "chunks", ["file_path"], if_not_exists=True)
    op.create_index("ix_weaknesses_repo_id", "weaknesses", ["repo_id"], if_not_exists=True)
    op.create_index("ix_symbols_repo_id", "symbols", ["repo_id"], if_not_exists=True)
    op.create_index("ix_symbols_file_path", "symbols", ["file_path"], if_not_exists=True)
    op.create_index("ix_graph_edges_repo_id", "graph_edges", ["repo_id"], if_not_exists=True)
    op.create_index("ix_graph_edges_source", "graph_edges", ["source"], if_not_exists=True)
    op.create_index("ix_graph_edges_target", "graph_edges", ["target"], if_not_exists=True)
    op.create_index("ix_graph_edges_relation", "graph_edges", ["relation"], if_not_exists=True)
    op.create_index("ix_chat_sessions_repo_id", "chat_sessions", ["repo_id"], if_not_exists=True)


def downgrade() -> None:
    op.drop_table("chat_sessions")
    op.drop_table("graph_edges")
    op.drop_table("symbols")
    op.drop_table("weaknesses")
    op.drop_table("chunks")
    op.drop_table("files")
    op.drop_table("analysis_runs")
    op.drop_table("repositories")
