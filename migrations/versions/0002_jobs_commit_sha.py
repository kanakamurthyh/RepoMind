"""analysis jobs table, commit sha and timestamps on analysis runs

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-29
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {column["name"] for column in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    existing = _columns("analysis_runs")
    if "commit_sha" not in existing:
        op.add_column("analysis_runs", sa.Column("commit_sha", sa.String(), nullable=True))
    if "created_at" not in existing:
        op.add_column("analysis_runs", sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True))
    op.create_table(
        "analysis_jobs",
        sa.Column("job_id", sa.String(), primary_key=True),
        sa.Column("repo_id", sa.String(), nullable=False),
        sa.Column("repo_url", sa.String(), nullable=False),
        sa.Column("state", sa.String(), nullable=False, server_default="queued"),
        sa.Column("stage", sa.String(), nullable=True),
        sa.Column("progress", sa.Integer(), server_default="0"),
        sa.Column("statuses", sa.JSON()),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        if_not_exists=True,
    )
    op.create_index("ix_analysis_jobs_repo_id", "analysis_jobs", ["repo_id"], if_not_exists=True)


def downgrade() -> None:
    op.drop_index("ix_analysis_jobs_repo_id", table_name="analysis_jobs")
    op.drop_table("analysis_jobs")
    op.drop_column("analysis_runs", "created_at")
    op.drop_column("analysis_runs", "commit_sha")
