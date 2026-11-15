from __future__ import annotations

from app.analysis.weaknesses import WeaknessDetector
from app.db.repository import MetadataRepository
from app.models import AnalysisResult, Chunk, FileRecord, Weakness


def _result(repo_id: str, weaknesses: list[Weakness]) -> AnalysisResult:
    return AnalysisResult(
        repo_id=repo_id,
        repo_url=f"https://github.com/example/{repo_id}",
        summary={"project_identity": {"project_name": repo_id}},
        weaknesses=weaknesses,
        graph={
            "nodes": [
                {"id": "file:README.md", "kind": "file", "label": "README.md", "language": "markdown", "role": "docs"},
                {"id": "function:app.py:main", "kind": "function", "label": "main", "file_path": "app.py", "start_line": 1, "end_line": 2},
            ],
            "edges": [{"source": "file:app.py", "target": "function:app.py:main", "relation": "defines_function"}],
        },
        chunks=[Chunk(f"{repo_id}:c1", repo_id, "README.md", "markdown", "docs", 1, 3, "hello")],
        statuses=["completed"],
        commit_sha="abc123",
    )


def test_round_trip_with_lazy_chunks_and_chat_history(tmp_path):
    repo = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata")
    weakness = Weakness(id="w1", severity="high", confidence="high", category="security", title="Secret", file_path="settings.py", line_number=1, why_it_matters="x", suggested_fix="y")
    repo.save(_result("demo-repo-0000000000", [weakness]))

    loaded = repo.load("demo-repo-0000000000")
    assert loaded.weaknesses[0].id == "w1" and loaded.weaknesses[0].confidence == "high"
    assert loaded.chunks == [] and loaded.commit_sha == "abc123"
    assert repo.load("demo-repo-0000000000", include_chunks=True).chunks[0].file_path == "README.md"

    repo.record_chat("demo-repo-0000000000", "what?", "A demo.", [])
    with repo.Session() as session:
        assert session.query(repo.ChatSessionRow).count() == 1
        assert session.query(repo.SymbolRow).count() == 1


def test_two_repos_with_identical_findings_both_save(tmp_path):
    """Regression: repo-wide findings (e.g. 'No license file') used to collide on the primary key."""
    repo = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata")
    records = [FileRecord("x", "app.py", "/tmp/app.py", "python", "code", 8, 1, "print(1)")]
    for repo_id in ("owner-a-0000000000", "owner-b-1111111111"):
        repo.save(_result(repo_id, WeaknessDetector().detect(records, repo_id=repo_id)))

    assert {row["repo_id"] for row in repo.list_repos()} == {"owner-a-0000000000", "owner-b-1111111111"}
    assert len(repo.load("owner-b-1111111111").weaknesses) == len(repo.load("owner-a-0000000000").weaknesses) > 3


def test_same_legacy_weakness_ids_across_repos_do_not_collide(tmp_path):
    repo = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata")
    shared = Weakness(id="same-id", severity="low", title="No license file detected", why_it_matters="x", suggested_fix="y")
    repo.save(_result("owner-a-0000000000", [shared]))
    repo.save(_result("owner-b-1111111111", [shared]))
    assert repo.load("owner-b-1111111111").weaknesses[0].id == "same-id"


def test_jobs_persist_and_interrupted_jobs_are_failed(tmp_path):
    repo = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata")
    job = repo.create_job("owner-a-0000000000", "https://github.com/owner/a")
    repo.update_job(job["job_id"], state="running", stage="indexing", progress=58)

    assert repo.get_job(job["job_id"])["stage"] == "indexing"
    assert repo.mark_interrupted_jobs() == 1
    assert repo.latest_job("owner-a-0000000000")["error_code"] == "interrupted"


def test_json_fallback_when_db_disabled(tmp_path):
    repo = MetadataRepository("sqlite:///unused", tmp_path / "metadata", enabled=False)
    repo.save(_result("owner-a-0000000000", []))
    assert repo.backend == "json" and repo.exists("owner-a-0000000000")
    assert repo.list_repos()[0]["repo_id"] == "owner-a-0000000000"


def test_db_failure_falls_back_loudly_when_allowed(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.ERROR, logger="app.db.repository"):
        repo = MetadataRepository("notadialect://user:hunter2-password@db/x", tmp_path / "metadata", allow_json_fallback=True)

    assert repo.backend == "json"
    info = repo.storage_info()
    assert info["degraded"] is True and info["configured_backend"] == "sqlalchemy" and info["fallback_reason"]
    assert any("FALLING BACK to JSON" in record.getMessage() and record.levelno == logging.ERROR for record in caplog.records)
    assert "hunter2-password" not in caplog.text and "hunter2-password" not in info["fallback_reason"]
    repo.save(_result("owner-a-0000000000", []))
    assert repo.exists("owner-a-0000000000")


def test_db_failure_fails_fast_when_fallback_not_allowed(tmp_path):
    import pytest

    from app.db.repository import MetadataStoreUnavailableError

    with pytest.raises(MetadataStoreUnavailableError):
        MetadataRepository("notadialect://db/x", tmp_path / "metadata", allow_json_fallback=False)


def test_storage_info_for_healthy_and_disabled_db(tmp_path):
    healthy = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata")
    assert healthy.storage_info()["degraded"] is False and healthy.storage_info()["fallback_reason"] is None
    disabled = MetadataRepository("sqlite:///unused", tmp_path / "metadata", enabled=False)
    assert disabled.storage_info() == {
        "backend": "json", "configured_backend": "json", "degraded": False,
        "fallback_reason": "disabled by USE_METADATA_DB=false", "json_fallback_allowed": True,
    }
