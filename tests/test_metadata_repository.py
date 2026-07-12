from __future__ import annotations

from app.db.repository import MetadataRepository
from app.models import AnalysisResult, Chunk, Weakness


def test_metadata_repository_round_trips_analysis_with_confidence(tmp_path):
    repo = MetadataRepository(f"sqlite:///{tmp_path / 'repomind.db'}", tmp_path / "metadata", enabled=True)
    result = AnalysisResult(
        repo_id="demo",
        repo_url="https://github.com/example/demo",
        summary={"project_identity": {"project_name": "demo"}},
        weaknesses=[
            Weakness(
                id="w1",
                severity="high",
                confidence="high",
                category="security",
                title="High-confidence hardcoded secret",
                file_path="settings.py",
                line_number=1,
                evidence="API_KEY='<redacted>'",
                why_it_matters="Secrets can leak.",
                suggested_fix="Move the value to an environment variable.",
            )
        ],
        graph={
            "nodes": [
                {"id": "file:README.md", "kind": "file", "label": "README.md", "language": "markdown", "role": "docs"},
                {"id": "function:app.py:main", "kind": "function", "label": "main", "file_path": "app.py", "start_line": 1, "end_line": 2},
            ],
            "edges": [{"source": "file:app.py", "target": "function:app.py:main", "relation": "defines_function"}],
        },
        chunks=[Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "hello")],
        statuses=["completed"],
    )

    repo.save(result)
    loaded = repo.load("demo")

    assert loaded.repo_id == "demo"
    assert loaded.weaknesses[0].confidence == "high"
    assert loaded.chunks[0].file_path == "README.md"
    repo.record_chat("demo", "what is this?", "A demo.", [{"file_path": "README.md", "start_line": 1, "end_line": 3}])
    repo.record_chat("demo", "what is this?", "A demo.", [{"file_path": "README.md", "start_line": 1, "end_line": 3}])

    if repo._sqlalchemy_ready:
        with repo.Session() as session:  # type: ignore[attr-defined]
            assert session.query(repo.RepositoryRow).count() == 1  # type: ignore[attr-defined]
            assert session.query(repo.FileRow).count() >= 1  # type: ignore[attr-defined]
            assert session.query(repo.SymbolRow).count() == 1  # type: ignore[attr-defined]
            assert session.query(repo.GraphEdgeRow).count() == 1  # type: ignore[attr-defined]
            assert session.query(repo.ChatSessionRow).count() == 2  # type: ignore[attr-defined]
