from __future__ import annotations

from app.config import Settings
from app.core.repo_analyzer import RepoAnalyzer


def test_analyzer_cache_requires_metadata_and_vector_index(tmp_path):
    settings = Settings(data_dir=tmp_path)
    analyzer = RepoAnalyzer(settings)
    repo_id = "owner-repo-abc123"
    analyzer._metadata_path(repo_id).write_text('{"repo_id":"owner-repo-abc123"}', encoding="utf-8")

    assert analyzer._cache_ready(repo_id) is False
