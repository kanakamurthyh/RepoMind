from __future__ import annotations

import pytest

from app.config import Settings
from app.core.repo_analyzer import RepoAnalyzer


def test_cache_requires_metadata_vector_index_and_graph(tmp_path):
    analyzer = RepoAnalyzer(Settings(data_dir=tmp_path))
    (analyzer.settings.metadata_dir / "owner-repo-0123456789.json").write_text('{"repo_id":"owner-repo-0123456789"}', encoding="utf-8")
    assert analyzer.cache_ready("owner-repo-0123456789") is False


def test_delete_refuses_invalid_ids(tmp_path):
    analyzer = RepoAnalyzer(Settings(data_dir=tmp_path))
    (tmp_path / "keep.txt").write_text("important")
    for bad in ("..", "../..", "x/../..", ""):
        with pytest.raises(ValueError):
            analyzer.delete(bad)
    assert (tmp_path / "keep.txt").exists()


def test_end_to_end_analysis_of_fixture(shop_env):
    analyzer, _, repo_id = shop_env
    result = analyzer.load(repo_id)
    summary = result.summary

    assert summary["repo_facts"]["frameworks"][:1] == ["FastAPI"]
    assert [e["path"] for e in summary["repo_facts"]["main_entrypoints"]][:1] == ["shopapi/server.py"]
    assert any(item["path"] == "shopapi/orders.py" for item in summary["architecture"]["central_files"])
    assert summary["ingestion"]["files_dropped"] == 0
    assert 0 < summary["quality_score"]["total"] <= 100
    assert analyzer.graph_index(repo_id).callers("function:shopapi/inventory.py:reserve_stock")[0]["label"] == "create_order"
    assert analyzer.cache_ready(repo_id)
    assert analyzer.analyze("https://github.com/repomind-evals/shopapi").statuses == ["loaded cached analysis"]


def test_metadata_fallback_policy(tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from app.core.repo_analyzer import metadata_fallback_allowed

    base = Settings(data_dir=tmp_path)
    # Default "auto": the implicit local SQLite file may degrade; an explicit DATABASE_URL fails fast.
    assert metadata_fallback_allowed(replace(base, database_url="")) is True
    assert metadata_fallback_allowed(replace(base, database_url="postgresql+psycopg://db/x")) is False
    assert metadata_fallback_allowed(SimpleNamespace(database_url="postgresql://db/x", metadata_db_fallback="true")) is True
    assert metadata_fallback_allowed(SimpleNamespace(database_url="", metadata_db_fallback=False)) is False
    assert metadata_fallback_allowed(SimpleNamespace(database_url="", metadata_db_fallback="auto")) is True


def test_analyzer_with_unreachable_explicit_database_fails_fast(tmp_path):
    from dataclasses import replace

    from app.db.repository import MetadataStoreUnavailableError

    with pytest.raises(MetadataStoreUnavailableError):
        RepoAnalyzer(replace(Settings(data_dir=tmp_path), database_url="notadialect://db/x"))
