from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event
from unittest.mock import patch

import pytest

from app.config import Settings
from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.graph.code_graph import CodeGraphBuilder
from app.rag.vector_store import IndexStaleError
from evals.run_eval import LocalGitClient

URL = "https://github.com/example/snapshot--test"
TOKEN = "ghp_" + "Z7bQ9mN2vR5sT8wX3yH6jK4pL0dF1cA9"


@pytest.fixture
def environment(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def original_function():\n    return 1\n")
    git = LocalGitClient(repo)
    git.head_sha = lambda path: "a" * 40
    config = replace(Settings(), data_dir=tmp_path / "data", parser_provider="regex", database_url="")
    analyzer = RepoAnalyzer(config, git_client=git)
    return repo, analyzer, analyzer.analyze(URL)


@pytest.mark.parametrize("component,method", [
    ("vector_store", "build"), ("graph_store", "save"), ("summary_generator", "generate"),
    ("metadata", "save"), ("snapshots", "publish"),
])
def test_failed_refresh_keeps_old_snapshot_and_discards_unpublished_artifacts(environment, component, method):
    repo, analyzer, original = environment
    before = set(analyzer.snapshots._repo(original.repo_id).iterdir())
    (repo / "main.py").write_text("def replacement_function():\n    return 42\n")
    analyzer.git.head_sha = lambda path: "b" * 40
    with patch.object(getattr(analyzer, component), method, side_effect=RuntimeError("injected boundary failure")):
        with pytest.raises(RuntimeError, match="injected"):
            analyzer.analyze(URL, force=True)
    loaded = analyzer.load(original.repo_id)
    assert loaded.commit_sha == "a" * 40
    assert analyzer.cache_ready(original.repo_id)
    assert analyzer.read_file(original.repo_id, "main.py")["text"].startswith("def original_function")
    assert analyzer.graph_index(original.repo_id).find_symbols("original_function")
    assert all("replacement_function" not in c.text for c in analyzer.vector_store.all_chunks(original.repo_id))
    assert set(analyzer.snapshots._repo(original.repo_id).iterdir()) == before
    assert analyzer.list_repos()[0]["commit_sha"] == "a" * 40


def test_reader_is_pinned_across_successful_refresh_and_old_source_remains_available(environment):
    repo, analyzer, first = environment
    with analyzer.read_snapshot(first.repo_id):
        (repo / "main.py").write_text("def replacement_function():\n    return 42\n")
        with ThreadPoolExecutor(max_workers=1) as executor:
            second = executor.submit(analyzer.analyze, URL, True).result(timeout=20)
        assert analyzer.load(first.repo_id).summary["ingestion"]["run_id"] == first.summary["ingestion"]["run_id"]
        assert analyzer.graph_index(first.repo_id).find_symbols("original_function")
        assert "original_function" in analyzer.read_file(first.repo_id, "main.py")["text"]
        assert all("replacement_function" not in c.text for c in analyzer.vector_store.all_chunks(first.repo_id))
    assert analyzer.graph_index(first.repo_id).find_symbols("replacement_function")
    assert analyzer.load(first.repo_id).summary["ingestion"]["run_id"] == second.summary["ingestion"]["run_id"]
    assert "original_function" in analyzer.read_file(first.repo_id, "main.py", run_id=first.summary["ingestion"]["run_id"])["text"]


def test_two_repositories_keep_separate_graphs_under_controlled_interleaving(tmp_path):
    roots = {}
    for name in ("alpha", "beta"):
        path = tmp_path / name
        path.mkdir()
        (path / f"{name}.py").write_text(f"def {name}_unique():\n    return 1\n")
        roots[f"https://github.com/example/{name}"] = path
    class Git(LocalGitClient):
        def clone_or_get(self, url, refresh=False):
            from app.utils.paths import repo_id_from_url
            return repo_id_from_url(url), roots[url], True
    analyzer = RepoAnalyzer(replace(Settings(), data_dir=tmp_path / "data", database_url=""), git_client=Git(tmp_path))
    original_build = CodeGraphBuilder.build
    alpha_built, release_alpha = Event(), Event()
    def interleave(builder, records, parsed=None):
        result = original_build(builder, records, parsed)
        if records[0].path == "alpha.py":
            alpha_built.set()
            assert release_alpha.wait(timeout=15)
        return result
    with patch.object(CodeGraphBuilder, "build", interleave), ThreadPoolExecutor(max_workers=2) as executor:
        alpha = executor.submit(analyzer.analyze, "https://github.com/example/alpha")
        try:
            assert alpha_built.wait(timeout=10)
            beta = executor.submit(analyzer.analyze, "https://github.com/example/beta").result(timeout=15)
        finally:
            release_alpha.set()
        alpha = alpha.result(timeout=15)
    assert analyzer.graph_index(alpha.repo_id).find_symbols("alpha_unique")
    assert not analyzer.graph_index(alpha.repo_id).find_symbols("beta_unique")
    assert analyzer.graph_index(beta.repo_id).find_symbols("beta_unique")


def test_snapshot_sources_metadata_chat_and_index_do_not_expose_credentials(environment):
    repo, analyzer, first = environment
    (repo / "README.md").write_text(f"# Demo\n\n## Run\n```bash\ncurl -H 'Authorization: Bearer {TOKEN}' https://example.invalid\n```\n")
    result = analyzer.analyze(URL, force=True)
    loaded = analyzer.load(result.repo_id, include_chunks=True)
    assert TOKEN not in json.dumps(loaded.to_dict())
    assert TOKEN not in analyzer.read_file(first.repo_id, "README.md")["text"]
    answer = ChatService(analyzer, analyzer.settings).answer(first.repo_id, "How do I run this?")
    assert TOKEN not in json.dumps(answer)
    assert answer["snapshot"]["run_id"] == loaded.summary["ingestion"]["run_id"]
    assert all(c["run_id"] == answer["snapshot"]["run_id"] for c in answer["citations"])


@pytest.mark.parametrize("damage", ["missing", "shape", "invalid"])
def test_dense_artifacts_are_validated_even_when_index_is_cached(environment, damage):
    _, analyzer, result = environment
    import numpy as np
    analyzer.vector_store.load(result.repo_id)
    key = analyzer.snapshots.artifact_id(result.repo_id)
    path = analyzer.vector_store.dense.path(key)
    if damage == "missing":
        path.unlink()
    elif damage == "shape":
        np.save(path, np.zeros((3, 3), dtype=np.float32))
    else:
        path.write_bytes(b"not a numpy matrix")
    assert not analyzer.cache_ready(result.repo_id)
    with pytest.raises(IndexStaleError):
        analyzer.vector_store.search(result.repo_id, "original_function")


def test_per_file_truncation_and_parse_failure_are_reported(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "main.py").write_text("def sample():\n    return '" + "x" * 300)
    analyzer = RepoAnalyzer(replace(Settings(), data_dir=tmp_path / "data", max_file_size_bytes=40, database_url=""), git_client=LocalGitClient(repo))
    coverage = analyzer.analyze(URL).summary["coverage"]
    assert coverage["partial"]
    assert coverage["truncated_files"] == 1
    assert coverage["parse_failures"] == 1
