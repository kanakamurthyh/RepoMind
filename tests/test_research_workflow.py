from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from threading import Event
from types import SimpleNamespace

import pytest

from app.api.services import JobRunner, QueueFullError, ServiceContainer
from app.cli import main
from app.config import Settings
from app.db.repository import MetadataRepository
from app.graph.code_graph import CodeGraphBuilder
from app.models import FileRecord
from evals.metrics import alternative_retrieval_metrics, retrieval_metrics
from evals.run_corpus import span_recall, validate_corpus


def test_metrics_distinguish_any_hit_fractional_recall_and_all_evidence():
    scores = retrieval_metrics([["a", "a", "other", "b"], ["irrelevant"]], [{"a", "b"}, set()])
    assert scores["hit@1"] == 1
    assert scores["recall@1"] == .5
    assert scores["complete@1"] == 0
    assert scores["complete@3"] == 1
    assert alternative_retrieval_metrics([["c"]], [[{"a", "b"}, {"c"}]])["complete@1"] == 1
    with pytest.raises(ValueError):
        retrieval_metrics([], [{"a"}])


def test_span_recall_requires_full_ranges_and_allows_alternative_evidence():
    spans = [[{"file_path": "a.py", "start_line": 1, "end_line": 10}]]
    assert span_recall([{"file_path": "a.py", "start_line": 1, "end_line": 5}], spans) == 0
    assert span_recall([{"file_path": "a.py", "start_line": 1, "end_line": 10}], spans) == 1


def test_unreviewed_corpus_and_mismatched_commits_cannot_be_quality_gates():
    corpus = {"repositories": [{"name": "example/tiny", "commit_sha": "a" * 40, "split": "dev"}],
              "questions": [{"id": "q1", "repo": "example/tiny", "commit_sha": "a" * 40,
                             "review_status": "draft", "answerable": False, "evidence_sets": []}]}
    with pytest.raises(ValueError, match="Unreviewed"):
        validate_corpus(corpus)
    validate_corpus(corpus, allow_draft=True)
    modified = deepcopy(corpus)
    modified["questions"][0]["commit_sha"] = "b" * 40
    with pytest.raises(ValueError, match="commit"):
        validate_corpus(modified, allow_draft=True)


def test_cli_local_analysis_ask_and_sarif_preserve_local_directory(tmp_path, capsys):
    repo = tmp_path / "source"
    repo.mkdir()
    (repo / "main.py").write_text("def hello():\n    return 'hello'\n")
    (repo / "README.md").write_text("# Hello\n\nA small greeting library for local command-line testing.\n")
    args = ["--data-dir", str(tmp_path / "data")]
    assert main([*args, "analyze", str(repo)]) == 0
    analysis = json.loads(capsys.readouterr().out)
    assert analysis["summary"]["ingestion"]["source_kind"] == "local"
    assert main([*args, "ask", analysis["repo_id"], "What does hello do?"]) == 0
    answer = json.loads(capsys.readouterr().out)
    assert answer["citations"] and answer["snapshot"]["run_id"]
    assert main([*args, "check", str(repo), "--format", "sarif", "--fail-on", "none"]) == 0
    assert json.loads(capsys.readouterr().out)["version"] == "2.1.0"
    assert (repo / "main.py").exists()


def test_queue_admission_is_bounded_and_duplicate_jobs_share_capacity(tmp_path):
    metadata = MetadataRepository("", tmp_path, enabled=False)
    started, release = Event(), Event()
    def analyze(*args, **kwargs):
        started.set()
        assert release.wait(timeout=10)
        return SimpleNamespace(statuses=["complete"])
    runner = JobRunner(SimpleNamespace(metadata=metadata, analyze=analyze), workers=1, capacity=1)
    try:
        first = runner.submit("https://github.com/example/one")
        assert started.wait(timeout=5)
        assert runner.submit("https://github.com/example/one")["job_id"] == first["job_id"]
        with pytest.raises(QueueFullError):
            runner.submit("https://github.com/example/two")
    finally:
        release.set()
        runner.shutdown()
    assert metadata.get_job(first["job_id"])["state"] == "completed"


def test_json_jobs_survive_restart_and_are_reconciled(tmp_path):
    first = MetadataRepository("", tmp_path, enabled=False)
    job = first.create_job("example-one-0123456789", "https://github.com/example/one")
    first.update_job(job["job_id"], state="running")
    second = MetadataRepository("", tmp_path, enabled=False)
    assert second.get_job(job["job_id"])["state"] == "running"
    assert second.mark_interrupted_jobs() == 1
    assert MetadataRepository("", tmp_path, enabled=False).get_job(job["job_id"])["error_code"] == "interrupted"


def test_second_api_worker_is_rejected_before_reconciling_live_jobs(tmp_path):
    settings = replace(Settings(), data_dir=tmp_path, database_url="")
    first, second = ServiceContainer(settings), ServiceContainer(settings)
    first.startup()
    try:
        job = first.analyzer.metadata.create_job("example-one-0123456789", "https://github.com/example/one")
        with pytest.raises(RuntimeError, match="already has an API worker"):
            second.startup()
        assert first.analyzer.metadata.get_job(job["job_id"])["state"] == "queued"
    finally:
        first.shutdown()
        second.shutdown()


def test_python_import_aliases_resolve_and_call_cycles_are_not_import_cycles():
    def rec(path, text):
        return FileRecord("r", path, path, "python", "code", len(text), len(text.splitlines()), text)
    graph = CodeGraphBuilder().build([
        rec("a.py", "from b import some_work as renamed\ndef root_task():\n    return renamed()\n"),
        rec("b.py", "def some_work():\n    return root_task()\n"),
    ])
    assert any(e["source"] == "function:a.py:root_task" and e["target"] == "function:b.py:some_work" for e in graph["edges"])
    assert graph["dependency_cycles"] == []


def test_concurrent_snapshot_writers_are_serialized_across_store_instances(tmp_path):
    from app.core.snapshots import SnapshotStore
    first, second = SnapshotStore(tmp_path), SnapshotStore(tmp_path)
    entered, attempted = Event(), Event()
    def writer():
        attempted.set()
        with second.writer_lock("example-one-0123456789"):
            entered.set()
    with ThreadPoolExecutor(max_workers=1) as executor:
        with first.writer_lock("example-one-0123456789"):
            future = executor.submit(writer)
            assert attempted.wait(timeout=5)
            assert not entered.wait(timeout=.1)
        future.result(timeout=5)
    assert entered.is_set()


def test_model_identity_keeps_explicit_revision_without_loading_optional_providers():
    from app.rag.model_identity import model_revision
    assert model_revision("unused", SimpleNamespace(_commit_hash="a" * 40)) == "a" * 40
