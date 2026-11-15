"""Coverage reporting: scan/chunk caps and duplicate collapsing are never silent."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.core.repo_analyzer import RepoAnalyzer
from app.ingestion.coverage import build_coverage, coverage_message, ensure_coverage
from app.ingestion.scanner import FileScanner
from app.models import FileRecord
from app.parsing.chunker import Chunker
from evals.run_eval import LocalGitClient


def _write_repo(root: Path, count: int, body: str = "def f{i}():\n    return {i}\n") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "README.md").write_text("# Demo\n\nA demo project.\n", encoding="utf-8")
    pkg = root / "pkg"
    pkg.mkdir(exist_ok=True)
    for i in range(count):
        (pkg / f"mod_{i:02d}.py").write_text(body.format(i=i), encoding="utf-8")
    return root


def _record(path: str, text: str, sha: str | None, role: str = "code", truncated: bool = False) -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", "python", role, len(text), text.count("\n") + 1, text, truncated=truncated, sha256=sha)


# --- scanner -------------------------------------------------------------------------------
def test_scanner_file_cap_counts_and_samples_skipped_files(tmp_path):
    repo = _write_repo(tmp_path / "repo", 6)  # 7 files: README.md + 6 modules

    scanner = FileScanner(max_file_size_bytes=10_000, max_files_per_repo=3)
    records = scanner.scan("demo", repo)
    stats = scanner.last_scan_stats

    assert len(records) == 3
    assert stats["limit_reached"] is True and stats["stop_reason"] == "max_files_per_repo"
    assert stats["files_scanned"] == 3 and stats["files_skipped_by_cap"] == 4 and stats["files_seen"] == 7
    assert stats["skipped_by_cap_exact"] is True
    assert len(stats["skipped_by_cap_sample"]) == 4
    assert not {r.path for r in records} & set(stats["skipped_by_cap_sample"])


def test_scanner_byte_cap_records_reason(tmp_path):
    repo = _write_repo(tmp_path / "repo", 4, body="x = '" + "a" * 100 + "'  # {i}\n")
    scanner = FileScanner(max_file_size_bytes=10_000, max_total_repo_bytes=250)
    scanner.scan("demo", repo)

    stats = scanner.last_scan_stats
    assert stats["stop_reason"] == "max_total_repo_bytes"
    assert stats["files_skipped_by_cap"] >= 1
    assert stats["files_seen"] == 5 == stats["files_scanned"] + stats["files_skipped_by_cap"]


def test_scanner_without_cap_reports_complete_scan(tmp_path):
    repo = _write_repo(tmp_path / "repo", 2)
    scanner = FileScanner(max_file_size_bytes=10_000)
    scanner.scan("demo", repo)

    assert scanner.last_scan_stats["stop_reason"] is None
    assert scanner.last_scan_stats["files_skipped_by_cap"] == 0
    assert scanner.last_scan_stats["files_seen"] == 3


# --- chunker -------------------------------------------------------------------------------
def test_chunk_cap_records_dropped_files_and_keeps_legacy_field():
    records = [_record(f"pkg/m{i}.py", f"def f{i}():\n    return {i}\n", f"sha{i}") for i in range(5)]
    chunker = Chunker()
    chunks = chunker.chunk_files(records, max_chunks=2)

    stats = chunker.last_stats
    assert len(chunks) == 2
    assert stats["chunk_cap_reached"] is True and stats["chunk_cap"] == 2
    assert stats["files_dropped"] == stats["files_dropped_by_chunk_cap"] == 3
    assert stats["files_indexed"] == 2
    assert stats["dropped_paths_sample"] == ["pkg/m2.py", "pkg/m3.py", "pkg/m4.py"]


def test_identical_files_are_indexed_once_and_duplicates_recorded():
    body = "def shared():\n    return 1\n"
    records = [
        _record("a/util.py", body, "same"),
        _record("b/util.py", body, "same"),
        _record("c/util.py", body, "same"),
        _record("d/other.py", "def other():\n    return 2\n", "other"),
        # Truncated copies hash only their prefix, so they are never collapsed.
        _record("e/big.py", body, "same", truncated=True),
    ]
    chunker = Chunker()
    chunks = chunker.chunk_files(records, max_chunks=100)

    indexed = {chunk.file_path for chunk in chunks}
    assert indexed == {"a/util.py", "d/other.py", "e/big.py"}
    assert chunker.last_stats["duplicates_collapsed"] == 2
    assert chunker.last_stats["files_dropped"] == 0
    canonical = next(chunk for chunk in chunks if chunk.file_path == "a/util.py")
    assert canonical.metadata["duplicate_paths"] == ["b/util.py", "c/util.py"]
    assert chunker.last_stats["duplicate_groups_sample"] == [{"indexed_path": "a/util.py", "duplicate_paths": ["b/util.py", "c/util.py"], "count": 2}]


def test_duplicate_collapsing_can_be_disabled():
    body = "def shared():\n    return 1\n"
    chunker = Chunker(collapse_duplicates=False)
    chunks = chunker.chunk_files([_record("a/u.py", body, "s"), _record("b/u.py", body, "s")], max_chunks=100)

    assert {chunk.file_path for chunk in chunks} == {"a/u.py", "b/u.py"}
    assert chunker.last_stats["duplicates_collapsed"] == 0


# --- coverage object -------------------------------------------------------------------------
def test_build_coverage_complete_and_partial():
    complete = build_coverage({"files_seen": 10, "files_scanned": 10}, {"files_indexed": 8, "files_dropped": 0})
    assert complete["partial"] is False and complete["reasons"] == [] and complete["analyzed_fraction"] == 1.0
    assert coverage_message(complete) == ""

    partial = build_coverage(
        {"files_seen": 10, "files_scanned": 6, "files_skipped_by_cap": 4, "stop_reason": "max_files_per_repo", "skipped_by_cap_sample": ["z.py"]},
        {"files_indexed": 4, "files_dropped_by_chunk_cap": 1, "chunk_cap": 50, "chunk_cap_reached": True, "dropped_paths_sample": ["tests/t.py"]},
        {"max_files_per_repo": 6, "chunk_cap": 50},
    )
    assert partial["partial"] is True
    assert partial["analyzed_fraction"] == 0.5
    assert [reason["code"] for reason in partial["reasons"]] == ["max_files_per_repo", "chunk_cap"]
    assert partial["reasons"][0]["limit"] == 6 and partial["reasons"][0]["files_affected"] == 4
    assert partial["sample_dropped_paths"] == ["z.py", "tests/t.py"]
    assert "50%" in coverage_message(partial) and "MAX_FILES_PER_REPO=6" in coverage_message(partial)


def test_ensure_coverage_upgrades_legacy_summaries():
    legacy = {"ingestion": {"files_seen": 20, "files_indexed": 15, "files_dropped": 5, "chunks": 40}}
    upgraded = ensure_coverage(legacy)

    assert "coverage" not in legacy
    assert upgraded["coverage"]["partial"] is True
    assert upgraded["coverage"]["files_dropped_by_chunk_cap"] == 5
    assert upgraded["coverage"]["derived_from_legacy_stats"] is True

    hit_scan_cap = ensure_coverage({"ingestion": {"files_seen": 20, "limit_reached": True}})["coverage"]
    assert hit_scan_cap["partial"] is True and hit_scan_cap["reasons"][0]["code"] == "scan_limit"
    assert hit_scan_cap["files_skipped_count_exact"] is False

    assert ensure_coverage({})["coverage"]["partial"] is False
    current = {"coverage": {"partial": False}}
    assert ensure_coverage(current) is current


# --- end to end ------------------------------------------------------------------------------
def test_analysis_of_repo_exceeding_caps_reports_partial_coverage(tmp_path):
    repo = _write_repo(tmp_path / "repo", 10)
    (repo / "pkg" / "copy_a.py").write_text("def same():\n    return 'dup'\n", encoding="utf-8")
    (repo / "pkg" / "copy_b.py").write_text("def same():\n    return 'dup'\n", encoding="utf-8")
    settings = replace(
        Settings(), data_dir=tmp_path / "data", max_files_per_repo=8, max_chunks_per_repo=4,
        osv_scan_enabled=False, github_api_checks_enabled=False, llm_provider="off", database_url="",
    )
    analyzer = RepoAnalyzer(settings, git_client=LocalGitClient(repo))

    result = analyzer.analyze("https://github.com/example/big-repo")
    coverage = result.summary["coverage"]

    assert coverage["partial"] is True
    assert coverage["files_seen"] == 13
    assert coverage["files_skipped_by_cap"] == 5
    assert coverage["files_dropped_by_chunk_cap"] >= 1
    assert coverage["duplicates_collapsed"] == 1  # copy_a/copy_b sort first, inside the file cap
    assert {reason["code"] for reason in coverage["reasons"]} == {"max_files_per_repo", "chunk_cap"}
    assert coverage["sample_dropped_paths"]
    assert 0 < coverage["analyzed_fraction"] < 1
    # Long-standing ingestion fields keep working.
    ingestion = result.summary["ingestion"]
    assert ingestion["files_dropped"] == coverage["files_dropped_by_chunk_cap"] and ingestion["limit_reached"] is True
    assert ingestion["coverage_partial"] is True
    assert any(status.startswith("Partial analysis:") for status in result.statuses)
    # Persisted and reloaded unchanged.
    assert analyzer.load(result.repo_id).summary["coverage"] == coverage


# --- Streamlit banner ------------------------------------------------------------------------
def _run_ui(monkeypatch, summary, findings=None, return_app=False):
    import requests
    from streamlit.testing.v1 import AppTest

    class FakeResponse:
        ok, status_code, reason, text = True, 200, "OK", ""

        def __init__(self, payload):
            self._payload = payload

        def json(self):
            return self._payload

    def fake_request(method, url, **kwargs):
        if url.endswith("/ready"):
            return FakeResponse({"version": "test", "storage": {"backend": "sqlalchemy", "degraded": False}})
        if url.endswith("/repos"):
            return FakeResponse([])
        if "/graph" in url:
            return FakeResponse({"nodes": [], "edges": [], "mermaid": "flowchart TD\nA[Repo]"})
        if url.endswith("/weaknesses"):
            return FakeResponse(findings or [])
        return FakeResponse({})

    monkeypatch.setattr(requests, "request", fake_request)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app" / "ui" / "streamlit_app.py"), default_timeout=30)
    app.session_state["summary"] = summary
    app.session_state["repo_id"] = "example-demo-0123456789"
    app.session_state["graph"] = {}
    app.run()
    assert not app.exception, app.exception
    return app if return_app else [warning.value for warning in app.warning]


def test_ui_shows_partial_coverage_banner(monkeypatch):
    coverage = build_coverage(
        {"files_seen": 10, "files_scanned": 6, "files_skipped_by_cap": 4, "stop_reason": "max_files_per_repo", "skipped_by_cap_sample": ["z.py"]},
        {"files_indexed": 6},
        {"max_files_per_repo": 6},
    )
    warnings = _run_ui(monkeypatch, {"project_identity": {"project_name": "Big"}, "ingestion": {"files_indexed": 6}, "coverage": coverage})

    banner = next(text for text in warnings if "Partial analysis" in text)
    assert "60%" in banner and "6 of 10 files" in banner and "MAX_FILES_PER_REPO=6" in banner


def test_ui_renders_legacy_summary_without_coverage(monkeypatch):
    warnings = _run_ui(monkeypatch, {"project_identity": {"project_name": "Old"}, "ingestion": {"files_indexed": 3, "chunks": 9}})
    assert not any("Partial analysis" in text for text in warnings)


def test_ui_displays_full_static_findings_and_unknown_dependency_state(monkeypatch):
    app = _run_ui(monkeypatch, {"project_identity": {"project_name": "Demo"},
                               "insights": {"dependencies": {"scan_status": "offline"}}},
                  findings=[{"severity": "high", "title": "Sentinel security finding", "category": "security",
                             "why_it_matters": "A demonstrated problem", "suggested_fix": "Repair it", "file_path": "main.py"}], return_app=True)
    assert any("Sentinel security finding" in item.label for item in app.get("expander"))
    assert any("unknown" in caption.value for caption in app.caption)
