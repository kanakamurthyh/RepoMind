from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from app.api.services import container
from app.main import app
from app.models import AnalysisResult


@pytest.fixture()
def client():
    return TestClient(app)


def test_health_endpoint_returns_ok(client):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("path", ["/repo/%2E%2E", "/repo/..%2F..", "/repo/owner-repo", "/repo/a%2Fb-0123456789"])
def test_delete_rejects_path_traversal_and_malformed_ids(client, path):
    deleted = []

    class Spy:
        def delete(self, repo_id):
            deleted.append(repo_id)

    container.analyzer = Spy()
    response = client.delete(path)

    assert response.status_code in {404, 422}
    assert deleted == []


def test_analysis_status_uses_cached_metadata_when_no_job(client):
    class Metadata:
        def latest_job(self, repo_id):
            return None

    class DummyAnalyzer:
        metadata = Metadata()

        def load(self, repo_id: str) -> AnalysisResult:
            return AnalysisResult(repo_id, "https://github.com/example/demo", {}, [], {}, [], ["completed"])

    container.analyzer = DummyAnalyzer()
    response = client.get("/analysis/example-demo-0123456789/status")

    assert response.status_code == 200
    assert response.json()["state"] == "completed"


def test_analyze_repo_reports_clone_failure_without_leaking_internals(client):
    from app.ingestion.git_client import GitError

    class DummyAnalyzer:
        def analyze(self, repo_url: str, force: bool = False):
            raise GitError("git clone failed: Could not resolve host: github.com /secret/internal/path")

    container.analyzer = DummyAnalyzer()
    response = client.post("/analyze_repo", json={"repo_url": "https://github.com/example/demo", "force": False})

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "clone_failed"
    assert "Could not clone the repository" in detail["message"]
    assert "/secret/internal/path" not in detail["message"]


def test_invalid_repo_url_is_400(client, shop_env):
    container.analyzer = shop_env[0]
    response = client.post("/analyze_repo", json={"repo_url": "https://gitlab.com/a/b"})
    assert response.status_code == 400


def test_api_key_is_enforced_when_configured(client, monkeypatch):
    from dataclasses import replace

    monkeypatch.setattr(container, "settings", replace(container.settings, api_key="s3cret-key"))
    assert client.get("/repos").status_code == 401
    assert client.get("/health").status_code == 200


def test_rate_limit_returns_429(client, monkeypatch, shop_env):
    from dataclasses import replace

    from app.api.security import limiter

    analyzer, chat, repo_id = shop_env
    container.analyzer = analyzer
    container.chat = chat
    monkeypatch.setattr(container, "settings", replace(container.settings, rate_limit_per_minute=2))
    limiter._hits.clear()
    codes = [client.post(f"/repo/{repo_id}/search", json={"query": "orders"}).status_code for _ in range(3)]
    limiter._hits.clear()
    assert codes == [200, 200, 429]


def test_repo_endpoints_on_analyzed_fixture(client, shop_env):
    analyzer, chat, repo_id = shop_env
    container.analyzer = analyzer
    container.chat = chat

    assert client.get(f"/repo/{repo_id}/summary").json()["project_identity"]["project_name"] == "ShopAPI"
    assert any(row["repo_id"] == repo_id for row in client.get("/repos").json())

    search = client.post(f"/repo/{repo_id}/search", json={"query": "How is the JWT token verified?", "k": 5}).json()
    assert search["results"][0]["file_path"] == "shopapi/auth.py"
    assert search["results"][0]["channels"]

    chat_response = client.post(f"/repo/{repo_id}/chat", json={"question": "Where is the order total computed?", "debug": True}).json()
    assert chat_response["citations"][0]["file_path"] == "shopapi/orders.py"
    assert "grounding" in chat_response and "debug" in chat_response
    assert "context" not in chat_response

    file_view = client.get(f"/repo/{repo_id}/file", params={"path": "shopapi/orders.py", "start": 1, "end": 5}).json()
    assert file_view["start_line"] == 1 and "import" in file_view["text"]
    assert client.get(f"/repo/{repo_id}/file", params={"path": "../../etc/passwd"}).status_code in {400, 404}

    symbols = client.get(f"/repo/{repo_id}/symbols", params={"name": "reserve_stock"}).json()
    assert symbols and any(caller["label"] == "create_order" for caller in symbols[0]["callers"])

    files_graph = client.get(f"/repo/{repo_id}/graph", params={"view": "files"}).json()
    assert any(edge["source"] == "shopapi/orders.py" and edge["target"] == "shopapi/inventory.py" for edge in files_graph["edges"])

    insights = client.get(f"/repo/{repo_id}/insights").json()
    assert insights["insights"]["docs_drift"]["issue_count"] >= 3

    investigation = client.post(f"/repo/{repo_id}/investigate", json={"question": "what calls reserve_stock?"}).json()
    assert investigation["mode"] == "deterministic"
    assert "create_order" in investigation["answer"]
    assert investigation["evidence"]

    high = client.get(f"/repo/{repo_id}/weaknesses", params={"severity": "high"}).json()
    assert all(item["severity"] == "high" for item in high)


def test_async_analysis_job_runs_to_completion(client, tmp_path):
    from evals.run_eval import FIXTURE_URL, build_environment

    analyzer, chat, _ = build_environment(tmp_path)
    container.analyzer = analyzer
    accepted = client.post("/analyze_repo_async", json={"repo_url": FIXTURE_URL, "force": True})
    assert accepted.status_code == 202
    job_id = accepted.json()["job_id"]
    for _ in range(100):
        job = client.get(f"/jobs/{job_id}").json()
        if job["state"] in {"completed", "failed"}:
            break
        time.sleep(0.1)
    assert job["state"] == "completed", job
    assert job["progress"] == 100
    events = client.get(f"/jobs/{job_id}/events").text
    assert "event: completed" in events


# --- error shape: every error is {"detail": {"code", "message"}} -----------------------------------
def _assert_error(response, status: int, code: str) -> dict:
    assert response.status_code == status, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, dict) and detail["code"] == code and isinstance(detail["message"], str) and detail["message"]
    return detail


class _MissingAnalyzer:
    """Analyzer whose artifacts are all missing (FileNotFoundError with an internal path)."""

    class metadata:  # noqa: N801 - mimics the attribute shape of MetadataRepository
        @staticmethod
        def latest_job(repo_id):
            return None

        @staticmethod
        def get_job(job_id):
            return None

    def load(self, repo_id):
        raise FileNotFoundError("/srv/internal/data/metadata/x.json")

    def graph_index(self, repo_id):
        return None

    def read_file(self, repo_id, path, start, end):
        if path == "bin.dat":
            raise ValueError("Binary files cannot be displayed.")
        raise FileNotFoundError(f"/srv/internal/repos/{path}")

    def delete(self, repo_id):
        raise ValueError("Invalid repository id.")


def test_not_found_errors_use_code_message_shape_without_leaking_paths(client):
    container.analyzer = _MissingAnalyzer()
    repo_id = "example-demo-0123456789"

    for path in (f"/repo/{repo_id}/summary", f"/repo/{repo_id}/weaknesses", f"/repo/{repo_id}/insights", f"/analysis/{repo_id}/status"):
        detail = _assert_error(client.get(path), 404, "analysis_not_found")
        assert "/srv/internal" not in detail["message"]
    _assert_error(client.get(f"/jobs/{'0' * 32}"), 404, "job_not_found")
    _assert_error(client.get(f"/jobs/{'0' * 32}/events"), 404, "job_not_found")
    _assert_error(client.get(f"/repo/{repo_id}/symbols", params={"name": "x"}), 404, "graph_missing")
    detail = _assert_error(client.get(f"/repo/{repo_id}/file", params={"path": "a.py"}), 404, "file_not_found")
    assert "/srv/internal" not in detail["message"]
    _assert_error(client.get(f"/repo/{repo_id}/file", params={"path": "bin.dat"}), 400, "invalid_input")


def test_framework_errors_are_normalised(client):
    container.analyzer = _MissingAnalyzer()
    detail = _assert_error(client.post("/analyze_repo", json={}), 422, "validation_error")
    assert "repo_url" in detail["message"] and detail["errors"]
    _assert_error(client.get("/no-such-endpoint"), 404, "not_found")
    _assert_error(client.put("/health"), 405, "method_not_allowed")
    # Global ValueError handler.
    _assert_error(client.delete("/repo/example-demo-0123456789"), 400, "invalid_input")


def test_auth_and_rate_limit_errors_are_normalised(client, monkeypatch):
    from dataclasses import replace

    from app.api.security import limiter

    monkeypatch.setattr(container, "settings", replace(container.settings, api_key="s3cret-key"))
    _assert_error(client.get("/repos"), 401, "unauthorized")

    monkeypatch.setattr(container, "settings", replace(container.settings, api_key=None, rate_limit_per_minute=1))
    container.analyzer = _MissingAnalyzer()
    limiter._hits.clear()
    client.post("/analyze_repo_async", json={"repo_url": "not a url"})
    _assert_error(client.post("/analyze_repo_async", json={"repo_url": "not a url"}), 429, "rate_limited")
    limiter._hits.clear()


def test_summary_of_legacy_cached_analysis_gains_coverage(client):
    class LegacyAnalyzer:
        def load(self, repo_id):
            summary = {"project_identity": {"project_name": "Old"}, "ingestion": {"files_seen": 10, "files_indexed": 7, "files_dropped": 3}}
            return AnalysisResult(repo_id, "https://github.com/example/demo", summary, [], {}, [], ["completed"])

    container.analyzer = LegacyAnalyzer()
    summary = client.get("/repo/example-demo-0123456789/summary").json()
    assert summary["coverage"]["partial"] is True
    assert summary["coverage"]["files_dropped_by_chunk_cap"] == 3
    insights = client.get("/repo/example-demo-0123456789/insights").json()
    assert insights["coverage"] == summary["coverage"]


def test_ready_reports_storage_backend(client, shop_env):
    container.analyzer = shop_env[0]
    container.chat = shop_env[1]
    ready = client.get("/ready").json()
    assert ready["status"] == "ready"
    assert ready["metadata_backend"] == "sqlalchemy"
    assert ready["storage"] == {
        "backend": "sqlalchemy", "configured_backend": "sqlalchemy", "degraded": False,
        "fallback_reason": None, "json_fallback_allowed": True,
    }


def test_ready_is_503_with_code_when_metadata_db_fails_fast(client, monkeypatch):
    from app.db.repository import MetadataStoreUnavailableError

    class Broken:
        settings = container.settings

        @property
        def analyzer(self):
            raise MetadataStoreUnavailableError("Metadata database is unavailable (OperationalError).")

    from app.api.services import get_container
    from app.main import app as fastapi_app

    fastapi_app.dependency_overrides[get_container] = lambda: Broken()
    try:
        detail = _assert_error(client.get("/ready"), 503, "storage_unavailable")
    finally:
        fastapi_app.dependency_overrides.clear()
    assert detail["storage"]["degraded"] is True
