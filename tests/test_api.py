from __future__ import annotations

from fastapi.testclient import TestClient

import app.api.routes as routes
from app.main import app
from app.models import AnalysisResult


def test_health_endpoint_returns_ok():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analysis_status_uses_cached_metadata_when_job_missing(monkeypatch):
    class DummyAnalyzer:
        def load(self, repo_id: str) -> AnalysisResult:
            return AnalysisResult(
                repo_id=repo_id,
                repo_url="https://github.com/example/demo",
                summary={},
                weaknesses=[],
                graph={},
                chunks=[],
                statuses=["completed"],
            )

    monkeypatch.setattr(routes, "analyzer", DummyAnalyzer())
    routes.analysis_jobs.clear()
    client = TestClient(app)

    response = client.get("/analysis/demo/status")

    assert response.status_code == 200
    assert response.json()["state"] == "completed"
