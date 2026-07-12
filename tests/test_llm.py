from __future__ import annotations

from app.rag.llm import GeminiClient, GroundedPromptBuilder


def test_grounded_prompt_contains_evidence_and_hides_retrieval_instructions():
    prompt = GroundedPromptBuilder().build(
        "how do I run this?",
        {"project_identity": {"project_name": "Demo"}},
        {
            "repo_facts": {
                "run_instructions": [
                    {
                        "command": "python app.py",
                        "source": "README.md",
                        "start_line": 10,
                        "end_line": 12,
                        "official": True,
                    }
                ],
                "important_files": [{"path": "README.md", "reason": "official docs"}],
            },
            "selected_chunks": [
                {
                    "file_path": "README.md",
                    "start_line": 10,
                    "end_line": 12,
                    "text": "Run with python app.py",
                }
            ],
            "source_references": [{"file_path": "README.md", "start_line": 10, "end_line": 12}],
            "weakness_facts": [],
        },
        "Direct answer:\nUse README.md.",
    )

    assert "python app.py" in prompt
    assert "README.md:10-12" in prompt
    assert "Do not mention retrieval mode" in prompt
    assert "chunk IDs" in prompt


def test_grounded_prompt_redacts_secret_like_source_values():
    prompt = GroundedPromptBuilder().build(
        "is there a secret?",
        {"project_identity": {"project_name": "Demo"}},
        {
            "repo_facts": {},
            "selected_chunks": [
                {
                    "file_path": "settings.py",
                    "start_line": 1,
                    "end_line": 1,
                    "text": "GEMINI_API_KEY='AIzaabcdefghijklmnopqrstuvwxyz12345'",
                }
            ],
            "source_references": [{"file_path": "settings.py", "start_line": 1, "end_line": 1}],
            "weakness_facts": [],
        },
        "Direct answer:\nA key-like value was detected.",
    )

    assert "<redacted-secret>" in prompt
    assert "AIzaabcdefghijklmnopqrstuvwxyz12345" not in prompt


def test_gemini_client_uses_certifi_verified_requests(monkeypatch):
    captured = {}

    class FakeResponse:
        text = ""

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"candidates": [{"content": {"parts": [{"text": "hello from Gemini"}]}}]}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    monkeypatch.setattr("app.rag.llm.certifi.where", lambda: "/tmp/cacert.pem")

    answer = GeminiClient("test-key", "gemini-test").generate("Say hello")

    assert answer == "hello from Gemini"
    assert captured["verify"] == "/tmp/cacert.pem"
    assert captured["json"]["contents"][0]["parts"][0]["text"] == "Say hello"


def test_gemini_client_retries_current_model_when_configured_model_is_stale(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, status_code: int, text: str, body: dict | None = None) -> None:
            self.status_code = status_code
            self.text = text
            self._body = body or {}

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError("boom", response=self)

        def json(self) -> dict:
            return self._body

    def fake_post(url, **kwargs):
        calls.append(url)
        if "gemini-1.5-flash" in url:
            return FakeResponse(404, '{"error":"not found"}')
        return FakeResponse(200, "", {"candidates": [{"content": {"parts": [{"text": "fallback model worked"}]}}]})

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    monkeypatch.setattr("app.rag.llm.certifi.where", lambda: "/tmp/cacert.pem")

    client = GeminiClient("test-key", "gemini-1.5-flash")
    answer = client.generate("hello")

    assert answer == "fallback model worked"
    assert client.last_model_used == "gemini-flash-latest"
    assert len(calls) == 2
