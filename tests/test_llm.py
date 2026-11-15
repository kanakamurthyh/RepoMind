from __future__ import annotations

import pytest
import requests

from app.config import Settings
from app.rag.grounding import GroundedPromptBuilder, verify_citations
from app.rag.llm import GeminiClient, LLMError, OpenAICompatibleClient, resolve_provider

CONTEXT = {
    "repo_facts": {"run_instructions": [{"command": "python app.py", "source": "README.md", "start_line": 10, "end_line": 12, "official": True}]},
    "selected_chunks": [{"file_path": "README.md", "start_line": 10, "end_line": 12, "text": "Run with python app.py"}],
    "source_references": [{"id": "S1", "file_path": "README.md", "start_line": 10, "end_line": 12}],
    "weakness_facts": [],
}


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text
        self.headers = {}

    def json(self):
        return self._body


def test_prompt_separates_system_rules_from_untrusted_sources():
    parts = GroundedPromptBuilder().build("how do I run this?", {"project_identity": {"project_name": "Demo"}}, CONTEXT)

    assert "untrusted data" in parts.system
    assert '<source id="S1" path="README.md" lines="10-10">' in parts.user
    assert "<question>\nhow do I run this?\n</question>" in parts.user
    assert parts.source_ids["S1"]["file_path"] == "README.md"


def test_prompt_redacts_secret_like_values():
    context = {**CONTEXT, "selected_chunks": [{"file_path": "settings.py", "start_line": 1, "end_line": 1, "text": "GEMINI_API_KEY='AIzaQ7vK2pX9mL4tR8wZ1nB5cY3dF6hJ0sAq'"}]}
    parts = GroundedPromptBuilder().build("is there a secret?", {}, context)

    assert "<redacted-secret>" in parts.user
    assert "AIzaQ7vK2pX9mL4tR8wZ1nB5cY3dF6hJ0sAq" not in str(parts)


def test_gemini_sends_key_in_header_not_url(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse(body={"candidates": [{"content": {"parts": [{"text": "hello"}]}}], "usageMetadata": {"promptTokenCount": 3}})

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    response = GeminiClient("test-key-123", "gemini-test").complete("Say hello", system="rules")

    assert response.text == "hello"
    assert "test-key-123" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "test-key-123"
    assert captured["json"]["systemInstruction"]["parts"][0]["text"] == "rules"
    assert response.usage["input_tokens"] == 3


def test_network_errors_never_expose_the_api_key(monkeypatch):
    def fake_post(url, **kwargs):
        raise requests.ConnectionError(f"Max retries exceeded with url: {url}?key=super-secret-key-value")

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    with pytest.raises(LLMError) as info:
        GeminiClient("super-secret-key-value", "gemini-test").complete("hi")

    assert info.value.code == "network"
    assert "super-secret-key-value" not in str(info.value)
    assert "super-secret-key-value" not in info.value.detail


def test_retries_rate_limits_then_succeeds(monkeypatch):
    responses = [FakeResponse(429, text="slow down"), FakeResponse(body={"candidates": [{"content": {"parts": [{"text": "ok"}]}}]})]
    monkeypatch.setattr("app.rag.llm.requests.post", lambda url, **kw: responses.pop(0))
    client = GeminiClient("k", "gemini-test", sleep=lambda seconds: None)

    assert client.complete("hi").text == "ok"


def test_stale_model_falls_back_to_current_flash(monkeypatch):
    calls = []

    def fake_post(url, **kwargs):
        calls.append(url)
        if "gemini-1.5-flash" in url:
            return FakeResponse(404, text="not found")
        return FakeResponse(body={"candidates": [{"content": {"parts": [{"text": "fallback model worked"}]}}]})

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    client = GeminiClient("k", "gemini-1.5-flash")

    assert client.complete("hello").text == "fallback model worked"
    assert client.last_model_used == "gemini-flash-latest"


def test_openai_compatible_client_parses_chat_completions(monkeypatch):
    captured = {}

    def fake_post(url, **kwargs):
        captured["url"], captured["headers"] = url, kwargs["headers"]
        return FakeResponse(body={"model": "local-llama", "choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 5}})

    monkeypatch.setattr("app.rag.llm.requests.post", fake_post)
    response = OpenAICompatibleClient("k", "llama3", base_url="http://localhost:11434/v1").complete("hi", system="rules")

    assert captured["url"] == "http://localhost:11434/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert response.text == "hi" and response.model == "local-llama"


def test_provider_resolution():
    assert resolve_provider(Settings(llm_provider="off", gemini_api_key="x"))[0] == "local"
    assert resolve_provider(Settings(llm_provider="auto", gemini_api_key="x"))[0] == "gemini"
    assert resolve_provider(Settings(llm_provider="auto", gemini_api_key=None, openai_model="llama3", openai_base_url="http://localhost:11434/v1"))[0] == "openai"
    assert resolve_provider(Settings(llm_provider="gemini", gemini_api_key=None))[0] == "local"


def test_citation_verifier_checks_ids_and_line_ranges():
    report = verify_citations(
        "Direct answer:\nThe app starts in main [S1] and is configured in config.py:1-4 and in ghost.py:10-12 [S7].",
        {"S1": {"file_path": "app/main.py", "start_line": 1, "end_line": 20}},
        [{"file_path": "app/config.py", "start_line": 1, "end_line": 30}],
    )

    assert report.valid_ids == ["S1"] and report.invalid_ids == ["S7"]
    assert report.path_refs_verified == ["config.py:1-4"]
    assert report.path_refs_unverified == ["ghost.py:10-12"]
    assert 0 < report.grounding_score < 1
