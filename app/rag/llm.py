"""Provider-agnostic LLM clients. Every provider is optional; failures surface as coded errors."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import certifi
import requests

from app.config import Settings
from app.core.interfaces import LLMProvider
from app.utils.redaction import redact_data, scrub

logger = logging.getLogger(__name__)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}
PUBLIC_ERROR_MESSAGES = {
    "auth": "The LLM provider rejected the API key.",
    "rate_limited": "The LLM provider is rate limiting requests.",
    "timeout": "The LLM provider timed out.",
    "network": "The LLM provider could not be reached.",
    "model_not_found": "The configured LLM model was not found.",
    "bad_response": "The LLM provider returned an unusable response.",
    "blocked": "The LLM provider blocked the response.",
    "http_error": "The LLM provider returned an error.",
    "not_configured": "No LLM provider is configured.",
}


class LLMError(RuntimeError):
    """Coded provider error. `str(error)` is always safe to show to users (no secrets, no raw bodies)."""

    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        super().__init__(PUBLIC_ERROR_MESSAGES.get(code, PUBLIC_ERROR_MESSAGES["http_error"]))


@dataclass(slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: int = 0


class _HTTPProvider(LLMProvider):
    name = "http"

    def __init__(self, api_key: str, model: str, timeout_seconds: int = 45, max_retries: int = 2, sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.sleep = sleep
        self.last_model_used: str | None = None

    def _post(self, url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                response = requests.post(url, json=redact_data(payload), headers=headers, timeout=self.timeout_seconds, verify=certifi.where())
            except requests.Timeout as exc:
                self._log_failure("timeout", exc)
                if attempt < self.max_retries:
                    attempt += 1
                    self.sleep(2**attempt)
                    continue
                raise LLMError("timeout", scrub(str(exc), self.api_key)) from None
            except requests.RequestException as exc:
                self._log_failure("network", exc)
                raise LLMError("network", scrub(str(exc), self.api_key)) from None
            status = getattr(response, "status_code", 200)
            if status in RETRYABLE_STATUS and attempt < self.max_retries:
                attempt += 1
                retry_after = response.headers.get("retry-after") if hasattr(response, "headers") else None
                self.sleep(min(20.0, float(retry_after)) if retry_after and str(retry_after).isdigit() else 2**attempt)
                continue
            if status >= 400:
                body = scrub(getattr(response, "text", "")[:400], self.api_key)
                code = {401: "auth", 403: "auth", 404: "model_not_found", 429: "rate_limited"}.get(status, "http_error")
                logger.warning("%s HTTP %s for model %s: %s", self.name, status, self.model, body)
                raise LLMError(code, f"HTTP {status}")
            try:
                return response.json()
            except (ValueError, json.JSONDecodeError) as exc:
                raise LLMError("bad_response", "invalid JSON") from exc

    def _log_failure(self, code: str, exc: Exception) -> None:
        logger.warning("%s request failed (%s): %s", self.name, code, scrub(str(exc), self.api_key))


class GeminiClient(_HTTPProvider):
    name = "gemini"
    FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite")

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False, max_tokens: int = 1400) -> LLMResponse:
        payload: dict[str, Any] = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "topP": 0.8, "maxOutputTokens": max_tokens},
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"
        headers = {"Content-Type": "application/json", "x-goog-api-key": self.api_key}
        started = time.perf_counter()
        last_error: LLMError | None = None
        body: dict[str, Any] | None = None
        for model in self._candidate_models():
            try:
                body = self._post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent", payload, headers)
                self.last_model_used = model
                break
            except LLMError as exc:
                last_error = exc
                if exc.code != "model_not_found":
                    raise
        if body is None:
            raise last_error or LLMError("model_not_found")
        try:
            candidate = body["candidates"][0]
            if candidate.get("finishReason") in {"SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"}:
                raise LLMError("blocked", candidate.get("finishReason", ""))
            text = "".join(part.get("text", "") for part in candidate["content"]["parts"]).strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("bad_response", "missing candidates") from exc
        if not text:
            raise LLMError("bad_response", "empty answer")
        usage = body.get("usageMetadata", {})
        return LLMResponse(text, "gemini", self.last_model_used or self.model, {
            "input_tokens": usage.get("promptTokenCount"), "output_tokens": usage.get("candidatesTokenCount"),
        }, int((time.perf_counter() - started) * 1000))

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text

    def _candidate_models(self) -> list[str]:
        candidates = [self.model, *self.FALLBACK_MODELS]
        seen: set[str] = set()
        return [model for model in candidates if model and not (model in seen or seen.add(model))]


class OpenAICompatibleClient(_HTTPProvider):
    """OpenAI Chat Completions API shape: works with OpenAI, Ollama, vLLM, LM Studio, Groq, etc."""

    name = "openai"

    def __init__(self, api_key: str, model: str, base_url: str = "https://api.openai.com/v1", **kwargs: Any) -> None:
        super().__init__(api_key, model, **kwargs)
        self.base_url = base_url.rstrip("/")

    def complete(self, prompt: str, system: str | None = None, json_mode: bool = False, max_tokens: int = 1400) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        started = time.perf_counter()
        body = self._post(f"{self.base_url}/chat/completions", payload, {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            text = (body["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("bad_response", "missing choices") from exc
        if not text:
            raise LLMError("bad_response", "empty answer")
        usage = body.get("usage", {})
        self.last_model_used = body.get("model", self.model)
        return LLMResponse(text, "openai", self.last_model_used or self.model, {
            "input_tokens": usage.get("prompt_tokens"), "output_tokens": usage.get("completion_tokens"),
        }, int((time.perf_counter() - started) * 1000))

    def generate(self, prompt: str) -> str:
        return self.complete(prompt).text


def resolve_provider(app_settings: Settings) -> tuple[str, str]:
    """Return (provider, reason). provider is 'gemini', 'openai' or 'local'."""
    requested = (app_settings.llm_provider or "auto").strip().lower()
    if requested in {"off", "disabled", "none"}:
        return "local", "LLM calls disabled by LLM_PROVIDER."
    openai_ready = bool(app_settings.openai_model and (app_settings.openai_api_key or "localhost" in app_settings.openai_base_url or "127.0.0.1" in app_settings.openai_base_url))
    if requested == "openai":
        return ("openai", "") if openai_ready else ("local", "OPENAI_MODEL and OPENAI_API_KEY (or a local OPENAI_BASE_URL) are required.")
    if requested in {"gemini", "auto", "local", ""}:
        if app_settings.gemini_api_key:
            return "gemini", ""
        if requested == "auto" and openai_ready:
            return "openai", ""
        if requested == "gemini":
            return "local", "GEMINI_API_KEY is not configured."
    return "local", "No LLM provider key configured."


def build_llm(app_settings: Settings) -> tuple[_HTTPProvider | None, str]:
    provider, reason = resolve_provider(app_settings)
    if provider == "gemini":
        return GeminiClient(app_settings.gemini_api_key or "", app_settings.gemini_model, timeout_seconds=app_settings.llm_timeout_seconds), ""
    if provider == "openai":
        return OpenAICompatibleClient(
            app_settings.openai_api_key or "local", app_settings.openai_model, app_settings.openai_base_url, timeout_seconds=app_settings.llm_timeout_seconds
        ), ""
    return None, reason
