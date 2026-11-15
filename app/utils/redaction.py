"""Single source of truth for secret detection and redaction."""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import wraps
from typing import Any

PROVIDER_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{25,}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)
_SECRET_NAME = r"[\w-]*(?:api[_-]?key|secret|password|passwd|token|access[_-]?key|auth[_-]?token|private[_-]?key|client[_-]?secret)[\w-]*"
# Quoted assignment in code/config: API_KEY = "value" / "token": "value"
ASSIGNED_SECRET_RE = re.compile(rf"(?i)\b({_SECRET_NAME})['\"]?\s*[:=]\s*(['\"])([^'\"\s]{{8,}})\2")
# Unquoted env-file style: API_KEY=value (whole line)
ENV_SECRET_RE = re.compile(rf"(?i)^\s*(?:export\s+)?({_SECRET_NAME})=()([^\s'\"$#]{{8,}})\s*$")
SEQUENTIAL_RE = re.compile(r"abcdefgh|bcdefghi|12345678|23456789|qwertyui|(.)\1{5,}", re.IGNORECASE)
URL_KEY_PARAM_RE = re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token)=)[^&\s'\"]+")
PLACEHOLDER_MARKERS = (
    "example", "your_", "your-", "changeme", "placeholder", "dummy", "xxxx", "<", "${", "{{", "os.getenv",
    "os.environ", "process.env", "settings.", "config.", "none", "null", "test", "fake", "sample", "redacted",
)
REDACTED = "<redacted-secret>"


def shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts = Counter(value)
    length = len(value)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def looks_like_placeholder(value: str) -> bool:
    lowered = value.lower()
    return any(marker in lowered for marker in PLACEHOLDER_MARKERS) or bool(SEQUENTIAL_RE.search(value))


def find_secret(line: str) -> str | None:
    """Return the matched secret value if the line contains a high-confidence secret."""
    for pattern in PROVIDER_SECRET_PATTERNS:
        match = pattern.search(line)
        if match and (match.group(0).startswith("-----BEGIN") or not looks_like_placeholder(match.group(0))):
            return match.group(0)
    assigned = ASSIGNED_SECRET_RE.search(line) or ENV_SECRET_RE.search(line)
    if not assigned:
        return None
    value = assigned.group(3)
    if looks_like_placeholder(value):
        return None
    # Identifiers such as `password = user_password` are not secrets.
    if (re.fullmatch(r"[a-z_][a-z0-9_.]*", value) or re.fullmatch(r"[A-Za-z_.]+", value)) and shannon_entropy(value) < 3.5:
        return None
    if len(value) < 12 and shannon_entropy(value) < 3.0:
        return None
    return value


def redact(text: str) -> str:
    redacted = URL_KEY_PARAM_RE.sub(lambda m: m.group(1) + REDACTED, text)
    for pattern in PROVIDER_SECRET_PATTERNS:
        redacted = pattern.sub(REDACTED, redacted)

    def _assigned(match: re.Match[str]) -> str:
        value = match.group(3)
        if looks_like_placeholder(value):
            return match.group(0)
        return match.group(0).replace(value, REDACTED)

    redacted = ASSIGNED_SECRET_RE.sub(_assigned, redacted)
    return "\n".join(ENV_SECRET_RE.sub(_assigned, line) for line in redacted.split("\n"))


def scrub(text: str, *secrets: str | None) -> str:
    """Remove known secret values (e.g. configured API keys) and secret-like tokens."""
    cleaned = text
    for secret in secrets:
        if secret:
            cleaned = cleaned.replace(secret, REDACTED)
    return redact(cleaned)


def redact_data(value: Any) -> Any:
    """Copy a JSON-friendly payload and redact every text value at an output boundary."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: redact_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact_data(item) for item in value]
    return value


def redacted_output(method):
    """Redact synchronous connector results without hiding their tool signature."""
    @wraps(method)
    def wrapped(*args, **kwargs):
        return redact_data(method(*args, **kwargs))
    return wrapped
