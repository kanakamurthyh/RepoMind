from __future__ import annotations

import pytest

from app.utils.redaction import find_secret, redact, scrub


@pytest.mark.parametrize(
    "line,expected",
    [
        ("API_KEY='Zr8Qx2LmP0vTq9Wd'", True),
        ("OPENAI_API_KEY=sk-Q7vK2pX9mL4tR8wZ1nB5cY3dF6hJ0sA", True),
        ('DB_PASSWORD: "hunter2hunter2X9"', True),
        ("-----BEGIN RSA PRIVATE KEY-----", True),
        ('GITHUB_TOKEN_RE = re.compile(r"x")', False),
        ("secret_match = self._secret_match(stripped)", False),
        ("token = os.getenv('TOKEN')", False),
        ("API_KEY='your_key_here'", False),
        ("AKIAIOSFODNN7EXAMPLE", False),
        ("key = 'sk-abcdefghijklmnopqrstuvwxyz123456'", False),
    ],
)
def test_secret_detection(line, expected):
    assert bool(find_secret(line)) is expected


def test_redact_and_scrub():
    assert "Zr8Qx2LmP0vTq9Wd" not in redact("API_KEY='Zr8Qx2LmP0vTq9Wd'")
    assert "abc123" not in redact("https://x/y?key=abc123&z=1")
    assert "my-live-key" not in scrub("request to host failed using my-live-key", "my-live-key")
