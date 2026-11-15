"""Test isolation: no network, no real API keys from .env, artifacts in temp dirs."""

from __future__ import annotations

import os
import tempfile

# Must run before any `app` import: Settings() reads the environment at import time,
# and python-dotenv never overrides variables that are already set.
_DATA_DIR = tempfile.mkdtemp(prefix="repomind-tests-")
os.environ.update(
    {
        "REPOMIND_DATA_DIR": _DATA_DIR,
        "GEMINI_API_KEY": "",
        "OPENAI_API_KEY": "",
        "GITHUB_TOKEN": "",
        "OPENAI_MODEL": "",
        "REPOMIND_API_KEY": "",
        "LLM_PROVIDER": "off",
        "EMBEDDING_PROVIDER": "local",
        "VECTOR_STORE_PROVIDER": "local",
        "RERANKER_PROVIDER": "local",
        "OSV_SCAN_ENABLED": "false",
        "GITHUB_API_CHECKS_ENABLED": "false",
        "DATABASE_URL": "",
        "LOG_LEVEL": "WARNING",
    }
)

import pytest  # noqa: E402

from app.api.services import container  # noqa: E402


@pytest.fixture(autouse=True)
def reset_container():
    yield
    container._analyzer = None
    container._chat = None
    container._detective = None
    if container._jobs is not None:
        container._jobs.shutdown()
    container._jobs = None


@pytest.fixture(scope="session")
def shop_env(tmp_path_factory):
    """Analyzed ShopAPI fixture repo: (analyzer, chat_service, repo_id)."""
    from evals.run_eval import build_environment

    return build_environment(tmp_path_factory.mktemp("shop"))
