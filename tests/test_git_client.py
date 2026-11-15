from __future__ import annotations

import logging
import os
import subprocess

import pytest
import requests

from app.analysis.history import HistoryAnalyzer
from app.ingestion import git_client as git_module
from app.ingestion.git_client import GitClient, GitError, RepoTooLargeError
from app.utils.paths import legacy_repo_id_from_url, repo_id_from_url


def fake_clone(root, name, url):
    git_dir = root / name / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "config").write_text(f'[core]\n\tbare = false\n[remote "origin"]\n\turl = {url}\n\tfetch = +refs/heads/*:refs/remotes/origin/*\n')
    return root / name


def test_legacy_cache_is_migrated_only_when_remote_url_matches(tmp_path):
    client = GitClient(tmp_path, github_api_checks=False)
    url = "https://github.com/Owner/Repo"
    legacy = fake_clone(tmp_path, legacy_repo_id_from_url(url), "https://github.com/Owner/Repo.git")
    destination = tmp_path / repo_id_from_url(url)

    client._migrate_legacy_cache(url, destination)

    assert destination.exists() and not legacy.exists()


def test_similarly_named_repo_is_never_used_as_cache(tmp_path):
    client = GitClient(tmp_path, github_api_checks=False)
    fake_clone(tmp_path, "facebook-react-native-0123456789", "https://github.com/facebook/react-native.git")
    destination = tmp_path / repo_id_from_url("https://github.com/facebook/react")

    client._migrate_legacy_cache("https://github.com/facebook/react", destination)

    assert not destination.exists()
    assert (tmp_path / "facebook-react-native-0123456789").exists()


# --- size limits (local repos only, no network) ------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._body = body or {}

    def json(self):
        return self._body


def make_source_repo(path, payload_bytes):
    path.mkdir()

    def git(*args):
        subprocess.run(["git", "-c", "user.name=T", "-c", "user.email=t@example.com", *args], cwd=path, check=True, capture_output=True)

    git("init", "-q")
    (path / "blob.bin").write_bytes(os.urandom(payload_bytes))  # incompressible, so the clone really is this big
    git("add", ".")
    git("commit", "-qm", "init")
    return path


@pytest.fixture
def local_clones(monkeypatch):
    # Local clones need the file protocol, which the production hardening (correctly) forbids.
    monkeypatch.setattr(git_module, "HARDENING_FLAGS", ["-c", "credential.helper=", "-c", "core.hooksPath=/dev/null"])


@pytest.mark.parametrize("status", [403, 429, 500])
def test_size_precheck_treats_api_errors_as_unknown(tmp_path, monkeypatch, caplog, status):
    monkeypatch.setattr(git_module.requests, "get", lambda *a, **k: FakeResponse(status))
    client = GitClient(tmp_path, github_api_checks=True, github_token="")

    with caplog.at_level(logging.WARNING):
        assert client._check_size("https://github.com/owner/repo") is None
    assert "size unknown" in caplog.text


def test_size_precheck_network_error_is_unknown_and_404_is_fatal(tmp_path, monkeypatch):
    client = GitClient(tmp_path, github_api_checks=True, github_token="")

    def boom(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(git_module.requests, "get", boom)
    assert client._check_size("https://github.com/owner/repo") is None
    monkeypatch.setattr(git_module.requests, "get", lambda *a, **k: FakeResponse(404))
    with pytest.raises(GitError):
        client._check_size("https://github.com/owner/repo")


def test_size_precheck_sends_token_without_logging_it(tmp_path, monkeypatch, caplog):
    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["headers"] = headers
        return FakeResponse(200, {"size": 3 * 1024})

    monkeypatch.setattr(git_module.requests, "get", fake_get)
    client = GitClient(tmp_path, github_api_checks=True, max_repo_size_mb=2, github_token="ghp_test_token_value")

    with caplog.at_level(logging.DEBUG), pytest.raises(RepoTooLargeError):
        client._check_size("https://github.com/owner/repo")
    assert captured["headers"]["Authorization"] == "Bearer ghp_test_token_value"
    assert "ghp_test_token_value" not in caplog.text and "ghp_test_token_value" not in repr(client)


def test_clone_within_cap_succeeds(tmp_path, local_clones):
    source = make_source_repo(tmp_path / "src", 64 * 1024)
    client = GitClient(tmp_path / "cache", max_repo_size_mb=5, github_api_checks=False, size_poll_seconds=0.05)

    client._clone(str(source), tmp_path / "cache" / "dest")

    assert (tmp_path / "cache" / "dest" / "blob.bin").exists()


def test_clone_over_disk_cap_is_aborted_and_cleaned_up(tmp_path, local_clones):
    source = make_source_repo(tmp_path / "src", 3 * 1024 * 1024)
    destination = tmp_path / "cache" / "dest"
    client = GitClient(tmp_path / "cache", max_repo_size_mb=1, github_api_checks=False, size_poll_seconds=0.05)

    with pytest.raises(RepoTooLargeError):
        client._clone(str(source), destination)
    assert not destination.exists()


def test_clone_failure_is_reported_and_cleaned_up(tmp_path, local_clones):
    destination = tmp_path / "cache" / "dest"
    client = GitClient(tmp_path / "cache", github_api_checks=False, size_poll_seconds=0.05)

    with pytest.raises(GitError, match="git clone failed"):
        client._clone(str(tmp_path / "missing"), destination)
    assert not destination.exists()


def test_history_uses_hardened_git_env(tmp_path, monkeypatch):
    calls = []
    real_run = subprocess.run

    def spy(command, **kwargs):
        calls.append((command, kwargs.get("env")))
        return real_run(command, **kwargs)

    repo = make_source_repo(tmp_path / "src", 16)
    monkeypatch.setattr("app.analysis.history.subprocess.run", spy)
    HistoryAnalyzer().analyze(repo)

    assert calls and all(command[:3] == ["git", "-c", "credential.helper="] for command, _ in calls)
    assert all(env and env["GIT_TERMINAL_PROMPT"] == "0" for _, env in calls)
