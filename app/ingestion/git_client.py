from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests

from app.utils.paths import legacy_repo_id_from_url, normalize_github_url, repo_id_from_url

logger = logging.getLogger(__name__)


class GitError(RuntimeError):
    """Raised when a repository cannot be fetched."""


class RepoTooLargeError(GitError):
    """Raised when the GitHub API reports a repository above the configured size budget."""


def git_env() -> dict[str, str]:
    """Environment for every git subprocess: never prompt, never use the host's credential helpers."""
    env = dict(os.environ)
    # RepoMind only reads public repos. GITHUB_TOKEN is for the REST API only and must never reach git.
    env.pop("GITHUB_TOKEN", None)
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo", "GCM_INTERACTIVE": "never", "GIT_LFS_SKIP_SMUDGE": "1"})
    return env


HARDENING_FLAGS = ["-c", "credential.helper=", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]


def git_command(*args: str) -> list[str]:
    """A git argv with the hardening flags applied (use with `env=git_env()`)."""
    return ["git", *HARDENING_FLAGS, *args]


def directory_size_bytes(path: Path) -> int:
    """Apparent size of every regular file under `path` (symlinks are not followed)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                continue  # files vanish while git renames its temporary packs
    return total


class GitClient:
    """Clone public GitHub repositories into a local cache (hardened, size-checked)."""

    def __init__(
        self,
        repos_dir: Path,
        history_depth: int = 1,
        timeout_seconds: int = 300,
        max_repo_size_mb: int = 1024,
        github_api_checks: bool = True,
        github_token: str | None = None,
        size_poll_seconds: float = 1.0,
    ) -> None:
        self.repos_dir = repos_dir
        self.repos_dir.mkdir(parents=True, exist_ok=True)
        self.history_depth = max(1, history_depth)
        self.timeout_seconds = timeout_seconds
        self.max_repo_size_mb = max_repo_size_mb
        self.github_api_checks = github_api_checks
        # Falls back to the environment so callers that do not pass Settings.github_token still benefit.
        self._github_token = github_token if github_token is not None else (os.getenv("GITHUB_TOKEN") or "").strip() or None
        self.size_poll_seconds = size_poll_seconds

    def clone_or_get(self, repo_url: str, refresh: bool = False) -> tuple[str, Path, bool]:
        """Return (repo_id, path, fetched). `refresh=True` pulls the latest default-branch commit."""
        normalized = normalize_github_url(repo_url)
        repo_id = repo_id_from_url(normalized)
        destination = self.repos_dir / repo_id
        self._migrate_legacy_cache(repo_url, destination)

        if destination.exists() and (destination / ".git").exists():
            if refresh:
                self._refresh(destination)
                return repo_id, destination, True
            logger.info("Using cached repository %s", destination)
            return repo_id, destination, False

        if destination.exists():
            logger.warning("Removing incomplete cache directory %s", destination)
            shutil.rmtree(destination)

        self._check_size(normalized)
        clone_url = f"{normalized}.git"
        logger.info("Cloning %s into %s (depth=%d)", clone_url, destination, self.history_depth)
        self._clone(clone_url, destination)
        return repo_id, destination, True

    def path_for(self, repo_id: str) -> Path:
        path = (self.repos_dir / repo_id).resolve()
        if path.parent != self.repos_dir.resolve():
            raise ValueError("Repository id escapes the cache directory.")
        return path

    def head_sha(self, repo_path: Path) -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, timeout=15, env=git_env(), check=True
            )
        except (subprocess.SubprocessError, OSError) as exc:
            logger.warning("Could not read HEAD for %s: %s", repo_path, exc)
            return None
        return result.stdout.strip() or None

    def _refresh(self, destination: Path) -> None:
        logger.info("Refreshing cached repository %s", destination)
        self._budgeted(git_command("fetch", "--depth", str(self.history_depth), "--no-tags", "origin", "HEAD"), destination, destination, "git fetch")
        self._budgeted(git_command("reset", "--hard", "FETCH_HEAD"), destination, destination, "git reset")
        self._run(git_command("clean", "-fdx"), cwd=destination, action="git clean")

    @property
    def max_repo_bytes(self) -> int | None:
        return self.max_repo_size_mb * 1024 * 1024 if self.max_repo_size_mb > 0 else None

    def _clone(self, clone_url: str, destination: Path) -> None:
        """Shallow-clone while polling the on-disk size; kill and clean up past MAX_REPO_SIZE_MB or the timeout.

        This is the authoritative size limit: the GitHub API pre-check is advisory and may be unavailable.
        """
        command = git_command("clone", "--depth", str(self.history_depth), "--no-tags", "--single-branch", clone_url, str(destination))
        try:
            self._budgeted(command, None, destination, "git clone")
        except GitError:
            shutil.rmtree(destination, ignore_errors=True)
            raise

    def _budgeted(self, command: list[str], cwd: Path | None, destination: Path, action: str) -> None:
        """Enforce the same disk/time budgets for clone, fetch and checkout."""
        limit = self.max_repo_bytes
        deadline = time.monotonic() + self.timeout_seconds
        with tempfile.TemporaryFile(mode="w+") as stderr:
            try:
                process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.DEVNULL, stderr=stderr, text=True, env=git_env(), start_new_session=True)
            except OSError as exc:
                raise GitError(f"{action} failed: {exc}") from exc
            failure: GitError | None = None
            returncode: int | None = None
            while True:
                try:
                    returncode = process.wait(timeout=self.size_poll_seconds)
                    break
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic() > deadline:
                    failure = GitError(f"{action} timed out after {self.timeout_seconds} seconds")
                elif limit is not None and destination.exists() and directory_size_bytes(destination) > limit:
                    failure = RepoTooLargeError(f"Repository exceeded MAX_REPO_SIZE_MB={self.max_repo_size_mb} on disk while cloning.")
                if failure is not None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        logger.warning("Process-group termination is restricted; terminating the git parent for %s", action)
                        process.kill()
                    process.wait()
                    break
            if failure is None and returncode != 0:
                stderr.seek(0)
                failure = GitError(f"{action} failed: {stderr.read().strip()[:500]}")
        if failure is None and limit is not None and directory_size_bytes(destination) > limit:
            failure = RepoTooLargeError(f"Repository is above MAX_REPO_SIZE_MB={self.max_repo_size_mb} on disk.")
        if failure is not None:
            logger.warning("%s aborted for %s: %s", action, destination.name, failure)
            raise failure

    def _run(self, command: list[str], cwd: Path | None, action: str, cleanup: Path | None = None) -> None:
        try:
            subprocess.run(
                command, cwd=cwd, check=True, capture_output=True, text=True, timeout=self.timeout_seconds, env=git_env()
            )
        except subprocess.CalledProcessError as exc:
            if cleanup is not None and cleanup.exists():
                shutil.rmtree(cleanup, ignore_errors=True)
            raise GitError(f"{action} failed: {(exc.stderr or '').strip()[:500]}") from exc
        except subprocess.TimeoutExpired as exc:
            if cleanup is not None and cleanup.exists():
                shutil.rmtree(cleanup, ignore_errors=True)
            raise GitError(f"{action} timed out after {self.timeout_seconds} seconds") from exc

    def _check_size(self, normalized_url: str) -> int | None:
        """Reject repositories GitHub reports as oversized; return the size in KB, or None when unknown.

        An unknown size (rate limit, network error) is not fatal because `_clone` enforces the cap on disk.
        """
        if not self.github_api_checks:
            return None
        owner, repo = [p for p in urlparse(normalized_url).path.split("/") if p][:2]
        headers = {"Accept": "application/vnd.github+json"}
        if self._github_token:
            headers["Authorization"] = f"Bearer {self._github_token}"
        try:
            response = requests.get(f"https://api.github.com/repos/{owner}/{repo}", headers=headers, timeout=6)
        except requests.RequestException as exc:
            logger.warning("GitHub size pre-check unavailable for %s/%s (%s); size unknown, enforcing on-disk cap", owner, repo, type(exc).__name__)
            return None
        if response.status_code == 404:
            raise GitError("git clone failed: repository not found or not public")
        if not response.ok:
            reason = "rate limited" if response.status_code in {403, 429} else "error"
            logger.warning(
                "GitHub size pre-check %s for %s/%s (HTTP %s); size unknown, enforcing on-disk cap",
                reason, owner, repo, response.status_code,
            )
            return None
        try:
            size_kb = int(response.json().get("size") or 0)
        except (ValueError, TypeError, AttributeError) as exc:
            logger.warning("GitHub size pre-check returned an unexpected body for %s/%s: %s", owner, repo, exc)
            return None
        if self.max_repo_size_mb > 0 and size_kb > self.max_repo_size_mb * 1024:
            raise RepoTooLargeError(
                f"Repository is {size_kb // 1024} MB, above MAX_REPO_SIZE_MB={self.max_repo_size_mb}."
            )
        return size_kb

    def _migrate_legacy_cache(self, repo_url: str, destination: Path) -> None:
        """Rename a clone created before URL lowercasing. Exact id match only - never a prefix glob."""
        if destination.exists():
            return
        try:
            legacy = self.repos_dir / legacy_repo_id_from_url(repo_url)
        except ValueError:
            return
        candidates = [legacy] if legacy.is_dir() else []
        prefix = destination.name.rsplit("-", 1)[0] + "-"
        candidates += [path for path in sorted(self.repos_dir.glob(f"{prefix}*")) if path != legacy and len(path.name) == len(destination.name)]
        normalized = normalize_github_url(repo_url)
        for candidate in candidates:
            if candidate == destination or not (candidate / ".git").exists():
                continue
            # Exact verification: the clone's own remote URL must normalize to the requested repo.
            if self._remote_url(candidate) == normalized:
                logger.info("Migrating legacy cache %s -> %s", candidate.name, destination.name)
                candidate.rename(destination)
                return

    def _remote_url(self, path: Path) -> str | None:
        config = path / ".git" / "config"
        try:
            text = config.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None
        match = re.search(r'\[remote "origin"\][^\[]*?url\s*=\s*(\S+)', text, re.S)
        if not match:
            return None
        try:
            return normalize_github_url(match.group(1))
        except ValueError:
            return None

    def delete(self, repo_id: str) -> None:
        path = self.repos_dir / repo_id
        if path.resolve().parent != self.repos_dir.resolve():
            raise ValueError("Refusing to delete a path outside the repository cache.")
        if path.is_dir():
            shutil.rmtree(path)
