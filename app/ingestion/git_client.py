from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from app.utils.paths import normalize_github_url, repo_id_from_url

logger = logging.getLogger(__name__)


class GitClient:
    """Clone public GitHub repositories into a local cache."""

    def __init__(self, repos_dir: Path) -> None:
        self.repos_dir = repos_dir
        self.repos_dir.mkdir(parents=True, exist_ok=True)

    def clone_or_get(self, repo_url: str) -> tuple[str, Path, bool]:
        normalized = normalize_github_url(repo_url)
        repo_id = repo_id_from_url(normalized)
        destination = self.repos_dir / repo_id
        if destination.exists() and (destination / ".git").exists():
            logger.info("Using cached repository %s", destination)
            return repo_id, destination, False

        if destination.exists():
            raise RuntimeError(f"Cache path exists but is not a git repository: {destination}")

        clone_url = f"{normalized}.git"
        logger.info("Cloning %s into %s", clone_url, destination)
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", clone_url, str(destination)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git clone failed: {exc.stderr.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("git clone timed out after 300 seconds") from exc
        return repo_id, destination, True
