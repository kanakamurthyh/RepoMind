from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse

# owner-repo-<10 hex chars>. Owner/repo segments are GitHub-safe characters; the
# pattern deliberately rejects "." / ".." and anything containing a path separator.
REPO_ID_PATTERN = r"^[a-z0-9_][a-z0-9_.-]{0,200}-[0-9a-f]{10}$"
REPO_ID_RE = re.compile(REPO_ID_PATTERN)
_GITHUB_SEGMENT_RE = re.compile(r"[\w.-]+")


def _github_parts(url: str) -> tuple[str, str]:
    url = url.strip()
    if url.endswith("/"):
        url = url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        raise ValueError("Only public https://github.com/{owner}/{repo} URLs are supported.")
    if parsed.params or parsed.username or parsed.password or parsed.port:
        raise ValueError("GitHub URL contains unsupported components.")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository name.")
    owner, repo = parts[0], parts[1]
    if not _GITHUB_SEGMENT_RE.fullmatch(owner) or not _GITHUB_SEGMENT_RE.fullmatch(repo):
        raise ValueError("GitHub owner or repository name contains unsupported characters.")
    if owner in {".", ".."} or repo in {".", ".."}:
        raise ValueError("GitHub owner or repository name is invalid.")
    return owner, repo


def normalize_github_url(url: str) -> str:
    owner, repo = _github_parts(url)
    return f"https://github.com/{owner.lower()}/{repo.lower()}"


def repo_id_from_url(url: str) -> str:
    normalized = normalize_github_url(url)
    owner, repo = [p for p in urlparse(normalized).path.split("/") if p][:2]
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{owner}-{repo}-{digest}".lower()


def legacy_repo_id_from_url(url: str) -> str:
    """Repo id produced before URLs were lowercased (hash of the original-case URL).

    Used only to migrate cache directories created by older versions; it is an exact
    lookup, never a prefix match.
    """
    owner, repo = _github_parts(url)
    original = f"https://github.com/{owner}/{repo}"
    digest = hashlib.sha1(original.encode("utf-8")).hexdigest()[:10]
    return f"{owner}-{repo}-{digest}".lower()


def is_valid_repo_id(repo_id: str) -> bool:
    return bool(REPO_ID_RE.fullmatch(repo_id or "")) and ".." not in repo_id


def validate_repo_id(repo_id: str) -> str:
    if not is_valid_repo_id(repo_id):
        raise ValueError(f"Invalid repository id: {repo_id!r}")
    return repo_id


def safe_join(base: Path, relative: str) -> Path:
    base_resolved = base.resolve()
    candidate = (base / relative).resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise ValueError("Path escapes base directory.")
    return candidate
