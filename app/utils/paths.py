from __future__ import annotations

import hashlib
import re
from pathlib import Path
from urllib.parse import urlparse


def normalize_github_url(url: str) -> str:
    url = url.strip()
    if url.endswith(".git"):
        url = url[:-4]
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.netloc.lower() != "github.com":
        raise ValueError("Only public https://github.com/{owner}/{repo} URLs are supported.")
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 2:
        raise ValueError("GitHub URL must include owner and repository name.")
    owner, repo = parts[0], parts[1]
    if not re.fullmatch(r"[\w.-]+", owner) or not re.fullmatch(r"[\w.-]+", repo):
        raise ValueError("GitHub owner or repository name contains unsupported characters.")
    return f"https://github.com/{owner}/{repo}"


def repo_id_from_url(url: str) -> str:
    normalized = normalize_github_url(url)
    parsed = urlparse(normalized)
    owner, repo = [p for p in parsed.path.split("/") if p][:2]
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{owner}-{repo}-{digest}".lower()


def safe_join(base: Path, relative: str) -> Path:
    candidate = (base / relative).resolve()
    if base.resolve() not in candidate.parents and candidate != base.resolve():
        raise ValueError("Path escapes base directory.")
    return candidate
