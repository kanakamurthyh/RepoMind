from __future__ import annotations

import hashlib
import logging
import mimetypes
from dataclasses import dataclass
from pathlib import Path

from app.models import FileRecord

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
    "target",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}

BINARY_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".mov",
    ".avi",
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".ico",
    ".woff",
    ".woff2",
    ".so",
    ".dylib",
    ".dll",
    ".exe",
}

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".md": "markdown",
    ".rst": "rst",
    ".txt": "text",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "shell",
}

CONFIG_NAMES = {
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "dockerfile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    "setup.py",
    "setup.cfg",
    "tsconfig.json",
}


@dataclass(frozen=True)
class IngestionPolicy:
    max_file_size_bytes: int
    max_files_per_repo: int = 15000
    max_total_repo_bytes: int = 250_000_000
    include_binary_metadata: bool = True
    follow_symlinks: bool = False
    read_block_size: int = 64 * 1024


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "unknown")


def detect_role(relative_path: str, language: str) -> str:
    lower = relative_path.lower()
    name = Path(lower).name
    if name in CONFIG_NAMES or language in {"json", "yaml", "toml", "ini"}:
        return "config"
    if name.startswith("readme") or "/docs/" in f"/{lower}/" or language in {"markdown", "rst"}:
        return "docs"
    if "test" in lower or lower.startswith("tests/"):
        return "test"
    if language in {"python", "javascript", "typescript", "java", "go", "rust", "ruby", "php"}:
        return "code"
    return "other"


class FileScanner:
    def __init__(
        self,
        max_file_size_bytes: int | None = None,
        policy: IngestionPolicy | None = None,
        **policy_overrides: int | bool,
    ) -> None:
        if policy is None:
            if max_file_size_bytes is None:
                raise ValueError("max_file_size_bytes or policy is required.")
            policy = IngestionPolicy(max_file_size_bytes=max_file_size_bytes)
        if policy_overrides:
            policy = IngestionPolicy(
                max_file_size_bytes=policy.max_file_size_bytes,
                max_files_per_repo=int(policy_overrides.get("max_files_per_repo", policy.max_files_per_repo)),
                max_total_repo_bytes=int(policy_overrides.get("max_total_repo_bytes", policy.max_total_repo_bytes)),
                include_binary_metadata=bool(policy_overrides.get("include_binary_metadata", policy.include_binary_metadata)),
                follow_symlinks=bool(policy_overrides.get("follow_symlinks", policy.follow_symlinks)),
                read_block_size=policy.read_block_size,
            )
        self.policy = policy
        self.max_file_size_bytes = policy.max_file_size_bytes

    def scan(self, repo_id: str, repo_path: Path) -> list[FileRecord]:
        records: list[FileRecord] = []
        repo_root = repo_path.resolve()
        file_count = 0
        total_bytes = 0
        for path in sorted(repo_path.rglob("*")):
            if path.is_symlink() and not self.policy.follow_symlinks:
                records.append(self._metadata_record(repo_id, repo_root, path, "symlink", ["Symlink skipped to prevent traversal."]))
                continue
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
                if repo_root != resolved and repo_root not in resolved.parents:
                    records.append(self._metadata_record(repo_id, repo_root, path, "unsafe_path", ["Resolved path escapes repository root."]))
                    continue
                relative_parts = path.relative_to(repo_path).parts
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unsafe path %s: %s", path, exc)
                continue
            if any(part in IGNORED_DIRS for part in relative_parts):
                continue

            file_count += 1
            if file_count > self.policy.max_files_per_repo:
                logger.warning("File scan stopped at max_files_per_repo=%d", self.policy.max_files_per_repo)
                break

            relative = path.relative_to(repo_path).as_posix()
            try:
                size = path.stat().st_size
            except OSError as exc:
                logger.warning("Could not stat %s: %s", relative, exc)
                continue
            total_bytes += size
            if total_bytes > self.policy.max_total_repo_bytes:
                logger.warning("File scan stopped at max_total_repo_bytes=%d", self.policy.max_total_repo_bytes)
                records.append(self._metadata_record(repo_id, repo_root, path, "repo_limit", ["Repository byte limit reached; remaining files were not scanned."]))
                break

            warnings: list[str] = []
            truncated = size > self.policy.max_file_size_bytes
            if path.suffix.lower() in BINARY_EXTENSIONS:
                if self.policy.include_binary_metadata:
                    records.append(self._metadata_record(repo_id, repo_root, path, "binary", ["Binary file metadata captured; content was not embedded."]))
                continue
            try:
                raw, digest = self._read_limited(path)
                if b"\x00" in raw[:2048]:
                    if self.policy.include_binary_metadata:
                        records.append(self._metadata_record(repo_id, repo_root, path, "binary", ["Null bytes detected; content was not embedded."], sha256=digest))
                    continue
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                if self.policy.include_binary_metadata:
                    records.append(self._metadata_record(repo_id, repo_root, path, "unsupported_text", ["Non-UTF8 file metadata captured; content was not embedded."]))
                continue
            except OSError as exc:
                logger.warning("Could not read %s: %s", relative, exc)
                continue
            if truncated:
                warnings.append(f"File truncated at {self.policy.max_file_size_bytes} bytes.")
            language = detect_language(path)
            role = detect_role(relative, language)
            records.append(
                FileRecord(
                    repo_id=repo_id,
                    path=relative,
                    abs_path=str(path),
                    language=language,
                    role=role,
                    size_bytes=size,
                    line_count=max(1, text.count("\n") + 1),
                    text=text,
                    truncated=truncated,
                    warnings=warnings,
                    kind="text",
                    mime_type=mimetypes.guess_type(relative)[0] or "text/plain",
                    sha256=digest,
                    metadata={"ingestion_policy": "text_limited_stream"},
                )
            )
        return records

    def _read_limited(self, path: Path) -> tuple[bytes, str]:
        digest = hashlib.sha256()
        remaining = self.policy.max_file_size_bytes
        chunks: list[bytes] = []
        with path.open("rb") as handle:
            while remaining > 0:
                block = handle.read(min(self.policy.read_block_size, remaining))
                if not block:
                    break
                digest.update(block)
                chunks.append(block)
                remaining -= len(block)
        return b"".join(chunks), digest.hexdigest()

    def _metadata_record(
        self,
        repo_id: str,
        repo_root: Path,
        path: Path,
        kind: str,
        warnings: list[str],
        sha256: str | None = None,
    ) -> FileRecord:
        try:
            relative = path.relative_to(repo_root).as_posix()
        except ValueError:
            relative = path.name
        size = path.stat().st_size if path.exists() and path.is_file() else 0
        language = detect_language(path)
        mime = mimetypes.guess_type(relative)[0] or "application/octet-stream"
        return FileRecord(
            repo_id=repo_id,
            path=relative,
            abs_path=str(path),
            language=language,
            role="binary" if kind in {"binary", "unsupported_text"} else "metadata",
            size_bytes=size,
            line_count=0,
            text="",
            truncated=False,
            warnings=warnings,
            kind=kind,
            mime_type=mime,
            sha256=sha256,
            metadata={"embedded": False},
        )
