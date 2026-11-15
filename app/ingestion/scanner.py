from __future__ import annotations

import hashlib
import logging
import mimetypes
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.models import FileRecord

logger = logging.getLogger(__name__)

IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "bower_components",
    "venv",
    ".venv",
    "env",
    "__pycache__",
    "dist",
    "build",
    "target",
    "out",
    ".next",
    ".nuxt",
    ".turbo",
    ".cache",
    "coverage",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".idea",
    ".vscode",
    ".gradle",
    "vendor",
    "site-packages",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".ico", ".svgz", ".psd",
    ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".mkv", ".webm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".jar", ".war", ".whl", ".egg",
    ".woff", ".woff2", ".ttf", ".otf", ".eot",
    ".so", ".dylib", ".dll", ".exe", ".bin", ".o", ".a", ".class", ".pyc", ".wasm",
    ".pt", ".pth", ".onnx", ".h5", ".pkl", ".pickle", ".npy", ".npz", ".parquet", ".feather", ".db", ".sqlite",
}

LANGUAGE_BY_EXT = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".md": "markdown",
    ".mdx": "markdown",
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
    ".scss": "css",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".ps1": "powershell",
    ".ipynb": "notebook",
}

CODE_LANGUAGES = {
    "python", "javascript", "typescript", "java", "kotlin", "scala", "go", "rust", "ruby", "php",
    "c", "cpp", "csharp", "swift", "shell", "powershell", "sql",
}

CONFIG_NAMES = {
    "requirements.txt",
    "requirements-dev.txt",
    "pyproject.toml",
    "package.json",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "compose.yml",
    "compose.yaml",
    ".env.example",
    ".gitignore",
    "setup.py",
    "setup.cfg",
    "tsconfig.json",
    "makefile",
    "procfile",
    "go.mod",
    "cargo.toml",
    "pom.xml",
    "build.gradle",
}

# Machine-generated files: kept as metadata, never embedded.
LOCKFILE_NAMES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "poetry.lock",
    "pipfile.lock",
    "cargo.lock",
    "composer.lock",
    "gemfile.lock",
    "go.sum",
    "uv.lock",
    "bun.lockb",
}
MINIFIED_RE = re.compile(r"\.min\.(js|css)$|\.bundle\.js$|\.map$", re.IGNORECASE)
TEST_DIR_SEGMENTS = {"test", "tests", "__tests__", "spec", "specs", "testing", "e2e"}
TEST_FILE_RE = re.compile(r"(^test_.*|.*_test\.\w+$|.*\.(test|spec)\.\w+$|^conftest\.py$|.*Tests?\.(java|kt|cs)$)")
DOC_DIR_SEGMENTS = {"docs", "doc", "documentation"}
# How many unscanned paths to remember (and how far to keep counting) once a scan cap is hit.
SKIPPED_SAMPLE_SIZE = 20
MAX_SKIPPED_COUNT = 1_000_000


@dataclass(frozen=True)
class IngestionPolicy:
    max_file_size_bytes: int
    max_files_per_repo: int = 15000
    max_total_repo_bytes: int = 250_000_000
    include_binary_metadata: bool = True
    follow_symlinks: bool = False
    read_block_size: int = 64 * 1024


def detect_language(path: Path) -> str:
    name = path.name.lower()
    if name == "dockerfile" or name.startswith("dockerfile."):
        return "dockerfile"
    if name == "makefile":
        return "makefile"
    return LANGUAGE_BY_EXT.get(path.suffix.lower(), "unknown")


def is_generated(relative_path: str) -> bool:
    name = PurePosixPath(relative_path).name.lower()
    return name in LOCKFILE_NAMES or bool(MINIFIED_RE.search(name))


def detect_role(relative_path: str, language: str) -> str:
    """Classify a file by path segments (never by raw substrings such as 'test' in 'latest')."""
    posix = PurePosixPath(relative_path)
    name = posix.name.lower()
    dir_segments = {part.lower() for part in posix.parts[:-1]}
    if is_generated(relative_path):
        return "generated"
    if name in CONFIG_NAMES or language in {"json", "yaml", "toml", "ini", "dockerfile", "makefile"}:
        return "config"
    if name.startswith("readme") or language in {"markdown", "rst"} or dir_segments & DOC_DIR_SEGMENTS and language in {"text", "html"}:
        return "docs"
    if dir_segments & TEST_DIR_SEGMENTS or TEST_FILE_RE.match(posix.name):
        return "test"
    if language in CODE_LANGUAGES:
        return "code"
    return "other"


def decode_text(raw: bytes, truncated: bool) -> str | None:
    """Decode UTF-8 text; a truncation cut in the middle of a character is tolerated."""
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Only the final (at most 3) bytes may belong to a cut multi-byte character.
        if truncated and exc.start >= len(raw) - 3:
            return raw[: exc.start].decode("utf-8", errors="strict")
        return None


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
        self.last_scan_stats: dict[str, Any] = {}

    def _walk(self, repo_path: Path) -> Iterator[Path]:
        """Yield files in deterministic order, pruning ignored directories before descending."""
        for root, dirs, files in os.walk(repo_path, followlinks=self.policy.follow_symlinks):
            root_path = Path(root)
            kept_dirs = []
            for directory in sorted(dirs):
                candidate = root_path / directory
                if directory in IGNORED_DIRS or directory.endswith(".egg-info"):
                    continue
                if candidate.is_symlink() and not self.policy.follow_symlinks:
                    yield candidate
                    continue
                kept_dirs.append(directory)
            dirs[:] = kept_dirs
            for name in sorted(files):
                yield root_path / name

    def scan(self, repo_id: str, repo_path: Path) -> list[FileRecord]:
        """Scan a checkout into FileRecords.

        Scan caps (file count, total bytes) never fail silently: when one is hit the
        remaining files are still counted (not read) and ``last_scan_stats`` records
        ``stop_reason``, ``files_skipped_by_cap`` and a sample of the skipped paths.
        """
        records: list[FileRecord] = []
        repo_root = repo_path.resolve()
        file_count = 0
        total_bytes = 0
        stats: dict[str, Any] = {
            "files_seen": 0, "files_scanned": 0, "text_files": 0, "binary_files": 0, "skipped": 0, "truncated": 0,
            "limit_reached": False, "stop_reason": None, "files_skipped_by_cap": 0, "skipped_by_cap_exact": True,
            "skipped_by_cap_sample": [],
        }
        walker = self._walk(repo_path)
        for path in walker:
            if path.is_symlink() and not self.policy.follow_symlinks:
                records.append(self._metadata_record(repo_id, repo_path, path, "symlink", ["Symlink skipped to prevent traversal."]))
                continue
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
                if repo_root != resolved and repo_root not in resolved.parents:
                    records.append(self._metadata_record(repo_id, repo_path, path, "unsafe_path", ["Resolved path escapes repository root."]))
                    continue
            except (OSError, ValueError) as exc:
                logger.warning("Skipping unsafe path %s: %s", path, exc)
                stats["skipped"] += 1
                continue

            relative = path.relative_to(repo_path).as_posix()
            if file_count >= self.policy.max_files_per_repo:
                self._record_cap_hit(stats, "max_files_per_repo", relative, walker, repo_path, repo_id)
                break
            file_count += 1
            try:
                size = path.stat().st_size
            except OSError as exc:
                logger.warning("Could not stat %s in %s: %s", relative, repo_id, exc)
                stats["skipped"] += 1
                continue

            if path.suffix.lower() in BINARY_EXTENSIONS:
                stats["binary_files"] += 1
                if self.policy.include_binary_metadata:
                    records.append(self._metadata_record(repo_id, repo_path, path, "binary", ["Binary file metadata captured; content was not embedded."]))
                continue

            read_bytes = min(size, self.policy.max_file_size_bytes)
            if total_bytes + read_bytes > self.policy.max_total_repo_bytes:
                file_count -= 1
                records.append(self._metadata_record(repo_id, repo_path, path, "repo_limit", ["Repository byte limit reached; remaining files were not scanned."]))
                self._record_cap_hit(stats, "max_total_repo_bytes", relative, walker, repo_path, repo_id)
                break
            total_bytes += read_bytes

            truncated = size > self.policy.max_file_size_bytes
            try:
                raw, digest = self._read_limited(path)
            except OSError as exc:
                logger.warning("Could not read %s in %s: %s", relative, repo_id, exc)
                stats["skipped"] += 1
                continue
            if b"\x00" in raw[:4096]:
                stats["binary_files"] += 1
                if self.policy.include_binary_metadata:
                    records.append(self._metadata_record(repo_id, repo_path, path, "binary", ["Null bytes detected; content was not embedded."], sha256=digest))
                continue
            text = decode_text(raw, truncated)
            if text is None:
                stats["skipped"] += 1
                if self.policy.include_binary_metadata:
                    records.append(self._metadata_record(repo_id, repo_path, path, "unsupported_text", ["Non-UTF8 file metadata captured; content was not embedded."], sha256=digest))
                continue

            warnings: list[str] = []
            if truncated:
                stats["truncated"] += 1
                warnings.append(f"File truncated at {self.policy.max_file_size_bytes} bytes.")
            language = detect_language(path)
            role = detect_role(relative, language)
            stats["text_files"] += 1
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
                    metadata={"ingestion_policy": "text_limited_stream", "generated": role == "generated"},
                )
            )
        stats["files_scanned"] = file_count
        stats["files_seen"] = file_count + stats["files_skipped_by_cap"]
        self.last_scan_stats = stats
        return records

    def _record_cap_hit(
        self, stats: dict[str, Any], reason: str, first_relative: str, walker: Iterator[Path], repo_path: Path, repo_id: str
    ) -> None:
        """Count (without reading) every file the cap prevents us from scanning."""
        sample = [first_relative]
        skipped = 1
        exact = True
        for path in walker:
            if path.is_symlink() and not self.policy.follow_symlinks:
                continue
            if not path.is_file():
                continue
            skipped += 1
            if len(sample) < SKIPPED_SAMPLE_SIZE:
                sample.append(path.relative_to(repo_path).as_posix())
            if skipped >= MAX_SKIPPED_COUNT:
                exact = False
                break
        stats.update(
            limit_reached=True, stop_reason=reason, files_skipped_by_cap=skipped,
            skipped_by_cap_exact=exact, skipped_by_cap_sample=sample,
        )
        limit = self.policy.max_files_per_repo if reason == "max_files_per_repo" else self.policy.max_total_repo_bytes
        logger.warning(
            "File scan for %s stopped at %s=%d; %s%d file(s) were not scanned (first: %s)",
            repo_id, reason, limit, "" if exact else "at least ", skipped, ", ".join(sample[:3]),
        )

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
        try:
            size = path.lstat().st_size if kind == "symlink" else (path.stat().st_size if path.is_file() else 0)
        except OSError:
            size = 0
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
