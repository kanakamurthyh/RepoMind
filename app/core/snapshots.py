"""Immutable analysis runs, published with one atomic pointer replacement.

The database remains a searchable metadata mirror. Published run files are the
read authority so a failed DB save or publication never changes the active run.
Context-local pins let an entire question use one run while a refresh publishes.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.models import AnalysisResult, Chunk, FileRecord, Weakness
from app.utils.paths import safe_join, validate_repo_id
from app.utils.redaction import redact_data

logger = logging.getLogger(__name__)
PIPELINE_VERSION = 5


def snapshot_read(method):
    """Pin service calls end-to-end; compatible with lightweight test analyzers."""
    @wraps(method)
    def wrapped(self, repo_id, *args, **kwargs):
        analyzer = getattr(self, "analyzer", self)
        context = analyzer.read_snapshot(repo_id) if hasattr(analyzer, "read_snapshot") else nullcontext()
        with context:
            return method(self, repo_id, *args, **kwargs)
    return wrapped


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._pins: ContextVar[dict[str, dict[str, Any] | None] | None] = ContextVar("analysis_snapshots", default=None)

    def _pinned(self) -> dict[str, dict[str, Any] | None]:
        return self._pins.get() or {}

    def _repo(self, repo_id: str) -> Path:
        return safe_join(self.root, validate_repo_id(repo_id))

    @contextmanager
    def writer_lock(self, repo_id: str) -> Iterator[None]:
        locks = self.root / ".locks"
        locks.mkdir(exist_ok=True)
        with (locks / f"{validate_repo_id(repo_id)}.lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def active(self, repo_id: str) -> dict[str, Any] | None:
        if repo_id in self._pinned():
            return self._pinned()[repo_id]
        path = self._repo(repo_id) / "active.json"
        if not path.exists():
            return None
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            run_id = manifest["run_id"]
            if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
                raise ValueError("invalid run id")
            if manifest["artifact_id"] != f"{repo_id}--{run_id}":
                raise ValueError("invalid artifact id")
            return manifest
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.error("Unreadable active snapshot for %s: %s", repo_id, exc)
            raise RuntimeError(f"Active analysis snapshot for {repo_id} is corrupt; rebuild it.") from exc

    @contextmanager
    def pin(self, repo_id: str) -> Iterator[dict[str, Any] | None]:
        validate_repo_id(repo_id)
        if repo_id in self._pinned():
            yield self._pinned()[repo_id]
            return
        active = self.active(repo_id)
        token = self._pins.set({**self._pinned(), repo_id: active})
        try:
            yield active
        finally:
            self._pins.reset(token)

    def artifact_id(self, repo_id: str) -> str:
        # Internal artifact keys and standalone vector-store test ids are not public repository ids.
        suffix = repo_id.rsplit("--", 1)[-1]
        if "--" in repo_id and len(suffix) == 32 and all(c in "0123456789abcdef" for c in suffix):
            return repo_id
        manifest = self.active(repo_id)
        return manifest["artifact_id"] if manifest else repo_id

    def begin(self, repo_id: str, commit_sha: str | None, fingerprint: str) -> dict[str, Any]:
        run_id = uuid4().hex
        manifest = {"run_id": run_id, "artifact_id": f"{repo_id}--{run_id}", "commit_sha": commit_sha,
                    "pipeline_version": PIPELINE_VERSION, "config_fingerprint": fingerprint}
        self.directory(repo_id, manifest).mkdir(parents=True, exist_ok=False)
        return manifest

    def directory(self, repo_id: str, manifest: dict[str, Any]) -> Path:
        return safe_join(self._repo(repo_id), manifest["run_id"])

    def historical(self, repo_id: str, run_id: str) -> dict[str, Any]:
        if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
            raise ValueError("Invalid run id")
        path = self._repo(repo_id) / run_id / "manifest.json"
        if not path.is_file():
            raise FileNotFoundError("Analysis snapshot not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def stage(self, result: AnalysisResult, records: list[FileRecord], manifest: dict[str, Any]) -> None:
        directory = self.directory(result.repo_id, manifest)
        payload = redact_data(result.to_dict(include_chunks=False))
        self._write(directory / "analysis.json", payload)
        # Source snapshots contain only the scanned text, with credentials redacted.
        # This avoids reading a refreshed, mutable checkout for old citations.
        sources = {}
        (directory / "sources").mkdir(exist_ok=True)
        for record in records:
            if record.kind != "text":
                continue
            filename = hashlib.sha256(record.path.encode()).hexdigest() + ".json"
            payload = {"text": redact_data(record.text), "truncated": record.truncated, "sha256": record.sha256}
            self._write(directory / "sources" / filename, payload)
            sources[record.path] = filename
        self._write(directory / "sources.json", sources)
        manifest["source_format"] = 2
        self._write(directory / "manifest.json", manifest)

    def publish(self, repo_id: str, manifest: dict[str, Any]) -> None:
        directory = self.directory(repo_id, manifest)
        for name in ("analysis.json", "sources.json", "manifest.json"):
            if not (directory / name).is_file():
                raise RuntimeError(f"Incomplete analysis run: {name} is missing")
        self._write(self._repo(repo_id) / "active.json", manifest)

    @staticmethod
    def _write(path: Path, payload: Any) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)

    def load(self, repo_id: str, manifest: dict[str, Any], include_chunks: bool = False) -> AnalysisResult:
        payload = redact_data(json.loads((self.directory(repo_id, manifest) / "analysis.json").read_text(encoding="utf-8")))
        return AnalysisResult(
            repo_id=payload["repo_id"], repo_url=payload["repo_url"], summary=payload["summary"],
            weaknesses=[Weakness(**{k: v for k, v in w.items() if k != "issue"}) for w in payload["weaknesses"]],
            graph=payload["graph"], chunks=[Chunk.from_dict(c) for c in payload.get("chunks", [])] if include_chunks else [],
            statuses=payload["statuses"], commit_sha=payload.get("commit_sha"),
        )

    def source(self, repo_id: str, manifest: dict[str, Any], relative_path: str) -> dict[str, Any] | None:
        directory = self.directory(repo_id, manifest)
        sources = json.loads((directory / "sources.json").read_text(encoding="utf-8"))
        entry = sources.get(relative_path)
        if entry is None:
            return None
        if manifest.get("source_format") != 2:
            return entry  # immutable runs created before per-file source storage
        path = safe_join(directory / "sources", entry)
        return json.loads(path.read_text(encoding="utf-8"))
