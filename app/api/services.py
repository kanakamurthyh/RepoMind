"""Service container and background job runner (created lazily, overridable in tests)."""

from __future__ import annotations

import fcntl
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.agent.detective import RepoDetective
from app.config import Settings, settings
from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.db.repository import MetadataStoreUnavailableError
from app.ingestion.git_client import GitError, RepoTooLargeError
from app.utils.paths import repo_id_from_url

logger = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    pass


def error_code_for(exc: Exception) -> tuple[str, str]:
    """Map an exception to (code, user-safe message)."""
    if isinstance(exc, QueueFullError):
        return "queue_full", "Analysis queue is full; retry after an active job completes."
    if isinstance(exc, RepoTooLargeError):
        return "repo_too_large", str(exc)
    if isinstance(exc, GitError):
        message = str(exc)
        if "not found" in message.lower() or "repository not found" in message.lower():
            return "repo_not_found", "Repository not found or not public."
        if "timed out" in message.lower():
            return "clone_timeout", "Cloning the repository timed out."
        return "clone_failed", "Could not clone the repository. Check the URL and network access, or use a cached repository."
    if isinstance(exc, MetadataStoreUnavailableError):
        return "storage_unavailable", "The metadata database is unavailable; see server logs."
    if isinstance(exc, ValueError):
        return "invalid_input", str(exc)
    return "internal_error", "Analysis failed unexpectedly; see server logs."


class JobRunner:
    def __init__(self, analyzer: RepoAnalyzer, workers: int = 2, capacity: int = 8) -> None:
        self.analyzer = analyzer
        self.executor = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="repomind-analysis")
        self._active: dict[str, str] = {}
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, workers, capacity))

    def submit(self, repo_url: str, force: bool = False) -> dict[str, Any]:
        repo_id = repo_id_from_url(repo_url)
        with self._lock:
            active = self._active.get(repo_id)
            if active:
                job = self.analyzer.metadata.get_job(active)
                if job and job["state"] in {"queued", "running"}:
                    return job
            if not self._slots.acquire(blocking=False):
                raise QueueFullError()
            try:
                job = self.analyzer.metadata.create_job(repo_id, repo_url)
            except Exception:
                self._slots.release()
                raise
            self._active[repo_id] = job["job_id"]
        try:
            future = self.executor.submit(self._run, job["job_id"], repo_id, repo_url, force)
        except Exception:
            with self._lock:
                self._active.pop(repo_id, None)
            self._slots.release()
            self.analyzer.metadata.update_job(job["job_id"], state="failed", error="Worker submission failed", error_code="worker_unavailable")
            raise
        def finished(future):
            try:
                if future.cancelled():
                    self.analyzer.metadata.update_job(job["job_id"], state="failed", error="Server shut down before the job started.", error_code="interrupted")
                    with self._lock:
                        self._active.pop(repo_id, None)
            finally:
                self._slots.release()
        future.add_done_callback(finished)
        return job

    def _run(self, job_id: str, repo_id: str, repo_url: str, force: bool) -> None:
        metadata = self.analyzer.metadata

        def progress(stage: str, percent: int, statuses: list[str]) -> None:
            metadata.update_job(job_id, stage=stage, progress=percent, statuses=statuses)

        try:
            metadata.update_job(job_id, state="running", stage="starting", progress=1)
            result = self.analyzer.analyze(repo_url, force=force, progress=progress)
            metadata.update_job(job_id, state="completed", stage="completed", progress=100, statuses=result.statuses)
        except Exception as exc:
            code, message = error_code_for(exc)
            logger.exception("Analysis job %s for %s failed (%s)", job_id, repo_id, code)
            metadata.update_job(job_id, state="failed", error=message, error_code=code)
        finally:
            with self._lock:
                if self._active.get(repo_id) == job_id:
                    self._active.pop(repo_id, None)

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)


class ServiceContainer:
    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self._analyzer: Any = None
        self._chat: Any = None
        self._detective: Any = None
        self._jobs: JobRunner | None = None
        self._lock = threading.RLock()
        self._server_lock: Any = None

    @property
    def analyzer(self) -> Any:
        with self._lock:
            if self._analyzer is None:
                self._analyzer = RepoAnalyzer(self.settings)
            return self._analyzer

    @analyzer.setter
    def analyzer(self, value: Any) -> None:
        self._analyzer, self._chat, self._detective, self._jobs = value, None, None, None

    @property
    def chat(self) -> Any:
        with self._lock:
            if self._chat is None:
                self._chat = ChatService(self.analyzer, self.settings)
            return self._chat

    @chat.setter
    def chat(self, value: Any) -> None:
        self._chat, self._detective = value, None

    @property
    def detective(self) -> RepoDetective:
        with self._lock:
            if self._detective is None:
                self._detective = RepoDetective(self.analyzer, self.chat)
            return self._detective

    @property
    def jobs(self) -> JobRunner:
        with self._lock:
            if self._jobs is None:
                self._jobs = JobRunner(self.analyzer, self.settings.analysis_workers, self.settings.analysis_queue_capacity)
            return self._jobs

    def startup(self) -> None:
        if self._server_lock is not None:
            return
        # Until jobs have distributed worker leases, reject a second API worker
        # instead of letting its startup mark another worker's jobs interrupted.
        self.settings.ensure_dirs()
        self._server_lock = (self.settings.data_dir / ".api-worker.lock").open("a")
        try:
            fcntl.flock(self._server_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._server_lock.close()
            self._server_lock = None
            raise RuntimeError("This data directory already has an API worker. Run one worker; distributed jobs are not configured.") from exc
        try:
            interrupted = self.analyzer.metadata.mark_interrupted_jobs()
            if interrupted:
                logger.warning("Marked %d interrupted analysis job(s) as failed", interrupted)
        except MetadataStoreUnavailableError as exc:
            logger.error("Metadata store unavailable at startup: %s", exc)
        except Exception as exc:
            logger.warning("Could not reconcile interrupted jobs: %s", exc)

    def shutdown(self) -> None:
        if self._jobs is not None:
            self._jobs.shutdown()
        if self._analyzer is not None and hasattr(self._analyzer, "close"):
            self._analyzer.close()
        if self._server_lock is not None:
            self._server_lock.close()
            self._server_lock = None


container = ServiceContainer()


def get_container() -> ServiceContainer:
    return container
