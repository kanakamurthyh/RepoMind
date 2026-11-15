from __future__ import annotations

import hashlib
import json
import logging
import shutil
import threading
from collections import defaultdict
from collections.abc import Callable
from copy import copy
from pathlib import Path
from typing import Any

from app.analysis.dependencies import DependencyScanner
from app.analysis.drift import DocsDriftDetector
from app.analysis.history import HistoryAnalyzer
from app.analysis.repo_facts import RepoFactsCollector
from app.analysis.summary import SummaryGenerator
from app.analysis.weaknesses import WeaknessDetector
from app.config import Settings, settings
from app.core.snapshots import PIPELINE_VERSION, SnapshotStore
from app.db.repository import MetadataRepository
from app.graph.code_graph import CodeGraphBuilder
from app.graph.graph_store import GraphIndex, GraphStore
from app.ingestion.coverage import build_coverage, coverage_message
from app.ingestion.git_client import GitClient
from app.ingestion.scanner import FileScanner, IngestionPolicy
from app.models import AnalysisResult, Chunk, FileRecord, Weakness
from app.parsing.chunker import Chunker
from app.parsing.multilang_parser import MultiLanguageParser
from app.rag.vector_store import VectorStore
from app.utils.paths import normalize_github_url, repo_id_from_url, safe_join, validate_repo_id
from app.utils.redaction import redact, redact_data

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, int, list[str]], None]
STAGE_PROGRESS = {
    "fetching": 5, "scanning": 15, "parsing": 28, "chunking": 38, "indexing": 58, "graph": 68, "facts": 74,
    "history": 79, "dependencies": 84, "drift": 88, "weaknesses": 92, "summary": 96, "saving": 99, "completed": 100,
}


def _bool_setting(app_settings: Any, name: str, default: bool) -> bool:
    value = getattr(app_settings, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def metadata_fallback_allowed(app_settings: Any) -> bool:
    """Whether a broken metadata DB may degrade to JSON files (setting ``metadata_db_fallback``).

    ``auto`` (the default) allows the fallback only for the implicit local SQLite file; when
    ``DATABASE_URL`` is set explicitly (e.g. Postgres) a DB failure fails fast instead of
    silently writing analyses somewhere the configured database never sees.
    """
    value = getattr(app_settings, "metadata_db_fallback", "auto")
    if isinstance(value, str) and value.strip().lower() in {"", "auto"}:
        return not getattr(app_settings, "database_url", "")
    return _bool_setting(app_settings, "metadata_db_fallback", True)


class RepoAnalyzer:
    def __init__(
        self,
        app_settings: Settings = settings,
        vector_store: VectorStore | None = None,
        dependency_scanner: DependencyScanner | None = None,
        git_client: GitClient | None = None,
    ) -> None:
        self.settings = app_settings
        self.settings.ensure_dirs()
        self.git = git_client or GitClient(
            self.settings.repos_dir,
            history_depth=self.settings.git_history_depth,
            timeout_seconds=self.settings.clone_timeout_seconds,
            max_repo_size_mb=self.settings.max_repo_size_mb,
            github_api_checks=self.settings.github_api_checks_enabled,
            github_token=self.settings.github_token,
        )
        self.scanner = FileScanner(
            policy=IngestionPolicy(
                max_file_size_bytes=self.settings.max_file_size_bytes,
                max_files_per_repo=self.settings.max_files_per_repo,
                max_total_repo_bytes=self.settings.max_total_repo_bytes,
                include_binary_metadata=self.settings.binary_metadata_enabled,
                follow_symlinks=self.settings.follow_symlinks,
            )
        )
        self.parser = MultiLanguageParser(prefer_tree_sitter=self.settings.parser_provider.lower() in {"tree_sitter", "tree-sitter", "auto"})
        self.chunker = Chunker(collapse_duplicates=self.settings.collapse_duplicate_files)
        self.graph_builder = CodeGraphBuilder(self.parser)
        self.graph_store = GraphStore(self.settings.graphs_dir)
        self.facts_collector = RepoFactsCollector()
        self.weakness_detector = WeaknessDetector()
        self.drift_detector = DocsDriftDetector()
        self.dependency_scanner = dependency_scanner or DependencyScanner(enabled=self.settings.osv_scan_enabled)
        self.history_analyzer = HistoryAnalyzer()
        self.summary_generator = SummaryGenerator()
        self.vector_store = vector_store or VectorStore(self.settings.indexes_dir, app_settings=self.settings)
        self.metadata = MetadataRepository(
            self.settings.resolved_database_url,
            self.settings.metadata_dir,
            enabled=self.settings.use_metadata_db,
            allow_json_fallback=metadata_fallback_allowed(self.settings),
        )
        self.snapshots = SnapshotStore(self.settings.data_dir / "runs")
        self.vector_store.resolve_id = self.snapshots.artifact_id
        self.graph_store.resolve_id = self.snapshots.artifact_id
        self._locks: dict[str, threading.RLock] = defaultdict(threading.RLock)
        self._locks_guard = threading.Lock()

    def _lock_for(self, repo_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks[repo_id]

    # --- pipeline --------------------------------------------------------------------------
    def analyze(self, repo_url: str, force: bool = False, progress: ProgressCallback | None = None) -> AnalysisResult:
        repo_id = repo_id_from_url(repo_url)
        with self._lock_for(repo_id), self.snapshots.writer_lock(repo_id):
            directory = self.snapshots._repo(repo_id)
            previous = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
            try:
                return self._analyze(repo_url, force, progress)
            except Exception:
                active = self.snapshots.active(repo_id)
                if directory.exists():
                    for path in directory.iterdir():
                        if path.is_dir() and path.name not in previous and (not active or active["run_id"] != path.name):
                            key = f"{repo_id}--{path.name}"
                            try:
                                self.vector_store.delete(key)
                                self.graph_store.delete(key)
                                shutil.rmtree(path)
                            except Exception:
                                logger.exception("Could not clean unpublished analysis run %s", key)
                raise

    def _analyze(self, repo_url: str, force: bool = False, progress: ProgressCallback | None = None) -> AnalysisResult:
        repo_id = repo_id_from_url(repo_url)
        statuses: list[str] = []

        def stage(name: str, message: str) -> None:
            statuses.append(message)
            logger.info("[%s] %s", repo_id, message)
            if progress:
                progress(name, STAGE_PROGRESS.get(name, 0), list(statuses))

        with self._lock_for(repo_id):
            if not force and self.cache_ready(repo_id):
                stage("completed", "loaded cached analysis")
                return self.load(repo_id, statuses=statuses)
            if not force and self.metadata.exists(repo_id):
                problem = self.vector_store.compatibility_problem(repo_id)
                stage("fetching", f"cached analysis is stale ({problem or 'graph missing'}); rebuilding")

            stage("fetching", "refreshing repository" if force else "fetching repository")
            repo_id, repo_path, fetched = self.git.clone_or_get(repo_url, refresh=force)
            commit_sha = self.git.head_sha(repo_path)
            run = self.snapshots.begin(repo_id, commit_sha, self._fingerprint())
            artifact_id = run["artifact_id"]
            # Mutable parser objects and diagnostic attributes belong to this run only.
            scanner, chunker = copy(self.scanner), copy(self.chunker)
            parser = MultiLanguageParser(prefer_tree_sitter=self.parser.prefer_tree_sitter)
            graph_builder = CodeGraphBuilder(parser)
            weakness_detector = WeaknessDetector()
            statuses[-1] = ("fetched latest commit" if fetched else "using cached clone") + (f" {commit_sha[:10]}" if commit_sha else "")

            stage("scanning", "scanning files")
            records = scanner.scan(repo_id, repo_path)
            scan_stats = dict(scanner.last_scan_stats)

            stage("parsing", f"parsing {scan_stats.get('text_files', len(records))} text files")
            parsed = graph_builder.parse_all(records)
            scan_stats["parse_failures"] = sum(bundle.backend == "parse-failed" for bundle in parsed.values())
            entities_by_file = {path: bundle.data["entities"] for path, bundle in parsed.items()}

            stage("chunking", "creating symbol-aware chunks")
            chunks = chunker.chunk_files(records, max_chunks=self.settings.max_chunks_per_repo, entities_by_file=entities_by_file)
            chunk_stats = dict(chunker.last_stats)
            chunks = [Chunk.from_dict(redact_data(c.to_dict())) for c in chunks]
            coverage = build_coverage(scan_stats, chunk_stats, self._coverage_limits())
            if coverage["partial"]:
                note = coverage_message(coverage)
                logger.warning("[%s] %s", repo_id, note)
                statuses.append(note)

            stage("indexing", f"embedding {len(chunks)} chunks ({self.vector_store.embedder.model_id})")
            self.vector_store.build(artifact_id, chunks)

            stage("graph", "building code graph")
            graph = graph_builder.build(records, parsed)
            full_graph = graph_builder.last_graph
            if full_graph is not None:
                for _, attrs in full_graph.nodes(data=True):
                    attrs.update(redact_data(attrs))
                for _, _, attrs in full_graph.edges(data=True):
                    attrs.update(redact_data(attrs))
                self.graph_store.save(artifact_id, full_graph)

            stage("facts", "extracting repository facts")
            facts = self.facts_collector.collect(repo_url, records, graph)

            stage("history", "mining git history")
            complexity_by_file = self._complexity_by_file(full_graph)
            history = self.history_analyzer.analyze(repo_path, {r.path: r.line_count for r in records if r.kind == "text"}, complexity_by_file)

            stage("dependencies", "scanning dependencies for known vulnerabilities")
            dependencies = self.dependency_scanner.scan(records, self._imported_packages(full_graph))

            stage("drift", "checking documentation against code")
            drift = self.drift_detector.detect(records, facts, graph, self._env_usage(full_graph))

            stage("weaknesses", "detecting weaknesses")
            weaknesses = weakness_detector.detect(records, facts, graph, repo_id=repo_id)
            weaknesses.extend(self._insight_weaknesses(repo_id, drift, dependencies))

            stage("summary", "generating summary")
            insights = {"drift": drift, "dependencies": dependencies, "history": history}
            summary = self.summary_generator.generate(repo_url, records, graph, weaknesses, facts, insights, coverage=coverage)
            summary["ingestion"] = {
                **scan_stats,
                **chunk_stats,
                "coverage_partial": coverage["partial"],
                "commit_sha": commit_sha,
                "embedding_model": self.vector_store.embedder.model_id,
                "vector_backend": self.vector_store.backend_name,
                "parser_backends": graph.get("parser_backends", {}),
                "run_id": run["run_id"],
                "artifact_id": artifact_id,
                "pipeline_version": PIPELINE_VERSION,
                "source_kind": getattr(self.git, "source_kind", "github"),
            }

            stage("saving", "saving analysis")
            result = AnalysisResult(repo_id, normalize_github_url(repo_url), redact_data(summary),
                                    [], redact_data(graph), chunks, list(statuses), commit_sha)
            result.weaknesses = [Weakness(**{k: v for k, v in redact_data(w.to_dict()).items() if k != "issue"}) for w in weaknesses]
            result.statuses.append("analysis complete")
            self.snapshots.stage(result, records, run)
            if not self.vector_store.exists(artifact_id) or not self.graph_store.exists(artifact_id):
                raise RuntimeError("Analysis artifacts failed integrity validation before publication")
            self.metadata.save(result)
            self.snapshots.publish(repo_id, run)
            stage("completed", "analysis complete")
            result.statuses = list(statuses)
            return result

    def _fingerprint(self) -> str:
        names = ("parser_provider", "max_file_size_bytes", "max_files_per_repo", "max_total_repo_bytes", "max_chunks_per_repo",
                 "collapse_duplicate_files", "osv_scan_enabled", "git_history_depth", "follow_symlinks", "binary_metadata_enabled")
        config = {name: getattr(self.settings, name) for name in names}
        config.update(embedding=self.vector_store.embedder.model_id, backend=self.vector_store.backend_name,
                      pipeline=PIPELINE_VERSION)
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    def read_snapshot(self, repo_id: str):
        return self.snapshots.pin(repo_id)

    def close(self) -> None:
        self.metadata.close()

    def _coverage_limits(self) -> dict[str, int]:
        return {
            "max_files_per_repo": self.settings.max_files_per_repo,
            "max_total_repo_bytes": self.settings.max_total_repo_bytes,
            "chunk_cap": self.settings.max_chunks_per_repo,
        }

    def _insight_weaknesses(self, repo_id: str, drift: dict[str, Any], dependencies: dict[str, Any]) -> list[Weakness]:
        findings: list[Weakness] = []
        for issue in drift.get("issues", [])[:25]:
            findings.append(WeaknessDetector().make(
                repo_id, severity=issue["severity"], category="documentation_drift", title=issue["title"], file_path=issue.get("doc_path"),
                line_number=issue.get("doc_line"), evidence=issue.get("evidence"), why=f"Docs and code disagree: {issue['detail']}",
                fix="Update the documentation or the code so they match.", confidence="high",
            ))
        for vuln in dependencies.get("vulnerabilities", [])[:25]:
            severity = {"critical": "critical", "high": "high", "moderate": "medium", "medium": "medium", "low": "low"}.get(vuln.get("severity"), "medium")
            fixed = f" Review advisory fixed versions ({', '.join(vuln['fixed_versions'][:2])}) and choose a supported release." if vuln.get("fixed_versions") else ""
            reach = f" Imported in {', '.join(vuln['imported_in'][:3])}." if vuln.get("reachable") else " Not imported directly by repository code."
            findings.append(WeaknessDetector().make(
                repo_id, severity=severity, category="dependency", title=f"Vulnerable dependency {vuln['package']} {vuln['version']} ({vuln['id']})",
                file_path=vuln.get("source"), line_number=None, evidence=(vuln.get("summary") or "")[:200],
                why=f"Known vulnerability in {vuln['package']} {vuln['version']}.{reach}", fix=f"Update the dependency.{fixed}", confidence="high",
            ))
        return findings

    def _imported_packages(self, graph: Any) -> dict[str, list[str]]:
        imported: dict[str, list[str]] = defaultdict(list)
        if graph is None:
            return {}
        for source, target, attrs in graph.edges(data=True):
            if attrs.get("relation") == "imports" and target.startswith("dependency:") and source.startswith("file:"):
                if graph.nodes[source].get("role") == "code":
                    imported[target.removeprefix("dependency:")].append(source.removeprefix("file:"))
        return dict(imported)

    def _env_usage(self, graph: Any) -> dict[str, list[str]]:
        usage: dict[str, list[str]] = defaultdict(list)
        if graph is None:
            return {}
        for source, target, attrs in graph.edges(data=True):
            if attrs.get("relation") == "reads_env" and graph.nodes[source].get("role") in {"code", "config"}:
                usage[target.removeprefix("env:")].append(f"{source.removeprefix('file:')}:{attrs.get('line')}")
        return dict(usage)

    def _complexity_by_file(self, graph: Any) -> dict[str, int]:
        result: dict[str, int] = {}
        if graph is None:
            return result
        for _, attrs in graph.nodes(data=True):
            if attrs.get("kind") == "function" and attrs.get("complexity"):
                path = attrs.get("file_path")
                result[path] = max(result.get(path, 0), int(attrs["complexity"]))
        return result

    # --- read side ---------------------------------------------------------------------------
    def cache_ready(self, repo_id: str) -> bool:
        with self.read_snapshot(repo_id) as run:
            return bool(run and run.get("pipeline_version") == PIPELINE_VERSION
                        and run.get("config_fingerprint") == self._fingerprint()
                        and (self.snapshots.directory(repo_id, run) / "analysis.json").is_file()
                        and (self.snapshots.directory(repo_id, run) / "sources.json").is_file()
                        and self.vector_store.exists(repo_id) and self.graph_store.exists(repo_id))

    # Backwards-compatible alias.
    _cache_ready = cache_ready

    def load(self, repo_id: str, statuses: list[str] | None = None, include_chunks: bool = False) -> AnalysisResult:
        validate_repo_id(repo_id)
        with self.read_snapshot(repo_id) as run:
            result = self.snapshots.load(repo_id, run, include_chunks) if run else self.metadata.load(repo_id, include_chunks=include_chunks)
            if run is None and result.summary.get("ingestion", {}).get("run_id"):
                raise FileNotFoundError(f"No published analysis found for {repo_id}")
            if statuses is not None:
                result.statuses = list(statuses)
            if run and include_chunks:
                result.chunks = self.vector_store.all_chunks(repo_id)
            result.summary = redact_data(result.summary)
            result.graph = redact_data(result.graph)
            result.weaknesses = [Weakness(**{k: v for k, v in redact_data(w.to_dict()).items() if k != "issue"}) for w in result.weaknesses]
            return result

    def graph_index(self, repo_id: str) -> GraphIndex | None:
        return self.graph_store.load(validate_repo_id(repo_id))

    def list_repos(self) -> list[dict[str, Any]]:
        rows = {}
        for row in self.metadata.list_repos():
            try:
                self.load(row["repo_id"])
            except FileNotFoundError:
                continue
            rows[row["repo_id"]] = row
        for path in self.snapshots.root.glob("*/active.json"):
            result = self.load(path.parent.name)
            rows[result.repo_id] = {**rows.get(result.repo_id, {}), "repo_id": result.repo_id, "repo_url": result.repo_url,
                                   "commit_sha": result.commit_sha, "run_id": result.summary.get("ingestion", {}).get("run_id")}
        return list(rows.values())

    def repo_path(self, repo_id: str) -> Path:
        validate_repo_id(repo_id)
        path = self.git.path_for(repo_id) if hasattr(self.git, "path_for") else safe_join(self.settings.repos_dir, repo_id)
        if not path.is_dir():
            raise FileNotFoundError(f"No cached clone for {repo_id}")
        return path

    def read_file(self, repo_id: str, relative_path: str, start: int = 1, end: int | None = None, max_lines: int = 400, *, run_id: str | None = None) -> dict[str, Any]:
        with self.read_snapshot(repo_id) as run:
            if run_id:
                run = self.snapshots.historical(repo_id, run_id)
            if run:
                source = self.snapshots.source(repo_id, run, relative_path)
                if source is None:
                    raise FileNotFoundError(f"File not included in this analysis snapshot: {relative_path}")
                return self._file_payload(repo_id, relative_path, source["text"], start, end, max_lines,
                                          run_id=run["run_id"], commit_sha=run["commit_sha"], truncated=source["truncated"])
        root = self.repo_path(repo_id)
        target = safe_join(root, relative_path)
        if ".git" in Path(relative_path).parts or not target.is_file():
            raise FileNotFoundError(f"File not found: {relative_path}")
        with target.open("rb") as handle:
            raw = handle.read(self.settings.max_file_size_bytes)
        if b"\x00" in raw[:4096]:
            raise ValueError("Binary files cannot be displayed.")
        return self._file_payload(repo_id, relative_path, raw.decode("utf-8", errors="replace"), start, end, max_lines)

    @staticmethod
    def _file_payload(repo_id: str, relative_path: str, text: str, start: int, end: int | None, max_lines: int, **meta: Any) -> dict[str, Any]:
        lines = text.splitlines()
        start = max(1, start)
        end = min(len(lines), end or start + max_lines - 1, start + max_lines - 1)
        if start > end:
            raise ValueError("Requested line range is empty or outside the scanned source.")
        return {
            "repo_id": repo_id,
            "path": relative_path,
            "start_line": start,
            "end_line": end,
            "total_lines": len(lines),
            "text": redact("\n".join(lines[start - 1 : end])),
            **meta,
        }

    def delete(self, repo_id: str) -> None:
        validate_repo_id(repo_id)
        with self._lock_for(repo_id), self.snapshots.writer_lock(repo_id):
            directory = self.snapshots._repo(repo_id)
            for manifest_path in directory.glob("*/manifest.json"):
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.vector_store.delete(manifest["artifact_id"])
                self.graph_store.delete(manifest["artifact_id"])
            if directory.exists():
                shutil.rmtree(directory)
            self.git.delete(repo_id)
            metadata_file = safe_join(self.settings.metadata_dir, f"{repo_id}.json")
            if metadata_file.exists():
                metadata_file.unlink()
            self.metadata.delete(repo_id)
            self.vector_store.delete(repo_id)
            self.graph_store.delete(repo_id)

    def records_for(self, repo_id: str) -> list[FileRecord]:
        return FileScanner(policy=self.scanner.policy).scan(repo_id, self.repo_path(repo_id))
