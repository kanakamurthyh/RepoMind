from __future__ import annotations

import logging
import shutil
from pathlib import Path

from app.analysis.summary import SummaryGenerator
from app.analysis.repo_facts import RepoFactsCollector
from app.analysis.weaknesses import WeaknessDetector
from app.config import Settings, settings
from app.db.repository import MetadataRepository
from app.graph.code_graph import CodeGraphBuilder
from app.ingestion.git_client import GitClient
from app.ingestion.scanner import FileScanner, IngestionPolicy
from app.models import AnalysisResult
from app.parsing.chunker import Chunker
from app.rag.vector_store import VectorStore
from app.utils.paths import repo_id_from_url

logger = logging.getLogger(__name__)


class RepoAnalyzer:
    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self.settings.ensure_dirs()
        self.git = GitClient(self.settings.repos_dir)
        self.scanner = FileScanner(
            policy=IngestionPolicy(
                max_file_size_bytes=self.settings.max_file_size_bytes,
                max_files_per_repo=self.settings.max_files_per_repo,
                max_total_repo_bytes=self.settings.max_total_repo_bytes,
                include_binary_metadata=self.settings.binary_metadata_enabled,
                follow_symlinks=self.settings.follow_symlinks,
            )
        )
        self.chunker = Chunker()
        self.graph_builder = CodeGraphBuilder()
        self.facts_collector = RepoFactsCollector()
        self.weakness_detector = WeaknessDetector()
        self.summary_generator = SummaryGenerator()
        self.vector_store = VectorStore(self.settings.indexes_dir)
        self.metadata = MetadataRepository(
            self.settings.resolved_database_url,
            self.settings.metadata_dir,
            enabled=self.settings.use_metadata_db,
        )

    def analyze(self, repo_url: str, force: bool = False) -> AnalysisResult:
        statuses: list[str] = []
        repo_id = repo_id_from_url(repo_url)
        metadata_path = self._metadata_path(repo_id)
        if self._cache_ready(repo_id) and not force:
            statuses.append("loaded cached analysis")
            return self.load(repo_id, statuses=statuses)
        if metadata_path.exists() and not self.vector_store.exists(repo_id) and not force:
            statuses.append("cached metadata found but vector index is missing; rebuilding")

        statuses.append("cloning")
        repo_id, repo_path, cloned = self.git.clone_or_get(repo_url)
        statuses[-1] = "cloned repository" if cloned else "using cached clone"

        statuses.append("scanning files")
        records = self.scanner.scan(repo_id, repo_path)
        logger.info("Scanned %d files", len(records))

        statuses.append("creating chunks")
        chunks = self.chunker.chunk_files(records, max_chunks=self.settings.max_chunks_per_repo)

        statuses.append("building embeddings")
        self.vector_store.build(repo_id, chunks)

        statuses.append("building graph")
        graph = self.graph_builder.build(records)

        statuses.append("extracting repo facts")
        facts = self.facts_collector.collect(repo_url, records, graph)

        statuses.append("detecting weaknesses")
        weaknesses = self.weakness_detector.detect(records, facts)

        statuses.append("generating summary")
        summary = self.summary_generator.generate(repo_url, records, graph, weaknesses, facts)

        result = AnalysisResult(repo_id, repo_url, summary, weaknesses, graph, chunks, statuses)
        self._save(result)
        return result

    def load(self, repo_id: str, statuses: list[str] | None = None) -> AnalysisResult:
        return self.metadata.load(repo_id, statuses=statuses)

    def delete(self, repo_id: str) -> None:
        for path in (self.settings.repos_dir / repo_id, self._metadata_path(repo_id)):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        self.metadata.delete(repo_id)
        self.vector_store.delete(repo_id)

    def _save(self, result: AnalysisResult) -> None:
        self.metadata.save(result)

    def _metadata_path(self, repo_id: str) -> Path:
        return self.settings.metadata_dir / f"{repo_id}.json"

    def _cache_ready(self, repo_id: str) -> bool:
        return self.metadata.exists(repo_id) and self.vector_store.exists(repo_id)
