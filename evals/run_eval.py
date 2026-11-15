"""Retrieval benchmark: dense-only vs BM25-only vs RepoMind hybrid.

Usage:
    python -m evals.run_eval            # print a markdown table
    python -m evals.run_eval --json     # machine-readable results
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.utils.paths import repo_id_from_url
from evals.fixture_repo import QUESTIONS, build_fixture
from evals.metrics import retrieval_metrics

FIXTURE_URL = "https://github.com/repomind-evals/shopapi"


class LocalGitClient:
    """Serves a local directory as if it were a cloned repository (no network)."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def clone_or_get(self, repo_url: str, refresh: bool = False) -> tuple[str, Path, bool]:
        return repo_id_from_url(repo_url), self.path, True

    def path_for(self, repo_id: str) -> Path:
        return self.path

    def head_sha(self, repo_path: Path) -> str | None:
        return None

    def delete(self, repo_id: str) -> None:
        return None


def build_environment(workdir: Path, settings: Settings | None = None) -> tuple[RepoAnalyzer, ChatService, str]:
    repo_path = build_fixture(workdir / "fixture")
    base = settings or Settings()
    app_settings = replace(
        base, data_dir=workdir / "data", osv_scan_enabled=False, github_api_checks_enabled=False,
        llm_provider="off", embedding_provider="local", vector_store_provider="local", reranker_provider="local", use_metadata_db=True,
        database_url="",
    )
    analyzer = RepoAnalyzer(app_settings, git_client=LocalGitClient(repo_path))
    result = analyzer.analyze(FIXTURE_URL)
    return analyzer, ChatService(analyzer, app_settings), result.repo_id


def _metrics(ranked_files: list[list[str]], k_values: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    return retrieval_metrics(ranked_files, [gold for _, gold, _ in QUESTIONS], k_values)


def _unique_files(chunks: list[Any]) -> list[str]:
    files: list[str] = []
    for chunk in chunks:
        path = chunk.file_path if hasattr(chunk, "file_path") else chunk["file_path"]
        if path not in files:
            files.append(path)
    return files


def evaluate(analyzer: RepoAnalyzer, chat: ChatService, repo_id: str) -> dict[str, Any]:
    systems: dict[str, Callable[[str], list[str]]] = {
        "dense_only": lambda q: _unique_files(analyzer.vector_store.search(repo_id, q, top_k=10)),
        "bm25_only": lambda q: _unique_files(analyzer.vector_store.lexical_search(repo_id, q, top_k=10)),
        "repomind_hybrid": lambda q: _unique_files(chat.retrieve(repo_id, q, top_k=10)["selected_chunks"]),
    }
    results: dict[str, Any] = {"questions": len(QUESTIONS), "systems": {}, "failures": {}}
    for name, system in systems.items():
        ranked = [system(question) for question, _, _ in QUESTIONS]
        results["systems"][name] = _metrics(ranked)
        results["failures"][name] = [
            {"question": question, "gold": sorted(gold), "top3": files[:3]}
            for files, (question, gold, _) in zip(ranked, QUESTIONS, strict=True)
            if not gold & set(files[:3])
        ]
    symbol_hits = 0
    symbol_total = 0
    for question, _gold, symbol in QUESTIONS:
        if not symbol:
            continue
        symbol_total += 1
        chunks = chat.retrieve(repo_id, question, top_k=5)["selected_chunks"]
        found = [(chunk.get("symbol") or "").split("#")[0] for chunk in chunks]
        # A class-level chunk that contains the gold method also counts.
        if any(name and (name == symbol or symbol.startswith(name + ".")) for name in found):
            symbol_hits += 1
    results["systems"]["repomind_hybrid"]["symbol_recall@5"] = round(symbol_hits / max(1, symbol_total), 3)
    return results


def run(workdir: Path | None = None) -> dict[str, Any]:
    if workdir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return run(Path(tmp))
    analyzer, chat, repo_id = build_environment(workdir)
    return evaluate(analyzer, chat, repo_id)


def to_markdown(results: dict[str, Any]) -> str:
    columns = ["hit@1", "hit@3", "hit@5", "recall@5", "complete@5", "mrr"]
    lines = [f"Retrieval benchmark ({results['questions']} questions, ShopAPI fixture)", "", "| system | " + " | ".join(columns) + " |", "|---|" + "---|" * len(columns)]
    for name, metrics in results["systems"].items():
        lines.append(f"| {name} | " + " | ".join(f"{metrics[c]:.3f}" for c in columns) + " |")
    lines.append("")
    lines.append(f"Hybrid symbol-level recall@5: {results['systems']['repomind_hybrid']['symbol_recall@5']:.3f}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run()
    print(json.dumps(results, indent=2) if args.json else to_markdown(results))


if __name__ == "__main__":
    main()
