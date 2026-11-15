"""Offline multi-repository retrieval ablations with explicit draft-label handling."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from app.config import Settings
from app.core.repo_analyzer import RepoAnalyzer
from app.ingestion.git_client import git_command, git_env
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from evals.metrics import alternative_retrieval_metrics
from evals.run_eval import LocalGitClient


def validate_corpus(corpus: dict[str, Any], allow_draft: bool = False) -> None:
    repositories = {repo["name"]: repo for repo in corpus["repositories"]}
    if len(repositories) != len(corpus["repositories"]):
        raise ValueError("Repository names must be unique")
    seen = set()
    for question in corpus["questions"]:
        if question["id"] in seen:
            raise ValueError("Question ids must be unique")
        seen.add(question["id"])
        repo = repositories[question["repo"]]
        if question["commit_sha"] != repo["commit_sha"]:
            raise ValueError("Question and repository commit disagree")
        if repo["split"] not in {"dev", "test"}:
            raise ValueError("Split must be assigned at repository level")
        if not allow_draft and question.get("review_status") != "approved":
            raise ValueError("Unreviewed labels are not a quality gate; review them or explicitly use --allow-draft")
        if question.get("review_status") == "approved" and not question.get("reviewers"):
            raise ValueError("Approved questions must record their reviewers")
        if question["answerable"] and not question["evidence_sets"]:
            raise ValueError("Answerable questions require gold evidence")
        if not question["answerable"] and question["evidence_sets"]:
            raise ValueError("Unsupported questions must not have affirmative gold evidence")
        for alternative in question["evidence_sets"]:
            if not alternative:
                raise ValueError("Gold evidence alternatives cannot be empty")
            for span in alternative:
                path = Path(span["file_path"])
                if path.is_absolute() or ".." in path.parts or span["start_line"] < 1 or span["end_line"] < span["start_line"]:
                    raise ValueError("Gold evidence has an invalid source range")


def span_recall(chunks: list[dict[str, Any]], alternatives: list[list[dict[str, Any]]]) -> float:
    """Best alternative's fraction of required spans fully covered by retrieved lines."""
    from app.rag.grounding import _range_covered
    return max((sum(_range_covered(s["file_path"], s["start_line"], s["end_line"], chunks) for s in spans) / len(spans)
                for spans in alternatives), default=0.0)


def _files(chunks: list[dict[str, Any]]) -> list[str]:
    return list(dict.fromkeys(chunk["file_path"] for chunk in chunks))


def cluster_bootstrap_difference(rows: list[dict[str, Any]], left: str, right: str, samples: int = 1000) -> dict[str, Any]:
    by_repo: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["answerable"]:
            for system in (left, right):
                by_repo[row["repo"]][system].append(float(row["systems"][system]["hit@5"]))
    differences = [statistics.mean(values[left]) - statistics.mean(values[right]) for values in by_repo.values()]
    if not differences:
        return {}
    rng = random.Random(42)
    draws = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    return {"difference": statistics.mean(differences), "interval_95": [draws[int(samples * .025)], draws[int(samples * .975)]],
            "method": "paired repository-cluster bootstrap", "repositories": len(differences), "seed": 42}


def run(corpus_path: Path, repos_root: Path, workdir: Path, split: str = "dev", allow_draft: bool = False,
        semantic: bool = False, neural_reranker: bool = False, force_index: bool = False) -> dict[str, Any]:
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    validate_corpus(corpus, allow_draft)
    rows, timings, repo_metrics = [], defaultdict(list), {}
    providers = {}
    embedder = reranker = None
    if semantic:
        from app.rag.embeddings import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        if embedder.revision == "unversioned":
            raise RuntimeError("Semantic evaluation requires a resolved model revision")
    if neural_reranker:
        from app.rag.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
        if reranker.revision == "unversioned":
            raise RuntimeError("Neural evaluation requires a resolved model revision")
    for repo in corpus["repositories"]:
        if repo["split"] != split:
            continue
        root = (repos_root / repo["name"].replace("/", "--")).resolve()
        sha = subprocess.run(git_command("rev-parse", "HEAD"), cwd=root, env=git_env(), capture_output=True, text=True, check=True).stdout.strip()
        if sha != repo["commit_sha"]:
            raise ValueError(f"Checkout SHA mismatch for {repo['name']}")
        questions = [q for q in corpus["questions"] if q["repo"] == repo["name"]]
        for question in questions:
            for alternative in question["evidence_sets"]:
                for span in alternative:
                    path = root / span["file_path"]
                    if hashlib.sha256(path.read_bytes()).hexdigest() != span["file_sha256"]:
                        raise ValueError(f"Gold file was modified: {path}")
        git = LocalGitClient(root)
        git.head_sha = lambda path, sha=sha: sha
        config = replace(Settings(), data_dir=workdir / repo["name"].replace("/", "--"), database_url="",
                         llm_provider="off", gemini_api_key=None, openai_api_key=None, github_token=None,
                         osv_scan_enabled=False, github_api_checks_enabled=False,
                         embedding_provider="local", vector_store_provider="local", reranker_provider="local")
        analyzer = RepoAnalyzer(config, git_client=git)
        started = time.perf_counter()
        analysis = analyzer.analyze(repo["url"], force=force_index)
        index_seconds = time.perf_counter() - started
        facts, weaknesses = analysis.summary["repo_facts"], [w.to_dict() for w in analysis.weaknesses]
        index = analyzer.graph_index(analysis.repo_id)
        retrievers = {
            "hybrid": HybridRetriever(analyzer.vector_store),
            "no_graph_expansion": HybridRetriever(analyzer.vector_store, use_graph_expansion=False),
            "no_reranker": HybridRetriever(analyzer.vector_store, use_reranker=False),
        }
        if embedder is not None:
            semantic_store = VectorStore(workdir / repo["name"].replace("/", "--") / "semantic", embedder=embedder, app_settings=config)
            semantic_started = time.perf_counter()
            if force_index or not semantic_store.exists(analysis.repo_id):
                semantic_store.build(analysis.repo_id, analyzer.load(analysis.repo_id, include_chunks=True).chunks)
            semantic_seconds = time.perf_counter() - semantic_started
            retrievers["semantic_hybrid"] = HybridRetriever(semantic_store)
            providers.setdefault("semantic", {"model": embedder.model_id, "device": embedder.device})
        if reranker is not None:
            retrievers["neural_reranker"] = HybridRetriever(analyzer.vector_store, reranker)
            providers.setdefault("neural_reranker", {"model": reranker.model_id, "device": reranker.device})
        providers[repo["name"]] = {"embedding": analyzer.vector_store.embedder.model_id, "reranker": "local-features",
                                   "analysis_seconds": round(index_seconds, 4), "analysis_loaded_from_cache": analysis.statuses == ["loaded cached analysis"],
                                   "semantic_build_seconds": round(semantic_seconds, 4) if embedder else None,
                                   "coverage": analysis.summary.get("coverage"), "commit_sha": sha}
        rankings: dict[str, list[list[str]]] = defaultdict(list)
        gold = []
        for question in questions:
            row = {"id": question["id"], "repo": repo["name"], "category": question["category"],
                   "language": repo["language"], "answerable": question["answerable"], "review_status": question["review_status"], "systems": {}}
            alternatives = question["evidence_sets"]
            gold_files = {s["file_path"] for alternative in alternatives for s in alternative}
            gold.append([{s["file_path"] for s in alternative} for alternative in alternatives])
            row["gold_file_alternatives"] = [sorted(s) for s in gold[-1]]
            for system in ("dense", "bm25", *retrievers):
                started = time.perf_counter()
                if system in retrievers:
                    context = retrievers[system].retrieve(analysis.repo_id, question["question"], analysis.graph, facts, weaknesses,
                                                         top_k=30, graph_index=index)
                    chunks = context["selected_chunks"]
                    evidence = context["evidence"]
                else:
                    search = analyzer.vector_store.search if system == "dense" else analyzer.vector_store.lexical_search
                    chunks = [c.to_dict() for c in search(analysis.repo_id, question["question"], top_k=30)]
                    evidence = None
                elapsed = (time.perf_counter() - started) * 1000
                timings[system].append(elapsed)
                files = _files(chunks)
                rankings[system].append(files)
                row["systems"][system] = {"files": files, "hit@5": bool(gold_files & set(files[:5])),
                                           "span_recall@30_chunks": span_recall(chunks, alternatives) if alternatives else None,
                                           "latency_ms": round(elapsed, 3), "evidence": evidence}
            rows.append(row)
        repo_metrics[repo["name"]] = {system: alternative_retrieval_metrics(ranks, gold) for system, ranks in rankings.items()}
        analyzer.close()
    if not repo_metrics:
        raise ValueError(f"No repositories selected for {split}")
    aggregate = {system: {metric: round(statistics.mean(metrics[system][metric] for metrics in repo_metrics.values()), 4)
                          for metric in next(iter(repo_metrics.values()))[system]} for system in timings}
    slices = {}
    for dimension in ("category", "language"):
        slices[dimension] = {}
        for value in sorted({r[dimension] for r in rows}):
            selected = [r for r in rows if r[dimension] == value]
            labels = [[set(g) for g in r["gold_file_alternatives"]] for r in selected]
            slices[dimension][value] = {s: alternative_retrieval_metrics([r["systems"][s]["files"] for r in selected], labels) for s in timings}
    return {"status": "preliminary_draft_labels" if any(r["review_status"] != "approved" for r in rows) else "reviewed_labels",
            "split": split, "questions": len(rows), "repositories": len(repo_metrics), "providers": providers,
            "metric_notes": "Recall and completeness accept alternative gold evidence sets. Unsupported cases are excluded from retrieval denominators. No answer correctness or calibrated abstention claims.",
            "systems": aggregate, "per_repo": repo_metrics,
            "slices": slices,
            "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
            "run_mode": "full index rebuild" if force_index else "reuse compatible indexes",
            "latency_ms": {s: {"p50": round(statistics.median(t), 3), "p95": round(sorted(t)[min(len(t) - 1, int(.95 * len(t)))], 3)} for s, t in timings.items()},
            "paired_comparisons": {"hybrid_minus_bm25_hit@5": cluster_bootstrap_difference(rows, "hybrid", "bm25"),
                                   "hybrid_minus_no_graph_hit@5": cluster_bootstrap_difference(rows, "hybrid", "no_graph_expansion")},
            "skipped_experiments": (["semantic embedding comparison: not requested"] if not semantic else [])
                                    + (["neural reranking comparison: not requested"] if not neural_reranker else [])
                                    + ["answer correctness, semantic citation support and user task success: independent annotation required"],
            "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=Path("data/evals/corpus.json"))
    parser.add_argument("--repos-root", type=Path, default=Path("data/evals/repos"))
    parser.add_argument("--workdir", type=Path, default=Path("data/evals/indexes"))
    parser.add_argument("--output", type=Path, default=Path("data/evals/results-dev.json"))
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--allow-draft", action="store_true")
    parser.add_argument("--semantic", action="store_true")
    parser.add_argument("--neural-reranker", action="store_true")
    parser.add_argument("--force-index", action="store_true", help="Measure full analysis and embedding rebuilds rather than cached loads")
    args = parser.parse_args()
    results = run(args.corpus, args.repos_root, args.workdir, args.split, args.allow_draft, args.semantic, args.neural_reranker, args.force_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in results.items() if k in {"status", "split", "questions", "repositories", "systems", "latency_ms", "paired_comparisons"}}, indent=2))


if __name__ == "__main__":
    main()
