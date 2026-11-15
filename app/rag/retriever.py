from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import Any

import networkx as nx

from app.graph.graph_store import GraphIndex
from app.models import Chunk
from app.rag.context import visible_lines
from app.rag.evidence import (
    DEFAULT_MIN_COVERAGE,
    DEFAULT_RERANK_RELEVANCE,
    EvidenceAssessment,
    TermGroup,
    Vocabulary,
    assess_evidence,
    content_term_groups,
)
from app.rag.fusion import reciprocal_rank_fusion
from app.rag.rerank import LocalReranker
from app.rag.tokenize import STOPWORDS, code_tokens, query_identifiers
from app.rag.vector_store import VectorStore
from app.utils.redaction import redact

logger = logging.getLogger(__name__)

INTENT_PATTERNS: dict[str, re.Pattern[str]] = {
    name: re.compile(pattern, re.IGNORECASE)
    for name, pattern in {
        "repo_overview": r"\b(overview|summar(?:y|ize|ise)|purpose|what\s+(?:does|is)\s+(?:this|the)\s+(?:repo|repository|project|app|application|tool|library|codebase)\b|what\s+is\s+this\b|explain\s+(?:this|the)\s+(?:repo|repository|project|codebase|app)|new\s+to\s+the\s+codebase|about\s+this\s+(?:repo|project))",
        "run_instructions": r"\b(install(?:ation|ing)?|set\s?up|setup|how\s+(?:do|can|to|should)\s+(?:i|we|you)?\s*(?:run|start|launch|use)|run\s+(?:it|this|the\s+\w+|locally)|quick\s?start|get(?:ting)?\s+started|deploy(?:ment|ing)?|docker|uninstall|fresh\s+vm|prerequisites?|environment\s+variables?)\b",
        "architecture": r"\b(architecture|structure|modules?|components?|design|data\s?flow|layers?|diagram|how\s+(?:does|do)\s+(?:it|this|the\s+(?:system|app|project|repo))\s+work|main\s+execution\s+flow|execution\s+flow)\b",
        "file_explanation": r"\b(which\s+files?|what\s+files?|where\s+(?:is|are|does|do)|located|read\s+first|important\s+files?|entry\s?points?)\b",
        "function_explanation": r"\b(function|method|class|implementation|implemented|signature|what\s+does\s+\S+\s+(?:do|return)|how\s+does\s+\S+\s+work)\b",
        "call_flow": r"\b(calls?|called|callers?|who\s+uses|used\s+by|trace|call\s+(?:chain|path|graph)|path\s+from|flows?\s+(?:from|to|through))\b",
        "dependency_question": r"\b(dependenc(?:y|ies)|imports?|packages?|librar(?:y|ies)|third[-\s]party|versions?)\b",
        "weakness_question": r"\b(weakness(?:es)?|risks?|issues?|problems?|bugs?|smells?|missing|flaws?|technical\s+debt)\b",
        "improvement_question": r"\b(improve|improvements?|suggest(?:ions?)?|recommend(?:ations?)?|better|refactor)\b",
        "testing_question": r"\b(tests?|testing|pytest|jest|coverage|unit\s+tests?|test\s+suite)\b",
        "security_question": r"\b(security|secrets?|vulnerab(?:le|ility|ilities)|injection|cve|xss|csrf|credentials?|unsafe)\b",
    }.items()
}
RUN_FILES = {"install.md", "installation.md", "setup.md", "quickstart.md", "readme.md", "pyproject.toml", "requirements.txt", "package.json", "makefile", "dockerfile", "docker-compose.yml", ".env.example"}
FOLLOW_UP_RE = re.compile(r"^\s*(and|also|what about|how about|why|then)\b|\b(it|its|that|this one|those|them|there)\b", re.IGNORECASE)

CHANNEL_WEIGHTS: dict[str, dict[str, float]] = {
    "default": {"dense": 1.0, "lexical": 1.0, "symbol": 1.6, "mentioned": 2.0, "priority": 0.6, "graph": 0.5},
    "run_instructions": {"facts": 2.4, "priority": 1.4, "dense": 0.7, "lexical": 0.8, "symbol": 0.5, "graph": 0.2},
    "repo_overview": {"priority": 1.4, "dense": 0.9, "lexical": 0.8, "graph": 0.4},
    "architecture": {"priority": 1.1, "graph": 1.0},
    "call_flow": {"symbol": 2.0, "graph": 1.2},
    "function_explanation": {"symbol": 2.0, "lexical": 1.2},
    "testing_question": {"priority": 1.2},
}


class HybridRetriever:
    """Multi-channel retrieval: dense + BM25 + exact symbol + intent priority + graph expansion,
    fused with weighted reciprocal rank fusion and reranked."""

    def __init__(
        self,
        vector_store: VectorStore,
        reranker: Any | None = None,
        min_coverage: float = DEFAULT_MIN_COVERAGE,
        rerank_relevance_threshold: float = DEFAULT_RERANK_RELEVANCE,
        use_graph_expansion: bool = True,
        use_reranker: bool = True,
        per_file_limit: int = 3,
    ) -> None:
        self.vector_store = vector_store
        self.reranker = reranker or LocalReranker()
        self.min_coverage = min_coverage
        self.rerank_relevance_threshold = rerank_relevance_threshold
        self.use_graph_expansion = use_graph_expansion
        self.use_reranker = use_reranker
        self.per_file_limit = per_file_limit
        self._repo_vocab: tuple[Any, Vocabulary] | None = None  # (bm25 index, its vocabulary)

    # --- query understanding -----------------------------------------------------------
    def classify(self, query: str) -> list[str]:
        intents = [intent for intent, pattern in INTENT_PATTERNS.items() if pattern.search(query)]
        if query_identifiers(query) and "function_explanation" not in intents:
            intents.append("function_explanation")
        if not intents:
            intents = ["general_code_question"]
        return intents

    def rewrite(self, query: str, history: list[dict[str, str]] | None) -> str:
        """Make follow-ups standalone by carrying key terms from the previous user turn."""
        if not history:
            return query
        previous = next((turn.get("content", "") for turn in reversed(history) if turn.get("role") == "user"), "")
        if not previous or not (FOLLOW_UP_RE.search(query) or len(code_tokens(query)) <= 3):
            return query
        carried = [token for token in dict.fromkeys(code_tokens(previous)) if token not in STOPWORDS][:8]
        return f"{query} {' '.join(carried)}".strip()

    # --- main entrypoint -----------------------------------------------------------------
    def retrieve(
        self,
        repo_id: str,
        query: str,
        graph: dict[str, Any],
        repo_facts: dict[str, Any] | None = None,
        weaknesses: list[dict[str, Any]] | None = None,
        top_k: int = 8,
        history: list[dict[str, str]] | None = None,
        graph_index: GraphIndex | None = None,
    ) -> dict[str, Any]:
        repo_facts = repo_facts or {}
        weaknesses = weaknesses or []
        intents = self.classify(query)
        search_query = self.rewrite(query, history)
        index = graph_index or self.graph_index_from_serialized(graph)
        identifiers = query_identifiers(search_query)

        channels: dict[str, list[Chunk]] = {
            "dense": self.vector_store.search(repo_id, search_query, top_k=30),
            "lexical": self.vector_store.lexical_search(repo_id, search_query, top_k=30),
        }
        symbol_nodes = self._symbol_nodes(search_query, identifiers, index)
        if symbol_nodes:
            ranges = [(str(n["file_path"]), int(n.get("start_line") or 1), int(n.get("end_line") or n.get("start_line") or 1)) for n in symbol_nodes]
            symbol_chunks = self.vector_store.chunks_for_ranges(repo_id, ranges)
            order = {(r[0], r[1]): i for i, r in enumerate(ranges)}
            symbol_chunks.sort(key=lambda c: (min((i for (path, start), i in order.items() if path == c.file_path and c.start_line <= start <= c.end_line), default=99), -(c.score or 0)))
            channels["symbol"] = symbol_chunks[:20]
        mentioned = self._mentioned_files(query, repo_facts, index)
        if mentioned:
            channels["mentioned"] = self.vector_store.chunks_for_files(repo_id, mentioned, limit_per_file=3, query=search_query)
        priority_files = self._priority_files(intents, repo_facts)
        if priority_files:
            priority_chunks = self.vector_store.chunks_for_files(repo_id, priority_files, limit_per_file=2, query=search_query)
            rank = {path: i for i, path in enumerate(self._priority_order(intents, repo_facts))}
            priority_chunks.sort(key=lambda c: (rank.get(c.file_path, 99), -(c.score or 0.0)))
            channels["priority"] = priority_chunks[:24]

        fact_ranges = self._fact_ranges(intents, repo_facts)
        if fact_ranges:
            fact_chunks = self.vector_store.chunks_for_ranges(repo_id, fact_ranges)
            order = {(path, start): i for i, (path, start, _end) in enumerate(fact_ranges)}
            fact_chunks.sort(key=lambda c: min((i for (path, start), i in order.items() if path == c.file_path and c.start_line <= start <= c.end_line), default=99))
            channels["facts"] = fact_chunks[:12]

        weights = self._weights(intents)
        seed, _ = reciprocal_rank_fusion({k: v for k, v in channels.items() if k in {"dense", "lexical", "symbol", "mentioned"}}, weights)
        seed_files = {chunk.file_path for chunk in seed[:5]}
        graph_files = index.expand_files(seed_files, hops=1, limit=8) if index and self.use_graph_expansion else []
        if graph_files:
            graph_chunks = self.vector_store.chunks_for_files(repo_id, set(graph_files), limit_per_file=1, query=search_query)
            order = {path: i for i, path in enumerate(graph_files)}
            graph_chunks.sort(key=lambda c: order.get(c.file_path, 99))
            channels["graph"] = graph_chunks

        fused, provenance = reciprocal_rank_fusion(channels, weights)
        if "run_instructions" in intents:
            fused = self._filter_run_chunks(query, fused, repo_facts)
        reranked = self.reranker.rerank(search_query, fused[:40], intents) if self.use_reranker else fused[:40]
        selected = self._diversify(reranked, top_k, per_file=self.per_file_limit)

        graph_context = self._graph_context(search_query, graph, index, selected, symbol_nodes)
        evidence = self.assess_evidence(repo_id, search_query, intents, selected, provenance, identifiers)
        return {
            "intents": intents,
            "search_query": search_query,
            "repo_facts": repo_facts,
            "selected_chunks": [chunk.to_dict() for chunk in selected],
            "chunks": [chunk.to_dict() for chunk in selected],
            "selected_graph_facts": graph_context["facts"],
            "graph_facts": graph_context["facts"],
            "graph_paths": graph_context["paths"],
            "weakness_facts": weaknesses if self._needs_weaknesses(intents) else [],
            "relevant_files": sorted({chunk.file_path for chunk in selected} | set(graph_context["files"])),
            "source_references": self._source_refs(selected),
            "retrieval_trace": [
                {"file_path": c.file_path, "lines": f"{c.start_line}-{c.end_line}", "score": c.score, "channels": provenance.get(c.id, {})}
                for c in selected
            ],
            "evidence_strength": evidence.level,
            "evidence": evidence.to_dict(),
        }

    # --- evidence strength -----------------------------------------------------------------
    def assess_evidence(
        self,
        repo_id: str,
        query: str,
        intents: list[str],
        selected: list[Chunk],
        provenance: dict[str, dict[str, int]],
        identifiers: set[str] | None = None,
    ) -> EvidenceAssessment:
        return assess_evidence(
            query,
            intents,
            selected,
            provenance,
            identifiers,
            absent_from_repo=self._absent_from_repo(repo_id, content_term_groups(query)),
            min_coverage=self.min_coverage,
            rerank_relevance_threshold=self.rerank_relevance_threshold,
        )

    def _absent_from_repo(self, repo_id: str, groups: list[TermGroup]) -> list[str]:
        """Content terms that occur nowhere in the indexed repository (checked against the BM25 vocabulary)."""
        if not groups:
            return []
        try:
            bm25 = self.vector_store.load(repo_id).bm25
        except Exception as exc:  # absence is an optional refinement; the assessment still works without it
            logger.info("Repository vocabulary unavailable for %s (%s); skipping absent-term check.", repo_id, exc)
            return []
        cached = self._repo_vocab
        if cached is None or cached[0] is not bm25:
            cached = (bm25, Vocabulary(set(bm25.idf)))
            self._repo_vocab = cached
        vocabulary = cached[1]
        return [group.label for group in groups if not group.matched_by(vocabulary)]

    # --- channels --------------------------------------------------------------------------
    def _weights(self, intents: list[str]) -> dict[str, float]:
        weights = dict(CHANNEL_WEIGHTS["default"])
        for intent in intents:
            for channel, value in CHANNEL_WEIGHTS.get(intent, {}).items():
                weights[channel] = max(weights.get(channel, 0.0), value) if intent != "run_instructions" else value
        return weights

    def _symbol_nodes(self, query: str, identifiers: set[str], index: GraphIndex | None) -> list[dict[str, Any]]:
        if index is None:
            return []
        exact = index.search_symbols(identifiers, limit=12)
        words = {token for token in code_tokens(query) if len(token) >= 4 and token not in STOPWORDS}
        loose = index.search_symbols(words, limit=12)
        seen: set[str] = set()
        nodes = []
        for node in exact + loose:
            if node.get("id") in seen or not node.get("file_path"):
                continue
            seen.add(node["id"])
            nodes.append(node)
        return nodes[:12]

    def _mentioned_files(self, query: str, facts: dict[str, Any], index: GraphIndex | None) -> set[str]:
        candidates = set(re.findall(r"[\w./-]+\.[A-Za-z]{1,6}\b", query))
        known = set(facts.get("file_tree", []))
        if index is not None:
            known |= {str(attrs.get("label")) for _, attrs in index.graph.nodes(data=True) if attrs.get("kind") == "file"}
        resolved = set()
        for candidate in sorted(candidates):
            if candidate in known:
                resolved.add(candidate)
                continue
            matches = sorted(path for path in known if path.endswith("/" + candidate) or PurePosixPath(path).name == candidate)
            resolved.update(matches[:3])
        return resolved

    def _priority_order(self, intents: list[str], facts: dict[str, Any]) -> list[str]:
        order: list[str] = []

        def add(paths: Any) -> None:
            for path in paths or []:
                if path and path not in order:
                    order.append(path)

        if "run_instructions" in intents:
            add(facts.get("install_doc_paths"))
            add(facts.get("run_doc_paths"))
            add(facts.get("official_doc_paths"))
            add(facts.get("readme_paths"))
            add(facts.get("dependency_files"))
            add(facts.get("deployment_files"))
            add(script["path"] for script in facts.get("helper_scripts", []) if script.get("mentioned_in_official_docs"))
            add(path for path in facts.get("file_tree", []) if PurePosixPath(path).name.lower() in RUN_FILES)
        if "repo_overview" in intents:
            add(facts.get("readme_paths"))
            add(item["path"] for item in facts.get("main_entrypoints", [])[:4])
            add(item["path"] for item in facts.get("important_modules", [])[:6])
            add(facts.get("dependency_files"))
        if "architecture" in intents or "call_flow" in intents:
            add(item["path"] for item in facts.get("main_entrypoints", [])[:6])
            add(item["path"] for item in facts.get("important_modules", [])[:10])
            add(facts.get("api_route_files"))
        if "file_explanation" in intents:
            add(item["path"] for item in facts.get("main_entrypoints", [])[:6])
            add(item["path"] for item in facts.get("important_files", [])[:8])
        if "testing_question" in intents:
            add(facts.get("test_files") or [p for p in facts.get("file_tree", []) if re.search(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_|_test\.|\.(test|spec)\.", p)])
        if "dependency_question" in intents:
            add(facts.get("dependency_files"))
        return order

    def _fact_ranges(self, intents: list[str], facts: dict[str, Any]) -> list[tuple[str, int, int]]:
        """Line ranges the deterministic fact extractor already tied to the question type."""
        ranges: list[tuple[str, int, int]] = []
        if "run_instructions" in intents:
            for item in facts.get("run_instructions", []):
                if item.get("official") and item.get("start_line"):
                    ranges.append((item["source"], int(item["start_line"]), int(item.get("end_line") or item["start_line"])))
            for section in facts.get("run_doc_sections", []):
                if section.get("start_line"):
                    ranges.append((section["source"], int(section["start_line"]), int(section.get("end_line") or section["start_line"])))
        return list(dict.fromkeys(ranges))[:30]

    def _priority_files(self, intents: list[str], facts: dict[str, Any]) -> set[str]:
        return set(self._priority_order(intents, facts)[:24])

    def _filter_run_chunks(self, query: str, chunks: list[Chunk], facts: dict[str, Any]) -> list[Chunk]:
        lowered = query.lower()
        excluded = {
            helper["path"]
            for helper in facts.get("helper_scripts", [])
            if not helper.get("mentioned_in_official_docs")
            and helper["path"].lower() not in lowered
            and helper["path"].rsplit("/", 1)[-1].lower() not in lowered
        }
        return [chunk for chunk in chunks if chunk.file_path not in excluded]

    def _diversify(self, chunks: list[Chunk], top_k: int, per_file: int = 3) -> list[Chunk]:
        selected: list[Chunk] = []
        per_file_count: dict[str, int] = {}
        for chunk in chunks:
            if per_file_count.get(chunk.file_path, 0) >= per_file:
                continue
            overlapping = any(
                other.file_path == chunk.file_path
                and min(other.end_line, chunk.end_line) - max(other.start_line, chunk.start_line) + 1
                > 0.5 * max(1, chunk.end_line - chunk.start_line + 1)
                for other in selected
            )
            if overlapping:
                continue
            selected.append(chunk)
            per_file_count[chunk.file_path] = per_file_count.get(chunk.file_path, 0) + 1
            if len(selected) >= top_k:
                break
        return selected

    # --- graph context -----------------------------------------------------------------------
    @staticmethod
    def graph_index_from_serialized(graph: dict[str, Any]) -> GraphIndex | None:
        nodes = graph.get("nodes") or []
        edges = graph.get("edges") or []
        if not nodes and not edges:
            return None
        digraph = nx.DiGraph()
        for node in nodes:
            attrs = {k: v for k, v in node.items() if k != "id"}
            digraph.add_node(node["id"], **attrs)
        for edge in edges:
            attrs = {k: v for k, v in edge.items() if k not in {"source", "target"}}
            digraph.add_edge(edge["source"], edge["target"], **attrs)
        return GraphIndex(digraph)

    def _graph_context(
        self,
        query: str,
        graph: dict[str, Any],
        index: GraphIndex | None,
        selected: list[Chunk],
        symbol_nodes: list[dict[str, Any]],
    ) -> dict[str, Any]:
        terms = {token for token in code_tokens(query) if len(token) > 2}
        selected_files = {chunk.file_path for chunk in selected}
        scored_facts: list[tuple[float, str]] = []
        for fact in graph.get("facts", []):
            fact_tokens = set(code_tokens(fact))
            score = len(terms & fact_tokens) + (1.5 if any(path in fact for path in selected_files) else 0.0)
            if score > 0:
                scored_facts.append((score, fact))
        scored_facts.sort(key=lambda item: item[0], reverse=True)

        paths: list[str] = []
        files: set[str] = set()
        if index is not None:
            for node in symbol_nodes[:5]:
                label = node.get("label")
                location = f"{node.get('file_path')}:{node.get('start_line')}"
                for caller in index.callers(node["id"], limit=4):
                    paths.append(f"{caller.get('label')} ({caller.get('file_path')}:{caller.get('line')}) calls {label} ({location})")
                    if caller.get("file_path"):
                        files.add(str(caller["file_path"]))
                for callee in index.callees(node["id"], limit=4):
                    paths.append(f"{label} ({location}) calls {callee.get('label')} ({callee.get('file_path')}:{callee.get('start_line')})")
                files.add(str(node["file_path"]))
        for edge in graph.get("edges", []):
            if len(paths) >= 12:
                break
            relation = str(edge.get("relation", ""))
            if relation in {"contains"}:
                continue
            haystack = set(code_tokens(f"{edge.get('source')} {edge.get('target')}"))
            if terms & haystack:
                source = str(edge.get("source", "")).split(":", 1)[-1]
                target = str(edge.get("target", "")).split(":", 1)[-1]
                paths.append(f"{source} {relation.replace('_', ' ')} {target}")
                for value in (str(edge.get("source", "")), str(edge.get("target", ""))):
                    if value.startswith("file:"):
                        files.add(value[5:])
        return {"facts": [fact for _, fact in scored_facts[:10]], "paths": list(dict.fromkeys(paths))[:12], "files": sorted(files)}

    def _needs_weaknesses(self, intents: list[str]) -> bool:
        return any(intent in {"weakness_question", "improvement_question", "security_question"} for intent in intents)

    def _source_refs(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        return [
            {
                "id": f"S{index}",
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "symbol": chunk.symbol,
                "excerpt": visible_lines(redact(chunk.text), 800),
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
