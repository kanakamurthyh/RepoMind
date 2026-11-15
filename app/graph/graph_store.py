"""Persist the full code graph and answer structural queries over it."""

from __future__ import annotations

import gzip
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)
SYMBOL_KINDS = {"function", "class"}


class GraphStore:
    """Stores the complete (untruncated) graph per repo as gzipped node-link JSON."""

    def __init__(self, graphs_dir: Path, cache_size: int = 8) -> None:
        self.graphs_dir = graphs_dir
        self.graphs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: OrderedDict[str, tuple[float, GraphIndex]] = OrderedDict()
        self._lock = threading.Lock()
        self.cache_size = cache_size
        self.resolve_id: Callable[[str], str] = lambda repo_id: repo_id

    def _path(self, repo_id: str) -> Path:
        repo_id = self.resolve_id(repo_id)
        return self.graphs_dir / f"{repo_id}.json.gz"

    def save(self, repo_id: str, graph: nx.DiGraph) -> None:
        payload = nx.node_link_data(graph, edges="links")
        path = self._path(repo_id)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        tmp.replace(path)
        with self._lock:
            self._cache.pop(repo_id, None)

    def exists(self, repo_id: str) -> bool:
        return self.load(repo_id) is not None

    def load(self, repo_id: str) -> GraphIndex | None:
        repo_id = self.resolve_id(repo_id)
        path = self._path(repo_id)
        if not path.exists():
            return None
        mtime = path.stat().st_mtime
        with self._lock:
            cached = self._cache.get(repo_id)
            if cached and cached[0] == mtime:
                self._cache.move_to_end(repo_id)
                return cached[1]
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                graph = nx.node_link_graph(json.load(handle), directed=True, edges="links")
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("Graph for %s is unreadable and should be rebuilt: %s", repo_id, exc)
            return None
        index = GraphIndex(graph)
        with self._lock:
            self._cache[repo_id] = (mtime, index)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return index

    def delete(self, repo_id: str) -> None:
        path = self._path(repo_id)
        if path.exists():
            path.unlink()
        with self._lock:
            self._cache.pop(repo_id, None)


class GraphIndex:
    """Read-only structural queries used by retrieval, the agent, and the API."""

    def __init__(self, graph: nx.DiGraph) -> None:
        self.graph = graph
        self.symbols_by_name: dict[str, list[str]] = {}
        for node, attrs in graph.nodes(data=True):
            if attrs.get("kind") in SYMBOL_KINDS:
                for key in {str(attrs.get("name", "")).lower(), str(attrs.get("label", "")).lower()}:
                    if key:
                        self.symbols_by_name.setdefault(key, []).append(node)

    def node(self, node_id: str) -> dict[str, Any]:
        return {"id": node_id, **self.graph.nodes[node_id]} if node_id in self.graph else {}

    def find_symbols(self, name: str, limit: int = 10) -> list[dict[str, Any]]:
        key = name.lower().strip()
        nodes = list(self.symbols_by_name.get(key, []))
        if not nodes and "." in key:
            nodes = list(self.symbols_by_name.get(key.split(".")[-1], []))
        return [self.node(node) for node in nodes[:limit]]

    def search_symbols(self, terms: set[str], limit: int = 20) -> list[dict[str, Any]]:
        """Symbols whose name equals a query identifier (exact, case-insensitive)."""
        hits: list[str] = []
        for term in sorted(terms):
            hits.extend(self.symbols_by_name.get(term.lower(), []))
        return [self.node(node) for node in list(dict.fromkeys(hits))[:limit]]

    def callers(self, node_id: str, limit: int = 25) -> list[dict[str, Any]]:
        if node_id not in self.graph:
            return []
        rows = []
        for source in self.graph.predecessors(node_id):
            attrs = self.graph.edges[source, node_id]
            if attrs.get("relation") in {"calls", "instantiates", "handled_by"}:
                rows.append({**self.node(source), "line": attrs.get("line"), "relation": attrs.get("relation")})
        return rows[:limit]

    def callees(self, node_id: str, limit: int = 25) -> list[dict[str, Any]]:
        if node_id not in self.graph:
            return []
        rows = []
        for target in self.graph.successors(node_id):
            attrs = self.graph.edges[node_id, target]
            if attrs.get("relation") in {"calls", "instantiates"}:
                rows.append({**self.node(target), "line": attrs.get("line"), "relation": attrs.get("relation")})
        return rows[:limit]

    def file_neighbors(self, path: str, limit: int = 20) -> dict[str, list[str]]:
        node = f"file:{path}"
        if node not in self.graph:
            return {"imports": [], "imported_by": [], "symbols": []}
        imports = [t[5:] for t in self.graph.successors(node) if t.startswith("file:") and self.graph.edges[node, t].get("relation") == "depends_on_file"]
        imported_by = [s[5:] for s in self.graph.predecessors(node) if s.startswith("file:") and self.graph.edges[s, node].get("relation") == "depends_on_file"]
        symbols = [self.graph.nodes[t].get("label") for t in self.graph.successors(node) if self.graph.nodes[t].get("kind") in SYMBOL_KINDS]
        return {"imports": imports[:limit], "imported_by": imported_by[:limit], "symbols": symbols[: limit * 2]}

    def expand_files(self, seed_files: set[str], hops: int = 1, limit: int = 12) -> list[str]:
        """Files structurally adjacent to the seeds (imports, importers, cross-file calls)."""
        found: dict[str, int] = {}
        frontier = {f"file:{path}" for path in seed_files if f"file:{path}" in self.graph}
        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in sorted(frontier):
                for neighbor in sorted(set(self.graph.successors(node)) | set(self.graph.predecessors(node))):
                    target = neighbor if neighbor.startswith("file:") else None
                    if target is None:
                        owner = self.graph.nodes[neighbor].get("file_path")
                        target = f"file:{owner}" if owner else None
                    if target and target not in frontier and target[5:] not in seed_files:
                        found[target[5:]] = found.get(target[5:], 0) + 1
                        next_frontier.add(target)
            frontier = next_frontier
        return [path for path, _ in sorted(found.items(), key=lambda item: (-item[1], item[0]))][:limit]

    def symbol_files(self, symbol_nodes: list[dict[str, Any]]) -> set[str]:
        return {str(node["file_path"]) for node in symbol_nodes if node.get("file_path")}

    def call_path(self, source: str, target: str, max_len: int = 8) -> list[dict[str, Any]]:
        """Shortest call chain between two symbols (by node id or name)."""
        sources = [source] if source in self.graph else [n["id"] for n in self.find_symbols(source)]
        targets = [target] if target in self.graph else [n["id"] for n in self.find_symbols(target)]
        call_graph = self.graph.edge_subgraph(
            [(s, t) for s, t, a in self.graph.edges(data=True) if a.get("relation") in {"calls", "instantiates", "handled_by"}]
        )
        best: list[str] | None = None
        for s in sources:
            for t in targets:
                if s in call_graph and t in call_graph:
                    try:
                        path = nx.shortest_path(call_graph, s, t)
                    except nx.NetworkXNoPath:
                        continue
                    if len(path) <= max_len and (best is None or len(path) < len(best)):
                        best = path
        return [self.node(node) for node in best or []]

    def routes(self) -> list[dict[str, Any]]:
        return [self.node(n) for n, a in self.graph.nodes(data=True) if a.get("kind") == "route"]
