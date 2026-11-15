from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

import networkx as nx

from app.graph.resolver import ImportResolver, external_package
from app.models import CodeEntity, FileRecord
from app.parsing.multilang_parser import MultiLanguageParser, ParseBundle

logger = logging.getLogger(__name__)

GENERIC_CALL_NAMES = {
    "get", "set", "run", "main", "init", "__init__", "start", "stop", "update", "save", "load", "print", "len", "str",
    "int", "list", "dict", "open", "close", "read", "write", "format", "join", "append", "add", "call", "handle",
    "render", "log", "info", "warning", "error", "debug", "then", "map", "filter", "push", "emit", "send",
}
MAX_API_NODES = 600
MAX_API_EDGES = 1500
MAX_FACTS = 400


def entity_node_id(entity: CodeEntity) -> str:
    return f"{entity.kind}:{entity.file_path}:{entity.qualname}"


def pagerank(graph: nx.DiGraph, damping: float = 0.85, iterations: int = 40) -> dict[str, float]:
    """Dependency-free power-iteration PageRank (networkx's version requires scipy)."""
    nodes = list(graph.nodes)
    if not nodes:
        return {}
    count = len(nodes)
    rank = dict.fromkeys(nodes, 1.0 / count)
    out_degree = {node: graph.out_degree(node) for node in nodes}
    for _ in range(iterations):
        dangling = damping * sum(rank[node] for node in nodes if out_degree[node] == 0) / count
        new_rank = {}
        for node in nodes:
            incoming = sum(rank[pred] / out_degree[pred] for pred in graph.predecessors(node) if out_degree[pred])
            new_rank[node] = (1 - damping) / count + dangling + damping * incoming
        rank = new_rank
    return rank


class CodeGraphBuilder:
    """Build a code knowledge graph: files, symbols, imports, calls, routes, env vars."""

    def __init__(self, parser: MultiLanguageParser | None = None) -> None:
        self.parser = parser or MultiLanguageParser()
        self.last_graph: nx.DiGraph | None = None

    def parse_all(self, records: list[FileRecord]) -> dict[str, ParseBundle]:
        return {record.path: self.parser.parse_with_backend(record) for record in records if record.kind == "text"}

    def build(self, records: list[FileRecord], parsed: dict[str, ParseBundle] | None = None) -> dict[str, Any]:
        parsed = parsed if parsed is not None else self.parse_all(records)
        graph = nx.DiGraph()
        facts: list[tuple[int, str]] = []
        resolver = ImportResolver(records)
        language_counts = Counter(record.language for record in records if record.kind == "text")
        parser_backends: Counter[str] = Counter(bundle.backend for bundle in parsed.values())
        entities_by_file: dict[str, list[CodeEntity]] = defaultdict(list)
        imported_symbols: dict[str, dict[str, tuple[str, str | None]]] = defaultdict(dict)
        unresolved_calls = 0

        # 1. Files, folders, declared packages.
        for record in records:
            folder = record.path.rsplit("/", 1)[0] if "/" in record.path else "."
            file_node = f"file:{record.path}"
            graph.add_node(f"folder:{folder}", kind="folder", label=folder)
            graph.add_node(
                file_node, kind="file", label=record.path, language=record.language, role=record.role,
                lines=record.line_count, file_kind=record.kind,
            )
            graph.add_edge(f"folder:{folder}", file_node, relation="contains")
            for package in self._declared_packages(record):
                package_node = f"dependency:{package}"
                graph.add_node(package_node, kind="dependency", label=package, declared=True)
                graph.add_edge(file_node, package_node, relation="declares_dependency")
                facts.append((3, f"{record.path} declares dependency {package}"))

        # 2. Symbols.
        for path, bundle in parsed.items():
            for entity in bundle.data["entities"]:
                node_id = entity_node_id(entity)
                attrs = {k: v for k, v in entity.metadata.items() if isinstance(v, (str, int, float, bool, list)) or v is None}
                graph.add_node(
                    node_id, kind=entity.kind, label=entity.qualname, name=entity.name, file_path=path,
                    start_line=entity.start_line, end_line=entity.end_line, **{k: v for k, v in attrs.items() if k not in {"qualname"}},
                )
                graph.add_edge(f"file:{path}", node_id, relation=f"defines_{entity.kind}")
                entities_by_file[path].append(entity)
                if entity.kind == "class" or not entity.metadata.get("parent"):
                    facts.append((4, f"{path} defines {entity.kind} {entity.qualname} at lines {entity.start_line}-{entity.end_line}"))

        name_index: dict[str, list[CodeEntity]] = defaultdict(list)
        for entities in entities_by_file.values():
            for entity in entities:
                name_index[entity.name].append(entity)

        # 3. Imports -> local files or external packages.
        for path, bundle in parsed.items():
            language = graph.nodes[f"file:{path}"].get("language", "")
            details = bundle.data.get("import_details") or [{"module": m, "names": [], "level": 0} for _, m in bundle.data["imports"]]
            for detail in details:
                targets = resolver.resolve(path, detail, language)
                if targets:
                    for target, symbol in targets:
                        graph.add_edge(f"file:{path}", f"file:{target}", relation="depends_on_file", import_name=detail.get("module"))
                        facts.append((2, f"{path} depends on {target}" + (f" (imports {symbol})" if symbol else "")))
                        if symbol:
                            alias = detail.get("aliases", {}).get(symbol, symbol)
                            imported_symbols[path][alias] = (target, symbol)
                        elif language != "python" and detail.get("names"):
                            for name in detail["names"]:
                                imported_symbols[path][name] = (target, name)
                        else:
                            alias = detail.get("alias") or (detail.get("module") or "").split(".")[-1]
                            if alias:
                                imported_symbols[path][alias] = (target, None)
                    continue
                module = detail.get("module") or ""
                if not module or (language == "python" and int(detail.get("level") or 0) > 0):
                    continue
                if language in {"javascript", "typescript"} and module.startswith((".", "/")):
                    continue
                package = external_package(module, language)
                if package:
                    dep_node = f"dependency:{package}"
                    graph.add_node(dep_node, kind="dependency", label=package)
                    graph.add_edge(f"file:{path}", dep_node, relation="imports", module=module)

        # 4. Calls, inheritance, routes, entrypoints, env vars.
        for path, bundle in parsed.items():
            local = entities_by_file.get(path, [])
            local_by_qualname = {entity.qualname: entity for entity in local}
            for caller, callee, line in bundle.data.get("calls", []):
                target = self._resolve_call(path, caller, callee, local, local_by_qualname, imported_symbols[path], entities_by_file, name_index)
                if target is None:
                    unresolved_calls += 1
                    continue
                target_entity, confidence = target
                caller_id = f"function:{path}:{caller}" if caller in local_by_qualname else f"file:{path}"
                relation = "instantiates" if target_entity.kind == "class" else "calls"
                target_id = entity_node_id(target_entity)
                if caller_id == target_id:
                    continue
                if graph.has_edge(caller_id, target_id):
                    graph.edges[caller_id, target_id]["count"] = graph.edges[caller_id, target_id].get("count", 1) + 1
                else:
                    graph.add_edge(caller_id, target_id, relation=relation, line=line, confidence=confidence, count=1)
            for entity in local:
                for base in entity.metadata.get("inherits", []) or []:
                    base_name = str(base).split(".")[-1]
                    base_entity = self._lookup_symbol(path, base_name, local, imported_symbols[path], entities_by_file, name_index, kinds={"class"})
                    base_node = entity_node_id(base_entity[0]) if base_entity else f"external_class:{base}"
                    if not base_entity:
                        graph.add_node(base_node, kind="external_class", label=str(base))
                    graph.add_edge(entity_node_id(entity), base_node, relation="inherits")
            for route in bundle.data.get("routes", []):
                route_node = f"route:{path}:{route.name}"
                graph.add_node(
                    route_node, kind="route", label=route.name, file_path=path, start_line=route.start_line, end_line=route.end_line,
                    method=route.metadata.get("method"), path=route.metadata.get("path"), handler=route.metadata.get("handler"),
                )
                graph.add_edge(f"file:{path}", route_node, relation="declares_route")
                handler = route.metadata.get("handler")
                if handler and handler in local_by_qualname:
                    graph.add_edge(route_node, entity_node_id(local_by_qualname[handler]), relation="handled_by")
                facts.append((0, f"{path} declares route {route.name} (lines {route.start_line}-{route.end_line})"))
            for _entry in bundle.data.get("entrypoints", []):
                entry_node = f"entrypoint:{path}"
                graph.add_node(entry_node, kind="entrypoint", label=f"{path} (__main__)", file_path=path)
                graph.add_edge(f"file:{path}", entry_node, relation="has_entrypoint")
                facts.append((0, f"{path} is directly executable (__main__ guard)"))
            for name, line in bundle.data.get("env_vars", []):
                env_node = f"env:{name}"
                graph.add_node(env_node, kind="env_var", label=name)
                if not graph.has_edge(f"file:{path}", env_node):
                    graph.add_edge(f"file:{path}", env_node, relation="reads_env", line=line)

        self.last_graph = graph
        return self.serialize(graph, records, facts, language_counts, parser_backends, unresolved_calls)

    # --- call / symbol resolution -----------------------------------------------------
    def _resolve_call(
        self,
        path: str,
        caller: str,
        callee: str,
        local: list[CodeEntity],
        local_by_qualname: dict[str, CodeEntity],
        imported: dict[str, tuple[str, str | None]],
        entities_by_file: dict[str, list[CodeEntity]],
        name_index: dict[str, list[CodeEntity]],
    ) -> tuple[CodeEntity, str] | None:
        parts = callee.split(".")
        last = parts[-1]
        head = parts[0]
        if head in {"self", "this", "cls"} and len(parts) == 2:
            owner = caller.rsplit(".", 1)[0] if "." in caller else ""
            method = local_by_qualname.get(f"{owner}.{last}") if owner else None
            if method:
                return method, "high"
        if callee in local_by_qualname:
            return local_by_qualname[callee], "high"
        if len(parts) == 1:
            top_level = [e for e in local if e.name == last and not e.metadata.get("parent")]
            if top_level:
                return top_level[0], "high"
        if head in imported:
            target_file, symbol = imported[head]
            wanted = symbol if len(parts) == 1 else last
            for entity in entities_by_file.get(target_file, []):
                if entity.qualname == wanted or (len(parts) == 1 and entity.name == wanted and not entity.metadata.get("parent")):
                    return entity, "high"
            if symbol and len(parts) == 2:  # ImportedClass.method
                for entity in entities_by_file.get(target_file, []):
                    if entity.qualname == f"{symbol}.{last}":
                        return entity, "high"
        if last in GENERIC_CALL_NAMES or len(last) < 4 or last.startswith("__"):
            return None
        candidates = [e for e in name_index.get(last, []) if not e.metadata.get("parent") or e.kind == "class"]
        if len(candidates) == 1 and candidates[0].file_path != path:
            return candidates[0], "low"
        if len(parts) > 1 and not candidates:
            # obj.method(): resolve when exactly one method with this name exists in the repo.
            methods = [e for e in name_index.get(last, []) if e.metadata.get("parent") and e.kind == "function"]
            if len(methods) == 1:
                return methods[0], "low"
        return None

    def _lookup_symbol(
        self,
        path: str,
        name: str,
        local: list[CodeEntity],
        imported: dict[str, tuple[str, str | None]],
        entities_by_file: dict[str, list[CodeEntity]],
        name_index: dict[str, list[CodeEntity]],
        kinds: set[str],
    ) -> tuple[CodeEntity, str] | None:
        for entity in local:
            if entity.name == name and entity.kind in kinds:
                return entity, "high"
        if name in imported:
            target_file, symbol = imported[name]
            for entity in entities_by_file.get(target_file, []):
                if entity.name == (symbol or name) and entity.kind in kinds:
                    return entity, "high"
        candidates = [e for e in name_index.get(name, []) if e.kind in kinds]
        if len(candidates) == 1:
            return candidates[0], "low"
        return None

    # --- serialization -----------------------------------------------------------------
    def file_graph(self, graph: nx.DiGraph) -> nx.DiGraph:
        """Collapse symbol-level edges into a file -> file dependency graph."""
        files = nx.DiGraph()
        for _node, attrs in graph.nodes(data=True):
            if attrs.get("kind") == "file" and attrs.get("file_kind", "text") == "text":
                files.add_node(attrs["label"])

        def owner(node: str) -> str | None:
            if node.startswith("file:"):
                return node[5:]
            return graph.nodes[node].get("file_path") if node in graph else None

        for source, target, attrs in graph.edges(data=True):
            if attrs.get("relation") not in {"depends_on_file", "calls", "instantiates", "inherits"}:
                continue
            a, b = owner(source), owner(target)
            if a and b and a != b and a in files and b in files:
                weight = files.edges[a, b]["weight"] + 1 if files.has_edge(a, b) else 1
                files.add_edge(a, b, weight=weight)
        return files

    def serialize(
        self,
        graph: nx.DiGraph,
        records: list[FileRecord],
        facts: list[tuple[int, str]],
        language_counts: Counter[str],
        parser_backends: Counter[str],
        unresolved_calls: int,
    ) -> dict[str, Any]:
        files = self.file_graph(graph)
        ranks = pagerank(files)
        central = sorted(ranks.items(), key=lambda item: item[1], reverse=True)
        imports = nx.DiGraph((s, t) for s, t, a in graph.edges(data=True) if a.get("relation") == "depends_on_file")
        cycles = [sorted(n.removeprefix("file:") for n in component)
                  for component in nx.strongly_connected_components(imports) if len(component) > 1]
        cycles.sort(key=len, reverse=True)
        complex_functions = sorted(
            (
                {
                    "symbol": attrs.get("label"),
                    "file_path": attrs.get("file_path"),
                    "start_line": attrs.get("start_line"),
                    "end_line": attrs.get("end_line"),
                    "complexity": attrs.get("complexity"),
                    "lines": (attrs.get("end_line") or 0) - (attrs.get("start_line") or 0) + 1,
                }
                for _, attrs in graph.nodes(data=True)
                if attrs.get("kind") == "function"
                and ((attrs.get("complexity") or 0) >= 12 or ((attrs.get("end_line") or 0) - (attrs.get("start_line") or 0)) >= 80)
            ),
            key=lambda item: ((item["complexity"] or 0), item["lines"]),
            reverse=True,
        )[:25]
        called = {target for _, target, attrs in graph.edges(data=True) if attrs.get("relation") in {"calls", "instantiates", "handled_by"}}
        unreferenced = [
            {"symbol": attrs.get("label"), "file_path": attrs.get("file_path"), "start_line": attrs.get("start_line")}
            for node, attrs in graph.nodes(data=True)
            if attrs.get("kind") == "function"
            and str(attrs.get("name", "")).startswith("_")
            and not str(attrs.get("name", "")).startswith("__")
            and node not in called
            and graph.nodes.get(f"file:{attrs.get('file_path')}", {}).get("role") == "code"
        ][:25]

        importance = {f"file:{path}": score for path, score in ranks.items()}
        keep: list[str] = []
        priority_kinds = {"route": 0, "entrypoint": 0, "file": 1, "class": 2, "function": 3, "dependency": 4, "env_var": 4, "folder": 5, "external_class": 6}
        ordered_nodes = sorted(
            graph.nodes(data=True),
            key=lambda item: (priority_kinds.get(item[1].get("kind", ""), 9), -importance.get(item[0], 0.0), -graph.degree(item[0]), item[0]),
        )
        for node, _ in ordered_nodes[:MAX_API_NODES]:
            keep.append(node)
        keep_set = set(keep)
        edges = [
            {"source": s, "target": t, **attrs}
            for s, t, attrs in graph.edges(data=True)
            if s in keep_set and t in keep_set and attrs.get("relation") != "contains"
        ]
        edge_priority = {"declares_route": 0, "handled_by": 0, "has_entrypoint": 0, "depends_on_file": 1, "calls": 2, "instantiates": 2, "inherits": 2, "imports": 3, "declares_dependency": 3, "reads_env": 3}
        edges.sort(key=lambda e: edge_priority.get(e.get("relation", ""), 5))
        facts.sort(key=lambda item: item[0])

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "languages": dict(language_counts),
            "parser_backends": dict(parser_backends),
            "node_kinds": dict(Counter(attrs.get("kind", "unknown") for _, attrs in graph.nodes(data=True))),
            "edge_relations": dict(Counter(attrs.get("relation", "unknown") for _, _, attrs in graph.edges(data=True))),
            "unresolved_calls": unresolved_calls,
            "top_dependencies": self._top_dependencies(graph),
            "dependencies": sorted({str(a.get("label")) for _, a in graph.nodes(data=True) if a.get("kind") == "dependency"})[:400],
            "central_files": [
                {"path": path, "score": round(score, 4), "in_degree": files.in_degree(path), "out_degree": files.out_degree(path)}
                for path, score in central[:20]
                if files.degree(path) > 0
            ],
            "dependency_cycles": cycles[:10],
            "complex_functions": complex_functions,
            "unreferenced_private_functions": unreferenced,
            "architecture_summary": self._architecture_summary(graph, central, cycles),
            "facts": [fact for _, fact in facts[:MAX_FACTS]],
            "mermaid": self._to_mermaid(files, central),
            "nodes": [{"id": node, **graph.nodes[node]} for node in keep],
            "edges": edges[:MAX_API_EDGES],
            "truncated": graph.number_of_nodes() > MAX_API_NODES or len(edges) > MAX_API_EDGES,
        }

    def _to_mermaid(self, files: nx.DiGraph, central: list[tuple[str, float]], limit: int = 24) -> str:
        top = [path for path, _ in central if files.degree(path) > 0][:limit]
        if not top:
            return 'flowchart TD\n    Repo["Repository (no internal dependencies detected)"]'
        top_set = set(top)
        lines = ["flowchart LR"]
        groups: dict[str, list[str]] = defaultdict(list)
        for path in top:
            groups[path.split("/", 1)[0] if "/" in path else "."].append(path)
        for group, paths in sorted(groups.items()):
            lines.append(f'    subgraph {self._mermaid_id("grp_" + group)}["{self._quote(group)}"]')
            for path in paths:
                lines.append(f'        {self._mermaid_id(path)}["{self._quote(PurePosixPath(path).name)}"]')
            lines.append("    end")
        for source, target in files.edges():
            if source in top_set and target in top_set:
                lines.append(f"    {self._mermaid_id(source)} --> {self._mermaid_id(target)}")
        return "\n".join(lines)

    def _mermaid_id(self, value: str) -> str:
        return "N" + "".join(char if char.isalnum() else "_" for char in value)[:80]

    def _quote(self, value: str) -> str:
        return value.replace('"', "'")

    def _top_dependencies(self, graph: nx.DiGraph) -> list[dict[str, int | str]]:
        counts: Counter[str] = Counter()
        for _, target, attrs in graph.edges(data=True):
            if attrs.get("relation") in {"imports", "declares_dependency"} and target.startswith("dependency:"):
                counts[target.removeprefix("dependency:")] += 1
        return [{"dependency": dep, "count": count} for dep, count in counts.most_common(20)]

    def _declared_packages(self, record: FileRecord) -> list[str]:
        name = record.path.rsplit("/", 1)[-1].lower()
        packages: list[str] = []
        if name.startswith("requirements") and name.endswith(".txt"):
            for line in record.text.splitlines():
                clean = line.split("#", 1)[0].strip()
                if not clean or clean.startswith("-"):
                    continue
                packages.append(re.split(r"[<>=~!;\[\s@]", clean, maxsplit=1)[0].strip())
        elif name == "package.json":
            try:
                data = json.loads(record.text)
            except json.JSONDecodeError:
                return []
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                section_data = data.get(section) or {}
                if isinstance(section_data, dict):
                    packages.extend(section_data.keys())
        elif name == "pyproject.toml":
            in_deps = False
            for line in record.text.splitlines():
                stripped = line.strip()
                if re.match(r"^(dependencies|requires|dev)\s*=\s*\[", stripped) or re.match(r"^\[(tool\.poetry\.)?(dev-)?dependencies\]", stripped):
                    in_deps = True
                if in_deps:
                    for match in re.finditer(r"""["']([A-Za-z0-9_.-]+)\s*(?:\[[^\]]*\])?\s*(?:[<>=~!;].*?)?["']""", stripped):
                        packages.append(match.group(1))
                    poetry = re.match(r"^([A-Za-z0-9_.-]+)\s*=", stripped)
                    if poetry and poetry.group(1) not in {"python", "dependencies", "requires", "dev"}:
                        packages.append(poetry.group(1))
                    if stripped.endswith("]") and not stripped.startswith("["):
                        in_deps = False
                    elif stripped.startswith("[") and "dependencies" not in stripped:
                        in_deps = False
        elif name == "go.mod":
            for match in re.finditer(r"^\s*(?:require\s+)?([\w./-]+\.[\w./-]+)\s+v[\d.]+", record.text, re.MULTILINE):
                packages.append(match.group(1))
        elif name == "cargo.toml":
            section = ""
            for line in record.text.splitlines():
                stripped = line.strip()
                if stripped.startswith("["):
                    section = stripped
                    continue
                if "dependencies" in section:
                    match = re.match(r"^([A-Za-z0-9_-]+)\s*=", stripped)
                    if match:
                        packages.append(match.group(1))
        return list(dict.fromkeys(package for package in packages if package))[:80]

    def _architecture_summary(self, graph: nx.DiGraph, central: list[tuple[str, float]], cycles: list[list[str]]) -> list[str]:
        summaries: list[str] = []
        for _node, attrs in graph.nodes(data=True):
            if attrs.get("kind") == "entrypoint":
                summaries.append(f"{attrs.get('file_path')} is an executable entrypoint.")
        route_files = Counter(attrs.get("file_path") for _, attrs in graph.nodes(data=True) if attrs.get("kind") == "route")
        for path, count in route_files.most_common(5):
            summaries.append(f"{path} declares {count} API route(s).")
        for path, _score in central[:5]:
            summaries.append(f"{path} is one of the most central modules in the dependency graph.")
        for cycle in cycles[:3]:
            summaries.append(f"Circular dependency between: {', '.join(cycle[:5])}.")
        return summaries[:30]
