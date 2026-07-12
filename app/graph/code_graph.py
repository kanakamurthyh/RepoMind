from __future__ import annotations

from collections import Counter
import json
from pathlib import PurePosixPath
import re
from typing import Any

import networkx as nx

from app.models import FileRecord
from app.parsing.multilang_parser import MultiLanguageParser


class CodeGraphBuilder:
    def __init__(self) -> None:
        self.parser = MultiLanguageParser()

    def build(self, records: list[FileRecord]) -> dict[str, Any]:
        graph = nx.DiGraph()
        facts: list[str] = []
        language_counts = Counter(record.language for record in records)
        parser_backends: Counter[str] = Counter()
        path_lookup = self._build_path_lookup(records)

        for record in records:
            folder = record.path.rsplit("/", 1)[0] if "/" in record.path else "."
            folder_node = f"folder:{folder}"
            file_node = f"file:{record.path}"
            graph.add_node(folder_node, kind="folder", label=folder)
            graph.add_node(file_node, kind="file", label=record.path, language=record.language, role=record.role)
            graph.add_edge(folder_node, file_node, relation="contains")
            for package in self._declared_packages(record):
                package_node = f"dependency:{package}"
                graph.add_node(package_node, kind="dependency", label=package)
                graph.add_edge(file_node, package_node, relation="depends_on_package")
                facts.append(f"{record.path} declares dependency {package}")

            parsed_bundle = self.parser.parse_with_backend(record)
            parse_data = parsed_bundle.data
            parser_backends[parsed_bundle.backend] += 1
            for source, module in parse_data["imports"]:
                dep_node = f"dependency:{module}"
                graph.add_node(dep_node, kind="dependency", label=module)
                graph.add_edge(file_node, dep_node, relation="imports")
                facts.append(f"{record.path} imports {module}")
                local_path = self._resolve_local_import(record.path, module, path_lookup)
                if local_path:
                    local_node = f"file:{local_path}"
                    graph.add_node(local_node, kind="file", label=local_path)
                    graph.add_edge(file_node, local_node, relation="depends_on_file", import_name=module)
                    facts.append(f"{record.path} depends on local file {local_path}")

            for entity in parse_data["entities"]:
                entity_node = f"{entity.kind}:{entity.file_path}:{entity.name}"
                graph.add_node(
                    entity_node,
                    kind=entity.kind,
                    label=entity.name,
                    file_path=entity.file_path,
                    start_line=entity.start_line,
                    end_line=entity.end_line,
                    **entity.metadata,
                )
                graph.add_edge(file_node, entity_node, relation=f"defines_{entity.kind}")
                facts.append(f"{record.path} defines {entity.kind} {entity.name} at lines {entity.start_line}-{entity.end_line}")
                for base in entity.metadata.get("inherits", []):
                    if base:
                        base_node = f"class:{base}"
                        graph.add_node(base_node, kind="class", label=base)
                        graph.add_edge(entity_node, base_node, relation="inherits")

            for route in parse_data["routes"]:
                route_node = f"route:{route.file_path}:{route.name}"
                graph.add_node(route_node, kind="route", label=route.name, file_path=route.file_path)
                graph.add_edge(file_node, route_node, relation="declares_route")
                facts.append(f"{route.file_path} declares route {route.name} handled near lines {route.start_line}-{route.end_line}")

            for caller, callee, line in parse_data.get("calls", []):
                if caller and callee:
                    caller_node = f"function:{record.path}:{caller}"
                    callee_node = f"call:{callee}"
                    graph.add_node(callee_node, kind="call", label=callee)
                    graph.add_edge(caller_node, callee_node, relation="calls", line=line)

            for entry in parse_data.get("entrypoints", []):
                graph.add_edge(file_node, f"entrypoint:{entry.name}", relation="has_entrypoint")
                graph.add_node(f"entrypoint:{entry.name}", kind="entrypoint", label=entry.name, file_path=entry.file_path)
                facts.append(f"{entry.file_path} has a Python main entrypoint")

        return {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "languages": dict(language_counts),
            "parser_backends": dict(parser_backends),
            "node_kinds": dict(Counter(attrs.get("kind", "unknown") for _, attrs in graph.nodes(data=True))),
            "edge_relations": dict(Counter(attrs.get("relation", "unknown") for _, _, attrs in graph.edges(data=True))),
            "top_dependencies": self._top_dependencies(graph),
            "architecture_summary": self._architecture_summary(graph),
            "facts": facts[:300],
            "mermaid": self._to_mermaid(records, graph),
            "nodes": [
                {"id": node, **attrs}
                for node, attrs in list(graph.nodes(data=True))[:500]
            ],
            "edges": [
                {"source": source, "target": target, **attrs}
                for source, target, attrs in list(graph.edges(data=True))[:800]
            ],
        }

    def _to_mermaid(self, records: list[FileRecord], graph: nx.DiGraph) -> str:
        folders = sorted({record.path.split("/", 1)[0] if "/" in record.path else "." for record in records})[:12]
        lines = ["flowchart TD", '    Repo["Repository"]']
        for folder in folders:
            node_id = self._mermaid_id(folder)
            lines.append(f'    Repo --> {node_id}["{self._quote(folder)}"]')
        for source, target, attrs in list(graph.edges(data=True))[:80]:
            if attrs.get("relation") in {"imports", "depends_on_file", "declares_route", "has_entrypoint"}:
                s_id = self._mermaid_id(source)
                t_id = self._mermaid_id(target)
                s_label = source.split(":", 1)[-1][:40]
                t_label = target.split(":", 1)[-1][:40]
                lines.append(f'    {s_id}["{self._quote(s_label)}"] -->|{attrs["relation"]}| {t_id}["{self._quote(t_label)}"]')
        return "\n".join(lines)

    def _mermaid_id(self, value: str) -> str:
        return "N" + "".join(char if char.isalnum() else "_" for char in value)[:60]

    def _quote(self, value: str) -> str:
        return value.replace('"', "'")

    def _top_dependencies(self, graph: nx.DiGraph) -> list[dict[str, int | str]]:
        counts: Counter[str] = Counter()
        for _, target, attrs in graph.edges(data=True):
            if attrs.get("relation") in {"imports", "depends_on_package"} and target.startswith("dependency:"):
                counts[target.removeprefix("dependency:").split(".", 1)[0]] += 1
        return [{"dependency": dep, "count": count} for dep, count in counts.most_common(15)]

    def _build_path_lookup(self, records: list[FileRecord]) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for record in records:
            if record.role != "code":
                continue
            path = PurePosixPath(record.path)
            without_suffix = str(path.with_suffix(""))
            candidates = {
                record.path,
                without_suffix,
                path.name,
                path.stem,
                without_suffix.replace("/", "."),
            }
            for candidate in candidates:
                lookup.setdefault(candidate, record.path)
        return lookup

    def _resolve_local_import(self, current_path: str, module: str, lookup: dict[str, str]) -> str | None:
        clean = module.strip().strip("'\"")
        candidates = [clean, clean.replace(".", "/")]
        if clean.startswith("."):
            base = PurePosixPath(current_path).parent
            joined = str((base / clean).as_posix())
            while joined.startswith("./"):
                joined = joined[2:]
            candidates.extend([joined, joined.replace(".", "/")])
        for candidate in candidates:
            normalized = candidate.removeprefix("/")
            if normalized in lookup:
                return lookup[normalized]
            for suffix in (".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".hpp"):
                if f"{normalized}{suffix}" in lookup:
                    return lookup[f"{normalized}{suffix}"]
        return None

    def _declared_packages(self, record: FileRecord) -> list[str]:
        name = record.path.rsplit("/", 1)[-1].lower()
        packages: list[str] = []
        if name == "requirements.txt":
            for line in record.text.splitlines():
                clean = line.strip()
                if not clean or clean.startswith("#") or clean.startswith("-"):
                    continue
                packages.append(re.split(r"[<>=~!;\[]", clean, maxsplit=1)[0].strip())
        elif name == "package.json":
            try:
                data = json.loads(record.text)
            except json.JSONDecodeError:
                return []
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                packages.extend((data.get(section) or {}).keys())
        elif name == "pyproject.toml":
            for line in record.text.splitlines():
                clean = line.strip().strip(",").strip('"').strip("'")
                if re.match(r"^[A-Za-z0-9_.-]+[<>=~!]", clean):
                    packages.append(re.split(r"[<>=~!;\[]", clean, maxsplit=1)[0].strip())
        return [package for package in packages if package][:40]

    def _architecture_summary(self, graph: nx.DiGraph) -> list[str]:
        summaries = []
        for source, target, attrs in list(graph.edges(data=True))[:120]:
            relation = attrs.get("relation")
            if relation == "has_entrypoint":
                summaries.append(f"{source.removeprefix('file:')} is an executable entrypoint.")
            elif relation == "declares_route":
                summaries.append(f"{source.removeprefix('file:')} declares API routes.")
            elif relation == "defines_function" and len(summaries) < 20:
                summaries.append(f"{source.removeprefix('file:')} defines {target.split(':')[-1]}.")
        return summaries[:30]
