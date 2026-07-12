from __future__ import annotations

import re
from typing import Any

from app.models import Chunk
from app.rag.vector_store import VectorStore


INTENT_KEYWORDS = {
    "repo_overview": {"overview", "summary", "what does", "explain", "explain me", "purpose", "project"},
    "run_instructions": {"install", "installation", "linux", "fresh vm", "uninstall", "run", "start", "setup", "requirements", "quickstart", "execute", "usage", "build", "deployment"},
    "architecture": {"architecture", "structure", "modules", "components", "flow", "how does it work"},
    "file_explanation": {"which file", "important files", "where is", "file", "located"},
    "function_explanation": {"function", "class", "method", "explain function"},
    "dependency_question": {"dependency", "dependencies", "import", "package", "library"},
    "weakness_question": {"weakness", "risk", "issue", "problem", "missing"},
    "improvement_question": {"improve", "suggest", "recommendation", "better"},
    "testing_question": {"test", "pytest", "coverage", "unit test"},
    "security_question": {"security", "secret", "eval", "performance", "sql", "vulnerability"},
}

RUN_FILES = {"install.md", "installation.md", "setup.md", "quickstart.md", "readme.md", "pyproject.toml", "requirements.txt", "package.json", "makefile", "dockerfile", "docker-compose.yml"}
SECRET_PATTERNS = [
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{25,}\b"),
    re.compile(r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"),
]


class HybridRetriever:
    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def classify(self, query: str) -> list[str]:
        lowered = query.lower()
        intents = [intent for intent, keys in INTENT_KEYWORDS.items() if any(key in lowered for key in keys)]
        if not intents:
            intents = ["general_code_question"]
        if "weakness_question" in intents and "improvement_question" not in intents and "improve" in lowered:
            intents.append("improvement_question")
        return intents

    def retrieve(
        self,
        repo_id: str,
        query: str,
        graph: dict[str, Any],
        repo_facts: dict[str, Any] | None = None,
        weaknesses: list[dict[str, Any]] | None = None,
        top_k: int = 8,
    ) -> dict[str, Any]:
        repo_facts = repo_facts or {}
        weaknesses = weaknesses or []
        intents = self.classify(query)
        priority_files = self._priority_files(query, intents, repo_facts, graph)
        vector_chunks = self.vector_store.search(repo_id, query, top_k=top_k)
        if "run_instructions" in intents:
            vector_chunks = self._filter_run_chunks(query, vector_chunks, repo_facts)
        graph_context = self._graph_context(query, graph, intents)
        graph_files = set(graph_context["files"])
        boosted_chunks = self.vector_store.chunks_for_files(repo_id, priority_files | graph_files, limit_per_file=3)
        chunks = self._merge_chunks(boosted_chunks, vector_chunks, top_k + 6)
        source_refs = self._source_refs(chunks)
        return {
            "intents": intents,
            "repo_facts": repo_facts,
            "selected_chunks": [chunk.to_dict() for chunk in chunks],
            "chunks": [chunk.to_dict() for chunk in chunks],
            "selected_graph_facts": graph_context["facts"],
            "graph_facts": graph_context["facts"],
            "graph_paths": graph_context["paths"],
            "weakness_facts": weaknesses if self._needs_weaknesses(intents) else [],
            "relevant_files": sorted({chunk.file_path for chunk in chunks} | priority_files | graph_files),
            "source_references": source_refs,
        }

    def _priority_files(self, query: str, intents: list[str], facts: dict[str, Any], graph: dict[str, Any]) -> set[str]:
        files: set[str] = set(re.findall(r"[\w./-]+\.\w+", query))
        if "repo_overview" in intents:
            files.update(facts.get("readme_paths", []))
            files.update(facts.get("dependency_files", []))
            files.update(item["path"] for item in facts.get("important_files", [])[:10])
        if "run_instructions" in intents:
            files.update(facts.get("run_doc_paths", []))
            files.update(facts.get("install_doc_paths", []))
            files.update(facts.get("official_doc_paths", []))
            files.update(facts.get("readme_paths", []))
            files.update(facts.get("dependency_files", []))
            files.update(facts.get("deployment_files", []))
            files.update(script["path"] for script in facts.get("helper_scripts", []) if script.get("mentioned_in_official_docs"))
            files.update(path for path in facts.get("file_tree", []) if PathName(path).name in RUN_FILES)
        if "architecture" in intents:
            files.update(item["path"] for item in facts.get("main_entrypoints", [])[:8])
            files.update(item["path"] for item in facts.get("important_modules", [])[:12])
            files.update(facts.get("api_route_files", []))
        if "file_explanation" in intents or "function_explanation" in intents or "general_code_question" in intents:
            files.update(self._exact_graph_files(query, graph))
        if "testing_question" in intents:
            files.update(path for path in facts.get("file_tree", []) if "test" in path.lower())
        return {file for file in files if file}

    def _filter_run_chunks(self, query: str, chunks: list[Chunk], facts: dict[str, Any]) -> list[Chunk]:
        lowered = query.lower()
        excluded = {
            helper["path"]
            for helper in facts.get("helper_scripts", [])
            if not helper.get("mentioned_in_official_docs") and helper["path"].lower() not in lowered and helper["path"].rsplit("/", 1)[-1].lower() not in lowered
        }
        official_docs = set(facts.get("official_doc_paths", []))
        install_docs = set(facts.get("install_doc_paths", []))
        allowed_docs = official_docs | install_docs
        filtered = [chunk for chunk in chunks if chunk.file_path not in excluded]
        filtered.sort(key=lambda chunk: (0 if chunk.file_path in install_docs else 1 if chunk.file_path in allowed_docs else 2, -(chunk.score or 0.0)))
        return filtered

    def _graph_context(self, query: str, graph: dict[str, Any], intents: list[str]) -> dict[str, Any]:
        facts = graph.get("facts", [])
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])
        query_terms = {term for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]+", query.lower()) if len(term) > 2}
        scored_facts: list[tuple[int, str]] = []
        scored_nodes: list[tuple[int, dict[str, Any]]] = []
        scored_paths: list[tuple[int, str, str | None]] = []

        for fact in facts:
            score = sum(1 for term in query_terms if term in fact.lower())
            if any(intent in {"architecture", "dependency_question", "function_explanation", "file_explanation"} for intent in intents):
                score += 1
            if score > 0:
                scored_facts.append((score, fact))

        for node in nodes:
            haystack = " ".join(str(node.get(key, "")) for key in ("id", "label", "file_path", "kind")).lower()
            score = sum(1 for term in query_terms if term in haystack)
            if score > 0:
                scored_nodes.append((score, node))

        relation_weights = self._relation_weights(intents)
        for edge in edges:
            relation = edge.get("relation", "")
            source = str(edge.get("source", ""))
            target = str(edge.get("target", ""))
            haystack = f"{source} {target} {relation}".lower()
            score = relation_weights.get(relation, 0) + sum(1 for term in query_terms if term in haystack)
            if score <= 0:
                continue
            scored_paths.append((score, self._plain_edge(edge), self._file_from_edge(edge)))

        scored_facts.sort(key=lambda item: item[0], reverse=True)
        scored_nodes.sort(key=lambda item: item[0], reverse=True)
        scored_paths.sort(key=lambda item: item[0], reverse=True)
        files = set()
        for _, node in scored_nodes[:10]:
            if node.get("kind") == "file":
                files.add(str(node.get("label")))
            if node.get("file_path"):
                files.add(str(node["file_path"]))
        for _, _, path in scored_paths[:12]:
            if path:
                files.add(path)
        return {
            "facts": [fact for _, fact in scored_facts[:8]],
            "paths": [path for _, path, _ in scored_paths[:8]],
            "nodes": [node for _, node in scored_nodes[:8]],
            "files": sorted(files),
        }

    def _relation_weights(self, intents: list[str]) -> dict[str, int]:
        weights = {"declares_route": 1, "has_entrypoint": 1}
        if "dependency_question" in intents:
            weights.update({"imports": 4, "depends_on_package": 4})
        if "architecture" in intents:
            weights.update({"contains": 1, "defines_function": 2, "defines_class": 2, "imports": 3, "has_entrypoint": 4})
        if "function_explanation" in intents:
            weights.update({"defines_function": 4, "defines_class": 3, "calls": 2})
        if "file_explanation" in intents:
            weights.update({"contains": 2, "defines_function": 2, "defines_class": 2})
        return weights

    def _exact_graph_files(self, query: str, graph: dict[str, Any]) -> set[str]:
        lowered = query.lower()
        files = set()
        for node in graph.get("nodes", []):
            label = str(node.get("label", "")).lower()
            file_path = str(node.get("file_path", "")).lower()
            if label and label in lowered and node.get("file_path"):
                files.add(str(node["file_path"]))
            if file_path and file_path in lowered:
                files.add(str(node.get("file_path") or node.get("label")))
        return files

    def _file_from_edge(self, edge: dict[str, Any]) -> str | None:
        for key in ("source", "target"):
            value = str(edge.get(key, ""))
            if value.startswith("file:"):
                return value.removeprefix("file:")
            for part in value.split(":"):
                if re.search(r"\.(py|js|jsx|ts|tsx|md|json|yaml|yml|toml|go|rs|java|sh)$", part):
                    return part
        return None

    def _plain_edge(self, edge: dict[str, Any]) -> str:
        relation = edge.get("relation", "relates_to").replace("_", " ")
        source = str(edge.get("source", "")).split(":", 1)[-1]
        target = str(edge.get("target", "")).split(":", 1)[-1]
        return f"{source} {relation} {target}"

    def _merge_chunks(self, primary: list[Chunk], secondary: list[Chunk], limit: int) -> list[Chunk]:
        merged: list[Chunk] = []
        seen = set()
        for chunk in primary + secondary:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            merged.append(chunk)
            if len(merged) >= limit:
                break
        return merged

    def _needs_weaknesses(self, intents: list[str]) -> bool:
        return any(intent in {"weakness_question", "improvement_question", "security_question"} for intent in intents)

    def _source_refs(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        return [
            {
                "file_path": chunk.file_path,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "excerpt": self._redact(chunk.text[:800]),
            }
            for chunk in chunks
        ]

    def _redact(self, text: str) -> str:
        redacted = text
        for pattern in SECRET_PATTERNS:
            redacted = pattern.sub("<redacted-secret>", redacted)
        return redacted


class PathName(str):
    @property
    def name(self) -> str:
        return self.rsplit("/", 1)[-1].lower()
