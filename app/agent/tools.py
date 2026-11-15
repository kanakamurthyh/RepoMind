"""Repository tools shared by the investigation agent, the API and the MCP server."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any

from app.graph.graph_store import GraphIndex
from app.rag.context import visible_lines
from app.utils.redaction import redact, redact_data

logger = logging.getLogger(__name__)

MAX_OBSERVATION_CHARS = 3500
MAX_GREP_PATTERN_CHARS = 200
MAX_GREP_LINE_CHARS = 2000  # only the head of very long (e.g. minified) lines is searched
MAX_GREP_RESULTS = 100
GREP_REGEX_TIMEOUT_SECONDS = 3.0

# Regex mode runs in a throwaway interpreter so catastrophic backtracking (ReDoS) is killed by a
# hard wall-clock timeout instead of pinning an API worker thread. Stdlib only, isolated mode.
_REGEX_WORKER = """
import json, re, sys
payload = json.load(sys.stdin)
regex = re.compile(payload["pattern"], re.IGNORECASE)
hits = []
for index, line in enumerate(payload["lines"]):
    if regex.search(line):
        hits.append(index)
        if len(hits) >= payload["max_results"]:
            break
json.dump(hits, sys.stdout)
"""


@dataclass(slots=True)
class ToolResult:
    tool: str
    args: dict[str, Any]
    output: Any
    evidence: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"error: {self.error}"
        if isinstance(self.output, list):
            return f"{len(self.output)} result(s)"
        if isinstance(self.output, dict) and "text" in self.output:
            return f"read {self.output.get('path')}:{self.output.get('start_line')}-{self.output.get('end_line')}"
        return "ok"


class RepoTools:
    """Deterministic, read-only tools over one analyzed repository."""

    SPECS: dict[str, str] = {
        "search_code": "search_code(query: str, k: int = 6) - hybrid semantic+keyword+symbol search; returns ranked code/doc snippets.",
        "grep": "grep(pattern: str, path_contains: str = '', regex: bool = False) - case-insensitive literal search over indexed files (regex=true for a Python regex); returns matching lines.",
        "read_file": "read_file(path: str, start: int = 1, end: int = start+80) - read exact lines of a file.",
        "find_symbol": "find_symbol(name: str) - locate function/class definitions by name or Class.method.",
        "callers": "callers(symbol: str) - functions/routes that call this symbol.",
        "callees": "callees(symbol: str) - functions this symbol calls.",
        "call_path": "call_path(source: str, target: str) - shortest call chain between two symbols.",
        "file_neighbors": "file_neighbors(path: str) - files it imports, files importing it, symbols it defines.",
        "list_files": "list_files(prefix: str = '') - repository files under a path prefix.",
    }

    def __init__(self, analyzer: Any, chat_service: Any, repo_id: str) -> None:
        self.analyzer = analyzer
        self.chat_service = chat_service
        self.repo_id = repo_id
        self._graph: GraphIndex | None = None

    @property
    def graph(self) -> GraphIndex | None:
        if self._graph is None:
            self._graph = self.analyzer.graph_index(self.repo_id)
        return self._graph

    def run(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        args = dict(args or {})
        handler = getattr(self, f"_tool_{tool}", None)
        if tool not in self.SPECS or handler is None:
            return ToolResult(tool, args, None, error=f"unknown tool {tool!r}")
        try:
            output, evidence = handler(**args)
        except TypeError as exc:
            return ToolResult(tool, args, None, error=f"bad arguments: {exc}")
        except (FileNotFoundError, ValueError) as exc:
            return ToolResult(tool, args, None, error=str(exc))
        output = redact_data(output)
        if isinstance(output, list):
            while output and len(json.dumps(output, ensure_ascii=False)) > MAX_OBSERVATION_CHARS:
                output.pop()
            evidence = [e for e in evidence if any(
                isinstance(row, dict) and row.get("file_path") == e.get("file_path")
                and row.get("start_line", row.get("line")) == e.get("start_line")
                and row.get("end_line", row.get("line")) == e.get("end_line") for row in output)]
        return ToolResult(tool, redact_data(args), output, evidence)

    # --- tools ----------------------------------------------------------------------------------
    def _tool_search_code(self, query: str, k: int = 6) -> tuple[Any, list[dict[str, Any]]]:
        context = self.chat_service.retrieve(self.repo_id, query, top_k=min(int(k), 10))
        rows = []
        for ref in context.get("source_references", []):
            preview = visible_lines(ref["excerpt"], 400)
            if preview:
                rows.append({"file_path": ref["file_path"], "start_line": ref["start_line"],
                             "end_line": ref["start_line"] + len(preview.splitlines()) - 1,
                             "symbol": ref.get("symbol"), "preview": preview})
        return rows, [self._ev(r["file_path"], r["start_line"], r["end_line"]) for r in rows]

    def _tool_grep(
        self, pattern: str, path_contains: str = "", max_results: int = 30, regex: bool = False
    ) -> tuple[Any, list[dict[str, Any]]]:
        pattern = str(pattern)
        if not pattern:
            raise ValueError("pattern must not be empty")
        if len(pattern) > MAX_GREP_PATTERN_CHARS:
            raise ValueError(f"pattern is longer than {MAX_GREP_PATTERN_CHARS} characters")
        use_regex = regex is True or str(regex).strip().lower() in {"1", "true", "yes"}
        if use_regex:
            try:
                re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(f"invalid regex: {exc}") from exc
        max_results = max(1, min(int(max_results), MAX_GREP_RESULTS))

        seen: set[tuple[str, int]] = set()
        candidates: list[tuple[str, int, str]] = []
        for chunk in self.analyzer.vector_store.all_chunks(self.repo_id):
            if path_contains and path_contains not in chunk.file_path:
                continue
            for offset, line in enumerate(chunk.text.splitlines()):
                if len(line) > MAX_GREP_LINE_CHARS:
                    continue
                number = chunk.start_line + offset
                if (chunk.file_path, number) in seen:
                    continue
                seen.add((chunk.file_path, number))
                candidates.append((chunk.file_path, number, line))

        lines = [line for _, _, line in candidates]
        if use_regex:
            hits = _regex_hits(pattern, lines, max_results)
        else:
            needle = pattern.casefold()
            hits = [index for index, line in enumerate(lines) if needle in line.casefold()][:max_results]
        rows = [
            {"file_path": candidates[i][0], "line": candidates[i][1], "text": redact(candidates[i][2])}
            for i in hits
        ]
        rows.sort(key=lambda row: (row["file_path"], row["line"]))
        return rows, [self._ev(r["file_path"], r["line"], r["line"]) for r in rows]

    def _tool_read_file(self, path: str, start: int = 1, end: int | None = None) -> tuple[Any, list[dict[str, Any]]]:
        start = max(1, int(start))
        end = int(end) if end else start + 80
        result = self.analyzer.read_file(self.repo_id, path, start, min(end, start + 200))
        result["text"] = visible_lines(result["text"], 1800, 1800)
        if not result["text"]:
            raise ValueError("Source line exceeds the tool observation budget; no lines were inspected.")
        result["end_line"] = result["start_line"] + len(result["text"].splitlines()) - 1
        return result, [self._ev(path, result["start_line"], result["end_line"])]

    def _tool_find_symbol(self, name: str) -> tuple[Any, list[dict[str, Any]]]:
        nodes = self.graph.find_symbols(name) if self.graph else []
        rows = [self._symbol_row(node) for node in nodes]
        return rows, [self._ev(r["file_path"], r["start_line"], r["end_line"], "location") for r in rows if r.get("file_path")]

    def _tool_callers(self, symbol: str) -> tuple[Any, list[dict[str, Any]]]:
        return self._relations(symbol, "callers")

    def _tool_callees(self, symbol: str) -> tuple[Any, list[dict[str, Any]]]:
        return self._relations(symbol, "callees")

    def _tool_call_path(self, source: str, target: str) -> tuple[Any, list[dict[str, Any]]]:
        path = self.graph.call_path(source, target) if self.graph else []
        rows = [self._symbol_row(node) for node in path]
        return rows, [self._ev(r["file_path"], r["start_line"], r["end_line"], "location") for r in rows if r.get("file_path")]

    def _tool_file_neighbors(self, path: str) -> tuple[Any, list[dict[str, Any]]]:
        return (self.graph.file_neighbors(path) if self.graph else {}), [self._ev(path, None, None)]

    def _tool_list_files(self, prefix: str = "") -> tuple[Any, list[dict[str, Any]]]:
        files = []
        if self.graph:
            files = sorted(str(a.get("label")) for _, a in self.graph.graph.nodes(data=True) if a.get("kind") == "file" and str(a.get("label", "")).startswith(prefix))
        return files[:200], []

    # --- helpers ------------------------------------------------------------------------------------
    def _relations(self, symbol: str, direction: str) -> tuple[Any, list[dict[str, Any]]]:
        if not self.graph:
            return [], []
        nodes = [symbol] if symbol in self.graph.graph else [n["id"] for n in self.graph.find_symbols(symbol)]
        rows = []
        for node_id in nodes[:3]:
            related = self.graph.callers(node_id) if direction == "callers" else self.graph.callees(node_id)
            for item in related:
                row = self._symbol_row(item)
                row["line"] = item.get("line")
                row["relation"] = item.get("relation")
                row["of"] = self.graph.node(node_id).get("label")
                rows.append(row)
        return rows, [self._ev(r["file_path"], r.get("line") or r["start_line"], r.get("line") or r["end_line"], "location") for r in rows if r.get("file_path")]

    def _symbol_row(self, node: dict[str, Any]) -> dict[str, Any]:
        return {
            "symbol": node.get("label"), "kind": node.get("kind"), "file_path": node.get("file_path"),
            "start_line": node.get("start_line"), "end_line": node.get("end_line"),
            "signature": node.get("signature"), "doc": node.get("docstring"),
        }

    def _ev(self, path: str, start: int | None, end: int | None, kind: str = "source") -> dict[str, Any]:
        return {"file_path": path, "start_line": start, "end_line": end, "kind": kind}


def _regex_hits(pattern: str, lines: list[str], max_results: int) -> list[int]:
    """Indexes of lines matching `pattern`, computed in a subprocess with a hard timeout."""
    if not lines:
        return []
    payload = json.dumps({"pattern": pattern, "lines": lines, "max_results": max_results})
    try:
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _REGEX_WORKER],
            input=payload, capture_output=True, text=True, timeout=GREP_REGEX_TIMEOUT_SECONDS, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("grep regex %r exceeded %.1fs over %d lines; aborted", pattern, GREP_REGEX_TIMEOUT_SECONDS, len(lines))
        raise ValueError(
            f"regex search exceeded {GREP_REGEX_TIMEOUT_SECONDS:g}s (pattern may backtrack catastrophically); "
            "use a literal search or a simpler pattern"
        ) from exc
    except (subprocess.CalledProcessError, OSError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        logger.warning("grep regex worker failed for %r: %s %s", pattern, exc, stderr.strip()[-300:])
        raise ValueError("regex search failed") from exc
    return [int(index) for index in json.loads(result.stdout or "[]")]
