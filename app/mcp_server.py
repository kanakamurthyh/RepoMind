"""RepoMind as an MCP server: expose repository intelligence tools to Claude Code, Cursor, etc.

Run over stdio (the default for local MCP clients):
    python -m app.mcp_server

Register with Claude Code:
    claude mcp add repomind -- /path/to/.venv/bin/python -m app.mcp_server
"""

from __future__ import annotations

from typing import Any

from app.api.services import ServiceContainer
from app.utils.redaction import redacted_output

try:  # MCP Python SDK v2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # pragma: no cover - v1 fallback
    try:
        from mcp.server.fastmcp import FastMCP as _Server  # type: ignore[no-redef]
    except ImportError:
        _Server = None  # type: ignore[assignment,misc]


def build_server(services: ServiceContainer | None = None) -> Any:
    if _Server is None:
        raise RuntimeError("The MCP SDK is not installed. Run: pip install -r requirements-advanced.txt")
    services = services or ServiceContainer()
    server = _Server(
        "repomind",
        instructions=(
            "RepoMind analyzes public GitHub repositories. Call analyze_repository first, then use the returned repo_id "
            "with search_code, ask_repository, investigate, find_symbol and read_file. Answers cite file:line evidence."
        ),
    )

    @server.tool(description="List repositories that have already been analyzed.")
    @redacted_output
    def list_repositories() -> list[dict[str, Any]]:
        return services.analyzer.list_repos()

    @server.tool(description="Clone (or reuse) and analyze a public GitHub repository. Returns repo_id, summary headline and top findings.")
    @redacted_output
    def analyze_repository(repo_url: str, force: bool = False) -> dict[str, Any]:
        result = services.analyzer.analyze(repo_url, force=force)
        summary = result.summary
        return {
            "repo_id": result.repo_id,
            "commit_sha": result.commit_sha,
            "summary": summary.get("project_identity", {}),
            "health_score": summary.get("quality_score", {}).get("total"),
            "central_files": summary.get("architecture", {}).get("central_files", [])[:8],
            "top_findings": [f"{w.severity}: {w.title} ({w.file_path or 'repo'})" for w in result.weaknesses[:10]],
        }

    @server.tool(description="Hybrid code search (semantic + keyword + symbol + graph). Returns ranked snippets with file:line.")
    @redacted_output
    def search_code(repo_id: str, query: str, k: int = 8) -> list[dict[str, Any]]:
        context = services.chat.retrieve(repo_id, query, top_k=max(1, min(k, 20)))
        return context["source_references"]

    @server.tool(description="Ask a question about a repository; returns a grounded answer with verified citations.")
    @redacted_output
    def ask_repository(repo_id: str, question: str) -> dict[str, Any]:
        answer = services.chat.answer(repo_id, question)
        return {key: answer.get(key) for key in ("answer", "citations", "grounding", "snapshot", "evidence", "coverage")}

    @server.tool(description="Multi-step investigation (search, read, follow callers/callees) for complex how/why questions.")
    @redacted_output
    def investigate(repo_id: str, question: str, max_steps: int = 6) -> dict[str, Any]:
        return services.detective.investigate(repo_id, question, max_steps)

    @server.tool(description="Find function/class definitions by name, with their callers and callees.")
    @redacted_output
    def find_symbol(repo_id: str, name: str) -> list[dict[str, Any]]:
        index = services.analyzer.graph_index(repo_id)
        if index is None:
            return []
        return [{**node, "callers": index.callers(node["id"], 10), "callees": index.callees(node["id"], 10)} for node in index.find_symbols(name)]

    @server.tool(description="Read exact lines of a file from an analyzed repository (secrets redacted).")
    @redacted_output
    def read_file(repo_id: str, path: str, start: int = 1, end: int | None = None) -> dict[str, Any]:
        return services.analyzer.read_file(repo_id, path, start, end)

    @server.tool(description="Docs-vs-code drift, dependency CVEs (OSV), git hotspots and code-health insights.")
    @redacted_output
    def repository_insights(repo_id: str) -> dict[str, Any]:
        summary = services.analyzer.load(repo_id).summary
        return {"insights": summary.get("insights", {}), "quality_score": summary.get("quality_score", {})}

    return server


def main() -> None:
    build_server().run()


if __name__ == "__main__":
    main()
