from __future__ import annotations

from app.graph.code_graph import CodeGraphBuilder
from app.models import FileRecord


def rec(path: str, text: str, language: str = "python", role: str = "code") -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def test_graph_builder_extracts_entrypoints_definitions_and_dependencies():
    records = [
        rec("main.py", "import requests\n\ndef main():\n    return requests.get('https://example.com')\n\nif __name__ == '__main__':\n    main()\n"),
        rec("requirements.txt", "requests>=2", "text", "config"),
    ]

    graph = CodeGraphBuilder().build(records)
    relations = {edge["relation"] for edge in graph["edges"]}

    assert "imports" in relations
    assert "defines_function" in relations
    assert "has_entrypoint" in relations
    assert "depends_on_package" in relations
    assert any(item["dependency"] == "requests" for item in graph["top_dependencies"])


def test_graph_builder_resolves_local_javascript_imports_to_file_edges():
    records = [
        rec("src/app.js", "import { helper } from './utils';\nexport function start() { helper(); }\n", "javascript"),
        rec("src/utils.js", "export function helper() { return true; }\n", "javascript"),
    ]

    graph = CodeGraphBuilder().build(records)

    assert any(
        edge["relation"] == "depends_on_file"
        and edge["source"] == "file:src/app.js"
        and edge["target"] == "file:src/utils.js"
        for edge in graph["edges"]
    )
