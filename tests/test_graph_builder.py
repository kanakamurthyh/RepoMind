from __future__ import annotations

from app.graph.code_graph import CodeGraphBuilder
from app.graph.graph_store import GraphIndex, GraphStore
from app.models import FileRecord


def rec(path: str, text: str, language: str = "python", role: str = "code") -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def _edges(graph, relation):
    return {(e["source"], e["target"]) for e in graph["edges"] if e["relation"] == relation}


def test_graph_builder_extracts_entrypoints_definitions_and_dependencies():
    records = [
        rec("main.py", "import requests\n\ndef main():\n    return requests.get('https://example.com')\n\nif __name__ == '__main__':\n    main()\n"),
        rec("requirements.txt", "requests>=2", "text", "config"),
    ]
    graph = CodeGraphBuilder().build(records)
    relations = {edge["relation"] for edge in graph["edges"]}

    assert {"imports", "defines_function", "has_entrypoint", "declares_dependency"} <= relations
    assert any(item["dependency"] == "requests" for item in graph["top_dependencies"])


def test_python_absolute_relative_and_symbol_imports_resolve_to_files():
    graph = CodeGraphBuilder().build([
        rec("app/models.py", "class Chunk:\n    def to_dict(self):\n        pass\n"),
        rec("app/core/chat.py", "from app.models import Chunk\nfrom .helpers import util\n\nclass A:\n    def run(self):\n        self.go()\n        util()\n        Chunk().to_dict()\n\n    def go(self):\n        pass\n\nclass B:\n    def run(self):\n        pass\n"),
        rec("app/core/helpers.py", "from app.core.chat import A\n\ndef util():\n    pass\n"),
    ])
    deps = _edges(graph, "depends_on_file")
    calls = _edges(graph, "calls")

    assert ("file:app/core/chat.py", "file:app/models.py") in deps
    assert ("file:app/core/chat.py", "file:app/core/helpers.py") in deps
    assert ("function:app/core/chat.py:A.run", "function:app/core/chat.py:A.go") in calls
    assert ("function:app/core/chat.py:A.run", "function:app/models.py:Chunk.to_dict") in calls
    assert {"function:app/core/chat.py:A.run", "function:app/core/chat.py:B.run"} <= {n["id"] for n in graph["nodes"]}
    assert graph["dependency_cycles"] == [["app/core/chat.py", "app/core/helpers.py"]]


def test_javascript_imports_and_named_import_calls_resolve():
    graph = CodeGraphBuilder().build([
        rec("src/app.js", "import { helper } from './utils';\nexport function start() { helper(); }\n", "javascript"),
        rec("src/utils.js", "export function helper() { return true; }\n", "javascript"),
    ])

    assert ("file:src/app.js", "file:src/utils.js") in _edges(graph, "depends_on_file")
    assert ("function:src/app.js:start", "function:src/utils.js:helper") in _edges(graph, "calls")


def test_graph_store_round_trip_and_queries(tmp_path):
    builder = CodeGraphBuilder()
    builder.build([
        rec("api.py", "from fastapi import FastAPI\nfrom svc import work\napp = FastAPI()\n\n@app.post('/jobs')\ndef create_job():\n    return work()\n"),
        rec("svc.py", "def work():\n    return helper_step()\n\ndef helper_step():\n    return 1\n"),
    ])
    store = GraphStore(tmp_path)
    store.save("demo", builder.last_graph)
    index = store.load("demo")

    assert isinstance(index, GraphIndex)
    assert [n["label"] for n in index.call_path("create_job", "helper_step")] == ["create_job", "work", "helper_step"]
    assert any(c["kind"] == "route" for c in index.callers("function:api.py:create_job"))
    assert index.file_neighbors("svc.py")["imported_by"] == ["api.py"]
