from __future__ import annotations

from app.models import FileRecord
from app.parsing.python_parser import PythonParser


def test_python_parser_extracts_imports_entities_routes_and_entrypoints():
    text = """
from fastapi import FastAPI
import os

app = FastAPI()

class Worker(BaseWorker):
    def run(self):
        return self.step()

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    health()
""".strip()
    record = FileRecord("demo", "main.py", "/tmp/main.py", "python", "code", len(text), text.count("\n") + 1, text)
    parsed = PythonParser().parse(record)

    assert ("main.py", "fastapi") in parsed.imports and ("main.py", "os") in parsed.imports
    assert {"module": "fastapi", "names": ["FastAPI"], "aliases": {}, "level": 0, "line": 1} in parsed.import_details
    assert {e.qualname for e in parsed.entities} == {"Worker", "Worker.run", "health"}
    route = parsed.routes[0]
    assert (route.name, route.metadata["handler"]) == ("GET /health", "health")
    assert parsed.entrypoints
    assert ("Worker.run", "self.step", 8) in parsed.calls
    assert ("<module>", "health", 15) in parsed.calls


def test_relative_imports_keep_their_level():
    text = "from . import utils\nfrom ..core.models import Chunk\n"
    record = FileRecord("demo", "pkg/sub/mod.py", "/tmp/x", "python", "code", len(text), 2, text)
    details = PythonParser().parse(record).import_details

    assert details[0] == {"module": "", "names": ["utils"], "aliases": {}, "level": 1, "line": 1}
    assert details[1]["module"] == "core.models" and details[1]["level"] == 2
