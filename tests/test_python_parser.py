from __future__ import annotations

from app.models import FileRecord
from app.parsing.python_parser import PythonParser


def test_python_parser_extracts_imports_entities_routes_and_entrypoints():
    text = """
from fastapi import FastAPI
import os

app = FastAPI()

class Worker(BaseWorker):
    pass

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    health()
""".strip()
    record = FileRecord("demo", "main.py", "/tmp/main.py", "python", "code", len(text), text.count("\n") + 1, text)

    parsed = PythonParser().parse(record)

    assert ("main.py", "fastapi.FastAPI") in parsed.imports
    assert ("main.py", "os") in parsed.imports
    assert any(entity.kind == "class" and entity.name == "Worker" for entity in parsed.entities)
    assert any(entity.kind == "function" and entity.name == "health" for entity in parsed.entities)
    assert parsed.routes
    assert parsed.entrypoints
    assert any(call[1] == "health" for call in parsed.calls)
