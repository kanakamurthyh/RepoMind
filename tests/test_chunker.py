from __future__ import annotations

from app.models import FileRecord
from app.parsing.chunker import Chunker


def make_record(text: str, language: str = "python", role: str = "code") -> FileRecord:
    return FileRecord(
        repo_id="demo",
        path="app.py",
        abs_path="/tmp/app.py",
        language=language,
        role=role,
        size_bytes=len(text),
        line_count=text.count("\n") + 1,
        text=text,
    )


def test_python_chunker_uses_functions_and_classes():
    record = make_record(
        """
import os

class Service:
    def run(self):
        return os.getcwd()

def main():
    return Service().run()
""".strip()
    )

    chunks = Chunker(max_lines=20).chunk_file(record)

    assert any("class Service" in chunk.text for chunk in chunks)
    assert any("def main" in chunk.text for chunk in chunks)
    assert all(chunk.start_line <= chunk.end_line for chunk in chunks)


def test_markdown_chunker_splits_on_headings():
    record = make_record("# One\nText\n\n## Two\nMore text", language="markdown", role="docs")
    record.path = "README.md"

    chunks = Chunker(max_lines=20).chunk_file(record)

    assert len(chunks) == 2
    assert chunks[0].start_line == 1
    assert chunks[1].start_line == 4


def test_javascript_chunker_uses_top_level_symbols():
    record = make_record(
        "import express from 'express'\n\nclass Server {}\n\nexport function start() {\n  return new Server()\n}\n",
        language="javascript",
        role="code",
    )
    record.path = "server.js"

    chunks = Chunker(max_lines=20).chunk_file(record)

    assert any("class Server" in chunk.text for chunk in chunks)
    assert any("function start" in chunk.text for chunk in chunks)
