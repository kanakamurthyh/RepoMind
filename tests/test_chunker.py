from __future__ import annotations

from app.models import FileRecord
from app.parsing.chunker import Chunker
from app.parsing.multilang_parser import MultiLanguageParser


def make_record(text: str, language: str = "python", role: str = "code", path: str = "app.py") -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def _covered(chunks):
    return {line for chunk in chunks for line in range(chunk.start_line, chunk.end_line + 1)}


def test_python_chunker_uses_symbols_without_duplicating_nested_methods():
    record = make_record("import os\n\nclass Service:\n    def run(self):\n        return os.getcwd()\n\ndef main():\n    return Service().run()\n")
    chunks = Chunker(max_lines=20).chunk_file(record)

    assert [c.symbol for c in chunks] == [None, "Service", "main"]
    assert chunks[1].metadata["header"] == "app.py › class Service"
    assert len({c.id for c in chunks}) == len(chunks)


def test_large_class_is_split_per_method_with_full_coverage():
    text = "class Big:\n    x = 1\n" + "".join(f"    def m{i}(self):\n" + "        a = 1\n" * 20 for i in range(8))
    chunks = Chunker().chunk_file(make_record(text))

    assert {c.symbol for c in chunks} >= {"Big", "Big.m0", "Big.m7"}
    assert len(_covered(chunks)) == text.count("\n")


def test_long_javascript_function_is_windowed_not_truncated():
    body = "export function big() {\n" + "".join(f"  step{i}();\n" for i in range(120)) + "}\n"
    record = make_record(body, "javascript", path="big.js")
    entities = MultiLanguageParser(prefer_tree_sitter=False).parse(record)["entities"]
    chunks = Chunker().chunk_file(record, entities)

    assert len(_covered(chunks)) == body.count("\n")


def test_markdown_chunker_splits_on_headings_but_not_inside_code_fences():
    record = make_record("# One\nText\n```bash\n# comment, not a heading\n```\n## Two\nMore text", "markdown", "docs", "README.md")
    chunks = Chunker(max_lines=20).chunk_file(record)

    assert [(c.start_line, c.symbol) for c in chunks] == [(1, "One"), (6, "Two")]
    assert chunks[1].metadata["header"] == "README.md › One › Two"


def test_generated_files_are_not_chunked():
    assert Chunker().chunk_file(make_record('{"lockfileVersion": 3}', "json", "generated", "package-lock.json")) == []
