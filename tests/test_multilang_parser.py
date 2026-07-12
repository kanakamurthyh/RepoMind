from __future__ import annotations

from app.models import FileRecord
from app.parsing.multilang_parser import MultiLanguageParser


def rec(path: str, text: str, language: str) -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, "code", len(text), text.count("\n") + 1, text)


def test_multilang_parser_extracts_javascript_symbols_imports_and_calls_with_fallback():
    parser = MultiLanguageParser(prefer_tree_sitter=False)
    parsed = parser.parse_with_backend(
        rec(
            "src/app.js",
            "import { helper } from './utils';\nclass App extends Base {}\nfunction start() { helper(); }\n",
            "javascript",
        )
    )

    assert parsed.backend == "regex-fallback"
    assert ("src/app.js", "./utils") in parsed.data["imports"]
    assert any(entity.kind == "class" and entity.name == "App" for entity in parsed.data["entities"])
    assert any(entity.kind == "function" and entity.name == "start" for entity in parsed.data["entities"])
    assert any(call[1] == "helper" for call in parsed.data["calls"])


def test_multilang_parser_supports_go_and_rust_function_signals():
    parser = MultiLanguageParser(prefer_tree_sitter=False)

    go = parser.parse(rec("main.go", "package main\nfunc main() { run() }\n", "go"))
    rust = parser.parse(rec("src/lib.rs", "pub fn build() { render(); }\nstruct Widget {}\n", "rust"))

    assert any(entity.name == "main" for entity in go["entities"])
    assert any(entity.name == "build" for entity in rust["entities"])
    assert any(entity.kind == "class" and entity.name == "Widget" for entity in rust["entities"])
