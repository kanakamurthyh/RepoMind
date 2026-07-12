from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.core.interfaces import SourceParser
from app.models import CodeEntity, FileRecord
from app.parsing.python_parser import PythonParser

IMPORT_PATTERNS = [
    re.compile(r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\))"""),
    re.compile(r"^\s*import\s+([\w.*/_-]+)", re.MULTILINE),
    re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]", re.MULTILINE),
    re.compile(r"^\s*use\s+([\w:]+)", re.MULTILINE),
]
LANGUAGE_ALIASES = {
    "javascript": "javascript",
    "typescript": "typescript",
    "jsx": "javascript",
    "tsx": "tsx",
    "java": "java",
    "c": "c",
    "cpp": "cpp",
    "go": "go",
    "rust": "rust",
}
FUNCTION_NODE_TYPES = {
    "function_declaration",
    "function_definition",
    "method_definition",
    "method_declaration",
    "function_item",
}
CLASS_NODE_TYPES = {
    "class_declaration",
    "class_definition",
    "interface_declaration",
    "struct_item",
    "struct_specifier",
}
CALL_NODE_TYPES = {"call_expression"}
IMPORT_NODE_TYPES = {"import_statement", "import_declaration", "preproc_include", "use_declaration"}
VARIABLE_FUNCTION_NODE_TYPES = {"variable_declarator"}
FUNCTION_PATTERNS = {
    "javascript": re.compile(r"(?:function\s+([\w$]+)|const\s+([\w$]+)\s*=\s*(?:async\s*)?\(|export\s+function\s+([\w$]+))"),
    "typescript": re.compile(r"(?:function\s+([\w$]+)|const\s+([\w$]+)\s*=\s*(?:async\s*)?\(|export\s+function\s+([\w$]+))"),
    "java": re.compile(r"\b(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\([^)]*\)\s*\{"),
    "go": re.compile(r"\bfunc\s+(?:\([^)]*\)\s*)?(\w+)\s*\("),
    "rust": re.compile(r"\bfn\s+(\w+)\s*\("),
    "c": re.compile(r"^\s*[\w*\s]+\s+(\w+)\s*\([^;]*\)\s*\{", re.MULTILINE),
    "cpp": re.compile(r"^\s*[\w:<>*&\s]+\s+(\w+)\s*\([^;]*\)\s*\{", re.MULTILINE),
}
CLASS_PATTERNS = {
    "javascript": re.compile(r"class\s+([\w$]+)(?:\s+extends\s+([\w$.]+))?"),
    "typescript": re.compile(r"class\s+([\w$]+)(?:\s+extends\s+([\w$.]+))?"),
    "java": re.compile(r"\bclass\s+(\w+)(?:\s+extends\s+(\w+))?"),
    "cpp": re.compile(r"\bclass\s+(\w+)(?:\s*:\s*(?:public|private|protected)?\s*(\w+))?"),
    "rust": re.compile(r"\bstruct\s+(\w+)"),
}
ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]")
CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


@dataclass(slots=True)
class ParseBundle:
    data: dict[str, list[Any]]
    backend: str


class MultiLanguageParser(SourceParser):
    """Tree-sitter-preferred multi-language parser with robust fallbacks."""

    def __init__(self, prefer_tree_sitter: bool = True) -> None:
        self.python_parser = PythonParser()
        self.prefer_tree_sitter = prefer_tree_sitter
        self._parser_cache: dict[str, Any] = {}
        self.tree_sitter_available = self._detect_tree_sitter()

    def parse(self, record: FileRecord) -> dict[str, list[Any]]:
        return self.parse_with_backend(record).data

    def parse_with_backend(self, record: FileRecord) -> ParseBundle:
        if record.kind != "text" or not record.text.strip():
            return ParseBundle(self._empty(), "skipped")
        if record.language == "python":
            parsed = self.python_parser.parse(record)
            return ParseBundle(
                {
                    "entities": parsed.entities,
                    "imports": parsed.imports,
                    "routes": parsed.routes,
                    "calls": parsed.calls,
                    "entrypoints": parsed.entrypoints,
                },
                "python-ast",
            )
        if self.prefer_tree_sitter and self.tree_sitter_available:
            tree_sitter_result = self._parse_tree_sitter(record)
            if tree_sitter_result is not None:
                return ParseBundle(tree_sitter_result, "tree-sitter")
        return ParseBundle(self._parse_regex(record), "regex-fallback")

    def _detect_tree_sitter(self) -> bool:
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            return False
        return True

    def _parse_tree_sitter(self, record: FileRecord) -> dict[str, list[Any]] | None:
        language = self._tree_sitter_language(record)
        if not language:
            return None
        parser = self._get_tree_sitter_parser(language)
        if parser is None:
            return None
        source = record.text.encode("utf-8", errors="ignore")
        try:
            tree = parser.parse(source)
        except Exception:
            return None
        data = self._empty()
        self._walk_tree_sitter(tree.root_node, source, record, data)
        return data

    def _get_tree_sitter_parser(self, language: str) -> Any | None:
        if language in self._parser_cache:
            return self._parser_cache[language]
        try:
            from tree_sitter_language_pack import get_parser  # type: ignore

            parser = get_parser(language)
        except Exception:
            return None
        self._parser_cache[language] = parser
        return parser

    def _walk_tree_sitter(self, node: Any, source: bytes, record: FileRecord, data: dict[str, list[Any]]) -> None:
        line = int(node.start_point[0]) + 1
        if node.type in IMPORT_NODE_TYPES:
            for module in self._extract_import_modules(node, source):
                data["imports"].append((record.path, module))
        elif node.type in FUNCTION_NODE_TYPES:
            name = self._node_name(node, source)
            if name:
                data["entities"].append(CodeEntity("function", name, record.path, line, int(node.end_point[0]) + 1, {"parser": "tree-sitter"}))
        elif node.type in VARIABLE_FUNCTION_NODE_TYPES and any(child.type in {"arrow_function", "function"} for child in node.children):
            name = self._node_name(node, source)
            if name:
                data["entities"].append(CodeEntity("function", name, record.path, line, int(node.end_point[0]) + 1, {"parser": "tree-sitter"}))
        elif node.type in CLASS_NODE_TYPES:
            name = self._node_name(node, source)
            if name:
                data["entities"].append(
                    CodeEntity(
                        "class",
                        name,
                        record.path,
                        line,
                        int(node.end_point[0]) + 1,
                        {"inherits": self._extract_inheritance(node, source), "parser": "tree-sitter"},
                    )
                )
        elif node.type in CALL_NODE_TYPES:
            callee = self._call_name(node, source)
            if callee:
                data["calls"].append(("<module>", callee, line))
        elif node.type in {"export_statement", "export_declaration"}:
            exported = self._node_name(node, source)
            if exported:
                data["exports"].append((record.path, exported, line))

        for child in node.children:
            self._walk_tree_sitter(child, source, record, data)

    def _extract_import_modules(self, node: Any, source: bytes) -> list[str]:
        text = self._node_text(node, source)
        modules = []
        for pattern in IMPORT_PATTERNS:
            for match in pattern.finditer(text):
                module = next((group for group in match.groups() if group), None)
                if module:
                    modules.append(module)
        if not modules and node.type == "preproc_include":
            include = re.search(r"[<\"]([^>\"]+)[>\"]", text)
            if include:
                modules.append(include.group(1))
        return modules[:20]

    def _extract_inheritance(self, node: Any, source: bytes) -> list[str]:
        text = self._node_text(node, source)
        inherits: list[str] = []
        for pattern in CLASS_PATTERNS.values():
            match = pattern.search(text)
            if match and len(match.groups()) > 1 and match.group(2):
                inherits.append(match.group(2))
                break
        if " extends " in text and not inherits:
            match = re.search(r"\bextends\s+([A-Za-z_][\w.]*)", text)
            if match:
                inherits.append(match.group(1))
        return inherits[:10]

    def _node_name(self, node: Any, source: bytes) -> str | None:
        named = node.child_by_field_name("name")
        if named is not None:
            return self._clean_identifier(self._node_text(named, source))
        for child in node.children:
            if child.type in {"identifier", "type_identifier", "property_identifier", "field_identifier"}:
                return self._clean_identifier(self._node_text(child, source))
        match = re.search(r"\b(?:function|func|fn|class|struct|interface)\s+([A-Za-z_][\w$]*)", self._node_text(node, source))
        return match.group(1) if match else None

    def _call_name(self, node: Any, source: bytes) -> str | None:
        function = node.child_by_field_name("function")
        if function is not None:
            text = self._node_text(function, source)
        elif node.children:
            text = self._node_text(node.children[0], source)
        else:
            return None
        name = text.rsplit(".", 1)[-1].rsplit("::", 1)[-1].strip()
        return self._clean_identifier(name)

    def _node_text(self, node: Any, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")

    def _clean_identifier(self, value: str) -> str | None:
        match = re.search(r"[A-Za-z_][\w$]*", value.strip())
        return match.group(0) if match else None

    def _tree_sitter_language(self, record: FileRecord) -> str | None:
        if record.path.endswith(".jsx"):
            return "javascript"
        if record.path.endswith(".tsx"):
            return "tsx"
        return LANGUAGE_ALIASES.get(self._normalized_language(record))

    def _parse_regex(self, record: FileRecord) -> dict[str, list[Any]]:
        data = self._empty()
        language = self._normalized_language(record)
        lines = record.text.splitlines()
        for index, line in enumerate(lines, start=1):
            for pattern in IMPORT_PATTERNS:
                for match in pattern.finditer(line):
                    module = next((group for group in match.groups() if group), None)
                    if module:
                        data["imports"].append((record.path, module))
            function_pattern = FUNCTION_PATTERNS.get(language)
            if function_pattern:
                match = function_pattern.search(line)
                if match:
                    name = next((group for group in match.groups() if group), None)
                    if name and name not in {"if", "for", "while", "switch"}:
                        data["entities"].append(CodeEntity("function", name, record.path, index, index, {"parser": "regex"}))
            class_pattern = CLASS_PATTERNS.get(language)
            if class_pattern:
                match = class_pattern.search(line)
                if match:
                    inherits = [match.group(2)] if len(match.groups()) > 1 and match.group(2) else []
                    data["entities"].append(CodeEntity("class", match.group(1), record.path, index, index, {"inherits": inherits, "parser": "regex"}))
            route_match = ROUTE_RE.search(line)
            if route_match:
                data["routes"].append(
                    CodeEntity(
                        "route",
                        f"{route_match.group(1).upper()} {route_match.group(2)}",
                        record.path,
                        index,
                        index,
                        {"method": route_match.group(1), "path": route_match.group(2), "parser": "regex"},
                    )
                )
            for call in CALL_RE.finditer(line):
                callee = call.group(1)
                if callee not in {"if", "for", "while", "switch", "return", "function", "class"}:
                    data["calls"].append(("<module>", callee, index))
        if record.path.endswith((".js", ".ts")) and re.search(r"\bnode\b|\bmain\b", record.text[:1000].lower()):
            pass
        return data

    def _normalized_language(self, record: FileRecord) -> str:
        if record.path.endswith((".jsx", ".tsx")):
            return "typescript" if record.path.endswith(".tsx") else "javascript"
        if record.path.endswith((".cc", ".cpp", ".cxx", ".hpp")):
            return "cpp"
        return record.language

    def _empty(self) -> dict[str, list[Any]]:
        return {"entities": [], "imports": [], "exports": [], "routes": [], "calls": [], "entrypoints": []}
