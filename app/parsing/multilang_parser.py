from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from app.core.interfaces import SourceParser
from app.models import CodeEntity, FileRecord
from app.parsing.python_parser import PythonParser

logger = logging.getLogger(__name__)

TREE_SITTER_LANGUAGES = {
    "javascript": "javascript",
    "typescript": "typescript",
    "java": "java",
    "kotlin": "kotlin",
    "go": "go",
    "rust": "rust",
    "c": "c",
    "cpp": "cpp",
    "csharp": "csharp",
    "ruby": "ruby",
    "php": "php",
    "swift": "swift",
    "scala": "scala",
}
FUNCTION_NODE_TYPES = {
    "function_declaration",
    "function_definition",
    "generator_function_declaration",
    "method_definition",
    "method_declaration",
    "constructor_declaration",
    "function_item",
    "method",
    "singleton_method",
}
CLASS_NODE_TYPES = {
    "class_declaration",
    "class_definition",
    "class_specifier",
    "interface_declaration",
    "enum_declaration",
    "struct_item",
    "struct_specifier",
    "trait_item",
    "enum_item",
    "object_declaration",
    "class",
    "module",
    "type_spec",
}
CALL_NODE_TYPES = {"call_expression", "call", "method_invocation", "invocation_expression"}
IMPORT_NODE_TYPES = {"import_statement", "import_declaration", "preproc_include", "use_declaration", "import_spec", "namespace_use_declaration"}
STRING_RE = re.compile(r"""['"`<]([^'"`<>]+)['"`>]""")

IMPORT_PATTERNS = [
    re.compile(r"""^\s*import\s+(?:[\w*{}\s,$]+\s+from\s+)?['"]([^'"]+)['"]"""),
    re.compile(r"""\brequire\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"""\bimport\(\s*['"]([^'"]+)['"]\s*\)"""),
    re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+(?:\.\*)?)\s*;"),
    re.compile(r"^\s*#include\s+[<\"]([^>\"]+)[>\"]"),
    re.compile(r"^\s*use\s+([\w:]+)"),
]
FUNCTION_PATTERNS = {
    "javascript": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*=>|[\w$]+\s*=>))"),
    "typescript": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?(?:function\*?\s+([\w$]+)|(?:const|let|var)\s+([\w$]+)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=]+)?=>|[\w$]+\s*=>))"),
    "java": re.compile(r"^\s*(?:(?:public|private|protected|static|final|abstract|synchronized)\s+)+[\w<>\[\],\s]+\s+(\w+)\s*\([^;]*$"),
    "kotlin": re.compile(r"^\s*(?:[\w]+\s+)*fun\s+(?:<[^>]+>\s*)?(?:[\w.]+\.)?(\w+)\s*\("),
    "go": re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?(\w+)\s*\("),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)"),
    "c": re.compile(r"^[\w*\s]+\s\**(\w+)\s*\([^;]*\)\s*\{?\s*$"),
    "cpp": re.compile(r"^[\w:<>*&\s]+\s[\*&]*([\w:~]+)\s*\([^;]*\)\s*(?:const)?\s*\{?\s*$"),
    "csharp": re.compile(r"^\s*(?:(?:public|private|protected|internal|static|async|virtual|override|abstract)\s+)+[\w<>\[\],\s]+\s+(\w+)\s*\("),
    "ruby": re.compile(r"^\s*def\s+(?:self\.)?([\w?!]+)"),
    "php": re.compile(r"^\s*(?:(?:public|private|protected|static)\s+)*function\s+(\w+)"),
    "swift": re.compile(r"^\s*(?:[\w]+\s+)*func\s+(\w+)"),
    "scala": re.compile(r"^\s*(?:[\w]+\s+)*def\s+(\w+)"),
}
CLASS_PATTERNS = {
    "default": re.compile(r"^\s*(?:export\s+)?(?:default\s+)?(?:abstract\s+|public\s+|final\s+|data\s+|sealed\s+|open\s+)*(?:class|interface|struct|trait|enum|object)\s+([\w$]+)(?:[^{]*?(?:extends|:)\s*([\w$.]+))?"),
    "go": re.compile(r"^\s*type\s+(\w+)\s+(?:struct|interface)\b"),
    "rust": re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:struct|enum|trait)\s+(\w+)"),
}
ROUTE_PATTERNS = [
    re.compile(r"""\b(?:app|router|server|api|routes?)\.(get|post|put|delete|patch|all|options|head)\(\s*['"`]([^'"`]+)['"`]""", re.IGNORECASE),
    re.compile(r"""@(Get|Post|Put|Delete|Patch|Request)Mapping\(\s*(?:value\s*=\s*|path\s*=\s*)?["']([^"']+)["']"""),
    re.compile(r"""\.(GET|POST|PUT|DELETE|PATCH)\(\s*["']([^"']+)["']"""),
    re.compile(r"""\bHandleFunc\(\s*["']([^"']+)["']"""),
    re.compile(r"""@(Get|Post|Put|Delete|Patch)\(\s*['"]([^'"]*)['"]"""),
]
ENV_PATTERNS = [
    re.compile(r"""\bos\.(?:getenv|environ\.get)\(\s*['"]([A-Z][A-Z0-9_]+)['"]"""),
    re.compile(r"""\bos\.environ\[\s*['"]([A-Z][A-Z0-9_]+)['"]\s*\]"""),
    re.compile(r"""\bprocess\.env\.([A-Z][A-Z0-9_]+)\b"""),
    re.compile(r"""\bprocess\.env\[\s*['"]([A-Z][A-Z0-9_]+)['"]\s*\]"""),
    re.compile(r"""\bimport\.meta\.env\.([A-Z][A-Z0-9_]+)\b"""),
    re.compile(r"""\b(?:os\.Getenv|os\.LookupEnv|env::var|System\.getenv|getenv|Environment\.GetEnvironmentVariable)\(\s*["']([A-Z][A-Z0-9_]+)["']"""),
    re.compile(r"""\b(?:_env|_bool_env|_int_env|_secret_env|_env_field|_bool_field|_int_field)\(\s*["']([A-Z][A-Z0-9_]+)["']"""),
]
CALL_RE = re.compile(r"\b([A-Za-z_$][\w$.]*)\s*\(")
KEYWORDS = {"if", "for", "while", "switch", "return", "function", "class", "catch", "new", "typeof", "sizeof", "super", "elif", "def", "fn", "func"}


@dataclass(slots=True)
class ParseBundle:
    data: dict[str, list[Any]]
    backend: str


def empty_parse() -> dict[str, list[Any]]:
    return {
        "entities": [],
        "imports": [],
        "import_details": [],
        "exports": [],
        "routes": [],
        "calls": [],
        "entrypoints": [],
        "env_vars": [],
    }


JS_IMPORT_CLAUSE_RE = re.compile(r"import\s+(?:type\s+)?(?:([\w$]+)\s*,?\s*)?(?:\*\s+as\s+([\w$]+))?\s*(?:\{([^}]*)\})?\s*from", re.S)


def js_import_names(text: str) -> list[str]:
    match = JS_IMPORT_CLAUSE_RE.search(text)
    if not match:
        return []
    names = [name for name in (match.group(1), match.group(2)) if name]
    for part in (match.group(3) or "").split(","):
        part = part.strip()
        if part:
            names.append(part.split(" as ")[-1].strip())
    return names


def extract_env_vars(text: str) -> list[tuple[str, int]]:
    found: list[tuple[str, int]] = []
    for index, line in enumerate(text.splitlines(), start=1):
        for pattern in ENV_PATTERNS:
            for match in pattern.finditer(line):
                found.append((match.group(1), index))
    return found


def extract_routes(record: FileRecord) -> list[CodeEntity]:
    routes: list[CodeEntity] = []
    for index, line in enumerate(record.text.splitlines(), start=1):
        for pattern in ROUTE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            groups = match.groups()
            method, path = (groups[0].upper().replace("REQUEST", "ANY"), groups[1]) if len(groups) == 2 else ("ANY", groups[0])
            if not path.startswith("/") and pattern is not ROUTE_PATTERNS[4]:
                continue
            routes.append(
                CodeEntity("route", f"{method} {path}", record.path, index, index, {"method": method, "path": path, "parser": "regex"})
            )
            break
    return routes


class MultiLanguageParser(SourceParser):
    """Tree-sitter-preferred multi-language parser with Python AST and regex fallbacks."""

    def __init__(self, prefer_tree_sitter: bool = True) -> None:
        self.python_parser = PythonParser()
        self.prefer_tree_sitter = prefer_tree_sitter
        self._parser_cache: dict[str, Any] = {}
        self.tree_sitter_available = self._detect_tree_sitter() if prefer_tree_sitter else False

    def parse(self, record: FileRecord) -> dict[str, list[Any]]:
        return self.parse_with_backend(record).data

    def parse_with_backend(self, record: FileRecord) -> ParseBundle:
        if record.kind != "text" or not record.text.strip() or record.role == "generated":
            return ParseBundle(empty_parse(), "skipped")
        if record.language == "python":
            parsed = self.python_parser.parse(record)
            data = empty_parse()
            data.update(
                {
                    "entities": parsed.entities,
                    "imports": parsed.imports,
                    "import_details": parsed.import_details,
                    "routes": parsed.routes,
                    "calls": parsed.calls,
                    "entrypoints": parsed.entrypoints,
                }
            )
            data["env_vars"] = extract_env_vars(record.text)
            return ParseBundle(data, "parse-failed" if parsed.failed else "python-ast")
        if record.language not in TREE_SITTER_LANGUAGES and record.language not in {"shell"}:
            data = empty_parse()
            data["env_vars"] = extract_env_vars(record.text) if record.role in {"code", "config", "test"} else []
            return ParseBundle(data, "skipped")
        if self.prefer_tree_sitter and self.tree_sitter_available:
            result = self._parse_tree_sitter(record)
            if result is not None:
                result["routes"] = extract_routes(record)
                result["env_vars"] = extract_env_vars(record.text)
                return ParseBundle(result, "tree-sitter")
        data = self._parse_regex(record)
        data["routes"] = extract_routes(record)
        data["env_vars"] = extract_env_vars(record.text)
        return ParseBundle(data, "regex-fallback")

    # --- tree-sitter -----------------------------------------------------------------
    def _detect_tree_sitter(self) -> bool:
        try:
            import tree_sitter  # noqa: F401
            import tree_sitter_language_pack  # noqa: F401
        except ImportError:
            return False
        return True

    def _tree_sitter_language(self, record: FileRecord) -> str | None:
        if record.path.endswith(".tsx"):
            return "tsx"
        return TREE_SITTER_LANGUAGES.get(record.language)

    def _get_tree_sitter_parser(self, language: str) -> Any | None:
        if language in self._parser_cache:
            return self._parser_cache[language]
        try:
            from tree_sitter_language_pack import get_parser  # type: ignore

            parser = get_parser(language)
        except Exception as exc:  # grammar missing or incompatible build
            logger.debug("tree-sitter grammar unavailable for %s: %s", language, exc)
            parser = None
        self._parser_cache[language] = parser
        return parser

    def _parse_tree_sitter(self, record: FileRecord) -> dict[str, list[Any]] | None:
        language = self._tree_sitter_language(record)
        if not language:
            return None
        parser = self._get_tree_sitter_parser(language)
        if parser is None:
            return None
        source = record.text.encode("utf-8", errors="replace")
        try:
            tree = parser.parse(source)
        except Exception as exc:
            logger.warning("tree-sitter failed on %s: %s", record.path, exc)
            return None
        data = empty_parse()
        # Iterative traversal: (node, class scope, enclosing function qualname)
        stack: list[tuple[Any, tuple[str, ...], str]] = [(tree.root_node, (), "<module>")]
        while stack:
            node, scope, caller = stack.pop()
            child_scope, child_caller = scope, caller
            node_type = node.type
            line = int(node.start_point[0]) + 1
            end_line = int(node.end_point[0]) + 1
            if node_type in IMPORT_NODE_TYPES:
                names = js_import_names(self._node_text(node, source)) if node_type == "import_statement" else []
                for module in self._import_modules(node, source):
                    data["imports"].append((record.path, module))
                    data["import_details"].append({"module": module, "names": names, "level": 0, "line": line})
            elif node_type in FUNCTION_NODE_TYPES or (
                node_type == "variable_declarator" and any(c.type in {"arrow_function", "function", "function_expression"} for c in node.children)
            ):
                name = self._node_name(node, source)
                if name:
                    qualname = ".".join([*scope, name])
                    data["entities"].append(
                        CodeEntity("function", name, record.path, line, end_line, {"qualname": qualname, "parent": ".".join(scope) or None, "parser": "tree-sitter"})
                    )
                    child_caller = qualname
                    child_scope = (*scope, name)
            elif node_type in CLASS_NODE_TYPES and node.child_count > 0:
                name = self._node_name(node, source)
                if name:
                    qualname = ".".join([*scope, name])
                    data["entities"].append(
                        CodeEntity(
                            "class", name, record.path, line, end_line,
                            {"qualname": qualname, "inherits": self._inheritance(node, source), "parent": ".".join(scope) or None, "parser": "tree-sitter"},
                        )
                    )
                    child_scope = (*scope, name)
            elif node_type in CALL_NODE_TYPES:
                callee = self._call_name(node, source)
                if callee:
                    if callee in {"require", "import"}:
                        module = self._first_string(node, source)
                        if module:
                            data["imports"].append((record.path, module))
                            data["import_details"].append({"module": module, "names": [], "level": 0, "line": line})
                    else:
                        data["calls"].append((caller, callee, line))
            elif node_type in {"export_statement", "export_declaration"}:
                exported = self._node_name(node, source)
                if exported:
                    data["exports"].append((record.path, exported, line))
            for child in reversed(node.children):
                stack.append((child, child_scope, child_caller))
        data["entities"].sort(key=lambda entity: entity.start_line)
        return data

    def _node_text(self, node: Any, source: bytes) -> str:
        return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

    def _import_modules(self, node: Any, source: bytes) -> list[str]:
        source_field = node.child_by_field_name("source") or node.child_by_field_name("path")
        if source_field is not None:
            match = STRING_RE.search(self._node_text(source_field, source))
            if match:
                return [match.group(1)]
        text = self._node_text(node, source)
        if node.type in {"preproc_include", "import_spec"}:
            match = STRING_RE.search(text)
            return [match.group(1)] if match else []
        if node.type in {"import_declaration", "use_declaration", "namespace_use_declaration"}:
            inner = re.sub(r"^\s*(?:import|use)\s+(?:static\s+)?", "", text).strip().rstrip(";").strip()
            if inner.startswith("("):  # Go grouped imports are handled via import_spec children
                return []
            return [inner[:200]] if inner else []
        match = STRING_RE.search(text)
        return [match.group(1)] if match else []

    def _first_string(self, node: Any, source: bytes) -> str | None:
        match = STRING_RE.search(self._node_text(node, source))
        return match.group(1) if match else None

    def _inheritance(self, node: Any, source: bytes) -> list[str]:
        header = self._node_text(node, source).split("{", 1)[0][:400]
        inherits: list[str] = []
        for pattern in (r"\bextends\s+([A-Za-z_$][\w$.]*)", r"\bimplements\s+([A-Za-z_$][\w$.]*)", r"class\s+\w+\s*[:(]\s*(?:public\s+|private\s+)?([A-Za-z_][\w.:]*)"):
            for match in re.finditer(pattern, header):
                inherits.append(match.group(1))
        return inherits[:10]

    def _node_name(self, node: Any, source: bytes) -> str | None:
        named = node.child_by_field_name("name")
        if named is not None:
            return self._clean_identifier(self._node_text(named, source))
        for child in node.children:
            if child.type in {"identifier", "type_identifier", "property_identifier", "field_identifier", "constant", "simple_identifier"}:
                return self._clean_identifier(self._node_text(child, source))
        declarator = node.child_by_field_name("declarator")
        if declarator is not None:
            return self._node_name(declarator, source)
        for child in node.children:
            if child.type in {"lexical_declaration", "variable_declaration", "function_declaration", "class_declaration"}:
                return self._node_name(child, source)
            if child.type == "variable_declarator":
                return self._node_name(child, source)
        return None

    def _call_name(self, node: Any, source: bytes) -> str | None:
        function = node.child_by_field_name("function") or node.child_by_field_name("name") or node.child_by_field_name("method")
        if function is None and node.children:
            function = node.children[0]
        if function is None:
            return None
        text = self._node_text(function, source).replace("?.", ".").replace("::", ".").replace("->", ".")
        text = re.sub(r"\s+", "", text)
        match = re.match(r"[A-Za-z_$][\w$.]*", text)
        return match.group(0)[:120] if match else None

    def _clean_identifier(self, value: str) -> str | None:
        match = re.search(r"[A-Za-z_$][\w$]*", value.strip())
        return match.group(0) if match else None

    # --- regex fallback ---------------------------------------------------------------
    def _parse_regex(self, record: FileRecord) -> dict[str, list[Any]]:
        data = empty_parse()
        language = record.language
        function_pattern = FUNCTION_PATTERNS.get(language)
        class_pattern = CLASS_PATTERNS.get(language, CLASS_PATTERNS["default"])
        current_function = "<module>"
        for index, line in enumerate(record.text.splitlines(), start=1):
            for pattern in IMPORT_PATTERNS:
                for match in pattern.finditer(line):
                    module = next((group for group in match.groups() if group), None)
                    if module:
                        data["imports"].append((record.path, module))
                        data["import_details"].append({"module": module, "names": [], "level": 0, "line": index})
            if function_pattern:
                match = function_pattern.search(line)
                if match:
                    name = next((group for group in match.groups() if group), None)
                    if name and name not in KEYWORDS:
                        current_function = name
                        data["entities"].append(CodeEntity("function", name, record.path, index, index, {"qualname": name, "parser": "regex"}))
            if language != "shell":
                match = class_pattern.search(line)
                if match:
                    inherits = [match.group(2)] if len(match.groups()) > 1 and match.group(2) else []
                    data["entities"].append(CodeEntity("class", match.group(1), record.path, index, index, {"qualname": match.group(1), "inherits": inherits, "parser": "regex"}))
            for call in CALL_RE.finditer(line):
                callee = call.group(1)
                if callee.split(".")[-1] not in KEYWORDS and callee != current_function:
                    data["calls"].append((current_function, callee, index))
        return data
