from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from typing import Any

from app.models import CodeEntity, FileRecord

logger = logging.getLogger(__name__)

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "head", "options", "route", "api_route", "websocket"}
BRANCH_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.ExceptHandler, ast.IfExp, ast.comprehension, ast.Assert)


@dataclass(slots=True)
class PythonParseResult:
    failed: bool = False
    entities: list[CodeEntity] = field(default_factory=list)
    imports: list[tuple[str, str]] = field(default_factory=list)
    import_details: list[dict[str, Any]] = field(default_factory=list)
    calls: list[tuple[str, str, int]] = field(default_factory=list)
    routes: list[CodeEntity] = field(default_factory=list)
    entrypoints: list[CodeEntity] = field(default_factory=list)


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value = dotted_name(node.value)
        return f"{value}.{node.attr}" if value else node.attr
    if isinstance(node, ast.Call):
        return dotted_name(node.func)
    if isinstance(node, ast.Subscript):
        return dotted_name(node.value)
    return ""


def decorator_label(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        func = dotted_name(node.func)
        args = [repr(arg.value) for arg in node.args if isinstance(arg, ast.Constant)]
        return f"{func}({', '.join(args)})" if args else func
    return dotted_name(node)


def cyclomatic_complexity(node: ast.AST) -> int:
    complexity = 1
    for child in ast.walk(node):
        if isinstance(child, BRANCH_NODES):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(1, len(child.values) - 1)
        elif isinstance(child, ast.match_case):
            complexity += 1
    return complexity


class PythonParser:
    def parse(self, record: FileRecord) -> PythonParseResult:
        result = PythonParseResult()
        if record.language != "python":
            return result
        try:
            tree = ast.parse(record.text)
        except (SyntaxError, ValueError) as exc:
            logger.warning("Python AST parse failed for %s: %s", record.path, type(exc).__name__)
            result.failed = True
            return result
        visitor = _PythonVisitor(record, result)
        visitor.visit(tree)
        if self._has_main_guard(tree):
            result.entrypoints.append(
                CodeEntity(
                    kind="entrypoint",
                    name="__main__",
                    file_path=record.path,
                    start_line=1,
                    end_line=max(1, record.line_count),
                    metadata={"pattern": 'if __name__ == "__main__"'},
                )
            )
        return result

    def _has_main_guard(self, tree: ast.Module) -> bool:
        for node in tree.body:
            if not isinstance(node, ast.If):
                continue
            test = node.test
            if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name) and test.left.id == "__name__":
                if any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators):
                    return True
        return False


class _PythonVisitor(ast.NodeVisitor):
    def __init__(self, record: FileRecord, result: PythonParseResult) -> None:
        self.record = record
        self.result = result
        self.scope: list[str] = []
        self.function_stack: list[str] = []

    # --- imports -----------------------------------------------------------------
    def visit_Import(self, node: ast.Import) -> Any:
        for alias in node.names:
            self.result.imports.append((self.record.path, alias.name))
            self.result.import_details.append(
                {"module": alias.name, "names": [], "level": 0, "line": node.lineno, "alias": alias.asname}
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Any:
        module = node.module or ""
        prefix = "." * (node.level or 0)
        self.result.imports.append((self.record.path, f"{prefix}{module}" or prefix))
        self.result.import_details.append(
            {
                "module": module,
                "names": [alias.name for alias in node.names],
                "aliases": {alias.name: alias.asname for alias in node.names if alias.asname},
                "level": node.level or 0,
                "line": node.lineno,
            }
        )

    # --- definitions -------------------------------------------------------------
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        qualname = ".".join([*self.scope, node.name])
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        self.result.entities.append(
            CodeEntity(
                kind="class",
                name=node.name,
                file_path=self.record.path,
                start_line=start,
                end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                metadata={
                    "qualname": qualname,
                    "inherits": [name for name in (dotted_name(base) for base in node.bases) if name],
                    "decorators": [decorator_label(d) for d in node.decorator_list],
                    "docstring": (ast.get_docstring(node) or "").strip().split("\n")[0][:200],
                    "parent": ".".join(self.scope) or None,
                    "parser": "python-ast",
                },
            )
        )
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> Any:
        qualname = ".".join([*self.scope, node.name])
        decorators = [decorator_label(d) for d in node.decorator_list]
        start = min([node.lineno, *[d.lineno for d in node.decorator_list]])
        end = getattr(node, "end_lineno", node.lineno) or node.lineno
        args = [arg.arg for arg in node.args.args]
        self.result.entities.append(
            CodeEntity(
                kind="function",
                name=node.name,
                file_path=self.record.path,
                start_line=start,
                end_line=end,
                metadata={
                    "qualname": qualname,
                    "async": isinstance(node, ast.AsyncFunctionDef),
                    "decorators": decorators,
                    "signature": f"{node.name}({', '.join(args)})",
                    "docstring": (ast.get_docstring(node) or "").strip().split("\n")[0][:200],
                    "complexity": cyclomatic_complexity(node),
                    "parent": ".".join(self.scope) or None,
                    "parser": "python-ast",
                },
            )
        )
        for decorator in node.decorator_list:
            route = self._route(decorator)
            if route:
                method, path = route
                self.result.routes.append(
                    CodeEntity(
                        kind="route",
                        name=f"{method} {path}",
                        file_path=self.record.path,
                        start_line=start,
                        end_line=end,
                        metadata={"handler": qualname, "method": method, "path": path, "decorator": decorator_label(decorator)},
                    )
                )
        self.scope.append(node.name)
        self.function_stack.append(qualname)
        self.generic_visit(node)
        self.function_stack.pop()
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> Any:
        name = dotted_name(node.func)
        if name:
            caller = self.function_stack[-1] if self.function_stack else "<module>"
            self.result.calls.append((caller, name, getattr(node, "lineno", 1)))
            # Django-style url patterns: path("users/", views.users)
            if name in {"path", "re_path", "url"} and self.record.path.endswith("urls.py") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    handler = dotted_name(node.args[1]) if len(node.args) > 1 else ""
                    self.result.routes.append(
                        CodeEntity(
                            kind="route",
                            name=f"ANY /{first.value.lstrip('/')}",
                            file_path=self.record.path,
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno) or node.lineno,
                            metadata={"handler": handler, "method": "ANY", "path": "/" + first.value.lstrip("/")},
                        )
                    )
        self.generic_visit(node)

    def _route(self, decorator: ast.AST) -> tuple[str, str] | None:
        if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
            return None
        attr = decorator.func.attr.lower()
        if attr not in HTTP_METHODS:
            return None
        path = None
        if decorator.args and isinstance(decorator.args[0], ast.Constant) and isinstance(decorator.args[0].value, str):
            path = decorator.args[0].value
        for keyword in decorator.keywords:
            if keyword.arg in {"path", "rule"} and isinstance(keyword.value, ast.Constant):
                path = str(keyword.value.value)
        if path is None:
            return None
        method = attr.upper()
        if attr in {"route", "api_route"}:
            method = "ANY"
            for keyword in decorator.keywords:
                if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                    methods = [elt.value for elt in keyword.value.elts if isinstance(elt, ast.Constant)]
                    if methods:
                        method = "|".join(str(m).upper() for m in methods)
        return method, path
