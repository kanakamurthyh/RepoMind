from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any

from app.models import CodeEntity, FileRecord


@dataclass(slots=True)
class PythonParseResult:
    entities: list[CodeEntity] = field(default_factory=list)
    imports: list[tuple[str, str]] = field(default_factory=list)
    calls: list[tuple[str, str, int]] = field(default_factory=list)
    routes: list[CodeEntity] = field(default_factory=list)
    entrypoints: list[CodeEntity] = field(default_factory=list)


class PythonParser:
    def parse(self, record: FileRecord) -> PythonParseResult:
        result = PythonParseResult()
        if record.language != "python":
            return result
        try:
            tree = ast.parse(record.text)
        except SyntaxError:
            return result

        parent_stack: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for module in self._import_modules(node):
                    result.imports.append((record.path, module))
            if isinstance(node, ast.ClassDef):
                bases = [self._name(base) for base in node.bases if self._name(base)]
                result.entities.append(
                    CodeEntity(
                        kind="class",
                        name=node.name,
                        file_path=record.path,
                        start_line=node.lineno,
                        end_line=getattr(node, "end_lineno", node.lineno),
                        metadata={"inherits": bases, "decorators": [self._name(d) for d in node.decorator_list]},
                    )
                )
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = [self._name(d) for d in node.decorator_list if self._name(d)]
                entity = CodeEntity(
                    kind="function",
                    name=node.name,
                    file_path=record.path,
                    start_line=node.lineno,
                    end_line=getattr(node, "end_lineno", node.lineno),
                    metadata={"async": isinstance(node, ast.AsyncFunctionDef), "decorators": decorators},
                )
                result.entities.append(entity)
                route_path = self._route_from_decorators(decorators)
                if route_path:
                    result.routes.append(
                        CodeEntity(
                            kind="route",
                            name=route_path,
                            file_path=record.path,
                            start_line=node.lineno,
                            end_line=getattr(node, "end_lineno", node.lineno),
                            metadata={"handler": node.name, "decorators": decorators},
                        )
                    )

        visitor = _CallVisitor(record.path)
        visitor.visit(tree)
        result.calls.extend(visitor.calls)
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

    def _import_modules(self, node: ast.AST) -> list[str]:
        if isinstance(node, ast.Import):
            return [alias.name for alias in node.names]
        if isinstance(node, ast.ImportFrom):
            prefix = "." * node.level + (node.module or "")
            return [f"{prefix}.{alias.name}".strip(".") for alias in node.names]
        return []

    def _route_from_decorators(self, decorators: list[str]) -> str | None:
        for decorator in decorators:
            lowered = decorator.lower()
            if any(method in lowered for method in (".get", ".post", ".put", ".delete", ".patch", ".route")):
                return decorator
        return None

    def _has_main_guard(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.If):
                continue
            source = ast.dump(node.test)
            if "__name__" in source and "__main__" in source:
                return True
        return False

    def _name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            value = self._name(node.value)
            return f"{value}.{node.attr}" if value else node.attr
        if isinstance(node, ast.Call):
            func = self._name(node.func)
            args = []
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    args.append(repr(arg.value))
            return f"{func}({', '.join(args)})" if args else func
        if isinstance(node, ast.Constant):
            return repr(node.value)
        return ""


class _CallVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.current_function = "<module>"
        self.calls: list[tuple[str, str, int]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        previous = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = previous

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self.visit_FunctionDef(node)

    def visit_Call(self, node: ast.Call) -> Any:
        name = PythonParser()._name(node.func)
        if name:
            self.calls.append((self.current_function, name, getattr(node, "lineno", 1)))
        self.generic_visit(node)
