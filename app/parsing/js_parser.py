from __future__ import annotations

import re

from app.models import CodeEntity, FileRecord


IMPORT_RE = re.compile(r"""(?:import\s+.*?\s+from\s+['"]([^'"]+)['"]|require\(['"]([^'"]+)['"]\))""")
FUNCTION_RE = re.compile(r"(?:function\s+([\w$]+)|const\s+([\w$]+)\s*=\s*(?:async\s*)?\(|export\s+function\s+([\w$]+))")
CLASS_RE = re.compile(r"class\s+([\w$]+)(?:\s+extends\s+([\w$.]+))?")
ROUTE_RE = re.compile(r"\b(?:app|router)\.(get|post|put|delete|patch)\(['\"]([^'\"]+)['\"]")


class JavaScriptParser:
    def parse(self, record: FileRecord) -> dict[str, list]:
        if record.language not in {"javascript", "typescript"}:
            return {"entities": [], "imports": [], "routes": []}
        entities: list[CodeEntity] = []
        imports: list[tuple[str, str]] = []
        routes: list[CodeEntity] = []
        lines = record.text.splitlines()

        for index, line in enumerate(lines, start=1):
            for match in IMPORT_RE.finditer(line):
                imports.append((record.path, match.group(1) or match.group(2)))
            function_match = FUNCTION_RE.search(line)
            if function_match:
                name = next(group for group in function_match.groups() if group)
                entities.append(CodeEntity("function", name, record.path, index, index, {}))
            class_match = CLASS_RE.search(line)
            if class_match:
                metadata = {"inherits": [class_match.group(2)] if class_match.group(2) else []}
                entities.append(CodeEntity("class", class_match.group(1), record.path, index, index, metadata))
            route_match = ROUTE_RE.search(line)
            if route_match:
                routes.append(
                    CodeEntity(
                        "route",
                        f"{route_match.group(1).upper()} {route_match.group(2)}",
                        record.path,
                        index,
                        index,
                        {"method": route_match.group(1), "path": route_match.group(2)},
                    )
                )
        return {"entities": entities, "imports": imports, "routes": routes}
