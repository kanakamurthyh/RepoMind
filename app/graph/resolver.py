"""Resolve import strings to repository files (Python, JS/TS, and path-like languages)."""

from __future__ import annotations

import posixpath
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

from app.models import FileRecord

JS_EXTENSIONS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte")
SOURCE_ROOTS = ("src", "lib", "python", "packages", "backend", "server", "app")


def python_module_name(path: str) -> str | None:
    posix = PurePosixPath(path)
    if posix.suffix not in {".py", ".pyi"}:
        return None
    parts = list(posix.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else None


def external_package(module: str, language: str) -> str:
    clean = module.strip().strip("'\"")
    if language in {"javascript", "typescript"}:
        if clean.startswith("@"):
            return "/".join(clean.split("/")[:2])
        return clean.split("/")[0]
    if language == "python":
        return clean.lstrip(".").split(".")[0]
    if language in {"java", "kotlin", "scala"}:
        return ".".join(clean.split(".")[:2])
    if language == "rust":
        return clean.split("::")[0]
    return clean.split("/")[0]


class ImportResolver:
    def __init__(self, records: list[FileRecord]) -> None:
        self.paths = {record.path for record in records if record.kind == "text"}
        self.language_by_path = {record.path: record.language for record in records}
        self.python_modules: dict[str, str] = {}
        self.python_suffixes: dict[str, set[str]] = defaultdict(set)
        self.stem_index: dict[str, set[str]] = defaultdict(set)
        for path in sorted(self.paths):
            module = python_module_name(path)
            if module:
                self.python_modules.setdefault(module, path)
                parts = module.split(".")
                for index in range(1, len(parts)):
                    suffix = ".".join(parts[index:])
                    self.python_suffixes[suffix].add(path)
            self.stem_index[str(PurePosixPath(path).with_suffix(""))].add(path)

    # --- python -----------------------------------------------------------------------
    def resolve_python(self, importer: str, detail: dict[str, Any]) -> list[tuple[str, str | None]]:
        """Return [(target_file, imported_symbol_or_None)] for one import statement."""
        module = detail.get("module") or ""
        names = detail.get("names") or []
        level = int(detail.get("level") or 0)
        if level > 0:
            package = list(PurePosixPath(importer).parent.parts)
            if level > 1:
                package = package[: max(0, len(package) - (level - 1))]
            base = ".".join([*package, *([module] if module else [])])
            return self._resolve_names(base, names, importer, exact_only=True)
        return self._resolve_names(module, names, importer, exact_only=False)

    def _resolve_names(self, base: str, names: list[str], importer: str, exact_only: bool) -> list[tuple[str, str | None]]:
        results: list[tuple[str, str | None]] = []
        base_file = self._lookup_python(base, importer, exact_only) if base else None
        for name in names:
            if name == "*":
                continue
            submodule = self._lookup_python(f"{base}.{name}" if base else name, importer, exact_only)
            if submodule:
                results.append((submodule, None))
            elif base_file:
                results.append((base_file, name))
        if not names and base_file:
            results.append((base_file, None))
        if names and not results and base_file:
            results.append((base_file, None))
        return [item for item in results if item[0] != importer]

    def _lookup_python(self, dotted: str, importer: str, exact_only: bool) -> str | None:
        if not dotted:
            return None
        if dotted in self.python_modules:
            return self.python_modules[dotted]
        for root in SOURCE_ROOTS:
            candidate = f"{root}.{dotted}"
            if candidate in self.python_modules:
                return self.python_modules[candidate]
        if exact_only or "." not in dotted and dotted not in self.python_suffixes:
            return None
        candidates = self.python_suffixes.get(dotted, set())
        if len(candidates) == 1:
            return next(iter(candidates))
        if candidates:
            return self._closest(importer, candidates)
        return None

    # --- javascript / typescript -------------------------------------------------------
    def resolve_js(self, importer: str, module: str) -> str | None:
        clean = module.strip()
        if clean.startswith("."):
            joined = posixpath.normpath(posixpath.join(posixpath.dirname(importer), clean))
        elif clean.startswith(("@/", "~/")):
            joined = posixpath.normpath(posixpath.join("src", clean[2:]))
        elif clean.startswith("/"):
            joined = clean.lstrip("/")
        else:
            return None
        if joined in self.paths:
            return joined
        stem = joined
        for ext in (".js", ".jsx", ".mjs"):
            if stem.endswith(ext):  # TS projects import "./x.js" that maps to x.ts
                stem = stem[: -len(ext)]
        for ext in JS_EXTENSIONS:
            if f"{stem}{ext}" in self.paths:
                return f"{stem}{ext}"
        for ext in JS_EXTENSIONS:
            if f"{stem}/index{ext}" in self.paths:
                return f"{stem}/index{ext}"
        return None

    # --- generic path-like modules (java, go, rust, c/c++) ------------------------------
    def resolve_generic(self, importer: str, module: str, language: str) -> str | None:
        clean = module.strip().strip("'\"<>").rstrip(";")
        if language in {"c", "cpp"}:
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(importer), clean))
            if relative in self.paths:
                return relative
            matches = {path for path in self.paths if path.endswith("/" + clean) or path == clean}
            return self._closest(importer, matches) if matches else None
        if language == "rust":
            clean = clean.replace("crate::", "").replace("self::", "").replace("super::", "")
            stem = "/".join(part for part in clean.split("::") if part and not part.startswith("{"))
        elif language == "go":
            stem = clean
        else:
            stem = clean.replace(".", "/").rstrip("/*")
        if not stem:
            return None
        matches: set[str] = set()
        for candidate_stem, files in self.stem_index.items():
            if candidate_stem == stem or candidate_stem.endswith("/" + stem):
                matches.update(files)
        if not matches and language == "go":
            matches = {path for path in self.paths if posixpath.dirname(path).endswith(stem) and path.endswith(".go")}
            matches = set(sorted(matches)[:1])
        return self._closest(importer, matches) if matches else None

    def resolve(self, importer: str, detail: dict[str, Any], language: str) -> list[tuple[str, str | None]]:
        if language == "python":
            return self.resolve_python(importer, detail)
        module = detail.get("module") or ""
        if language in {"javascript", "typescript"}:
            target = self.resolve_js(importer, module)
        else:
            target = self.resolve_generic(importer, module, language)
        return [(target, None)] if target and target != importer else []

    def _closest(self, importer: str, candidates: set[str]) -> str:
        importer_parts = PurePosixPath(importer).parts

        def shared(path: str) -> tuple[int, str]:
            parts = PurePosixPath(path).parts
            common = 0
            for a, b in zip(importer_parts, parts, strict=False):
                if a != b:
                    break
                common += 1
            return (-common, path)

        return sorted(candidates, key=shared)[0]
