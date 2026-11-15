"""Docs-vs-code drift detection: verify what the documentation claims against the repository."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote

from app.models import FileRecord

ENV_NAME_RE = re.compile(r"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b")
ENV_ASSIGN_RE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]+)\s*=", re.MULTILINE)
DOC_ROUTE_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\s+`?(/[\w\-./{}:<>]*)`?")
MD_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)\)")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SYSTEM_ENV = {"HOME", "PATH", "USER", "PWD", "SHELL", "TERM", "LANG", "TMPDIR", "CI", "NODE_ENV", "PYTHONPATH", "VIRTUAL_ENV", "GITHUB_TOKEN", "GITHUB_ACTIONS", "RUNNER_OS", "HOSTNAME", "DISPLAY", "EDITOR"}
LOCAL_RUNNERS = re.compile(r"^(?:sudo\s+)?(?:python3?|py)\s+(?!-m\b|-c\b|-V\b|--)(\S+\.py)\b")
MODULE_RUNNER = re.compile(r"^(?:sudo\s+)?(?:python3?|py)\s+-m\s+([\w.]+)")
SCRIPT_RUNNER = re.compile(r"^(?:sudo\s+)?(?:bash|sh|zsh|source|\.)\s+(\S+\.(?:sh|bash|zsh))\b|^(\./\S+)")
FILE_RUNNERS = re.compile(r"^(?:streamlit\s+run|node|ts-node|tsx|deno\s+run|go\s+run)\s+(\S+\.(?:py|js|mjs|ts|go))\b")
UVICORN_RE = re.compile(r"^(?:uvicorn|gunicorn|hypercorn)\s+(?:[^\s]+\s+)*?([\w.]+):([\w]+)")
REQ_RE = re.compile(r"\bpip3?\s+install\s+(?:[^\n]*?\s)?-r\s+(\S+)")
NPM_SCRIPT_RE = re.compile(r"^(?:npm|pnpm)\s+run\s+([\w:.-]+)|^yarn\s+(?!add\b|install\b|global\b|dlx\b)([\w:.-]+)$")
MAKE_RE = re.compile(r"^make\s+([\w.-]+)")
COPY_RE = re.compile(r"^cp\s+(\S+)\s+\S+")
THIRD_PARTY_MODULES = {"pip", "venv", "pytest", "http", "unittest", "streamlit", "uvicorn", "flask", "django", "build", "twine", "black", "ruff", "mypy", "spacy", "nltk", "alembic", "celery", "gunicorn", "jupyter", "ipykernel", "json", "timeit", "cProfile", "pdb", "poetry", "pipx", "grpc_tools", "playwright"}


@dataclass(slots=True)
class DriftIssue:
    kind: str
    severity: str
    title: str
    detail: str
    doc_path: str | None = None
    doc_line: int | None = None
    evidence: str | None = None
    code_refs: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_route(path: str) -> str:
    path = re.sub(r"\{[^}]+\}|:[A-Za-z_]\w*|<[^>]+>", "{}", path.strip().rstrip("/") or "/")
    return path.lower()


class DocsDriftDetector:
    def detect(self, records: list[FileRecord], facts: dict[str, Any], graph: dict[str, Any], env_usage: dict[str, list[str]] | None = None) -> dict[str, Any]:
        paths = {record.path for record in records}
        dirs = {str(parent) for path in paths for parent in PurePosixPath(path).parents if str(parent) != "."}
        by_path = {record.path: record for record in records}
        docs = [r for r in records if r.kind == "text" and r.role == "docs" and r.language in {"markdown", "rst"} and (r.path.count("/") == 0 or r.path.lower().startswith(("docs/", "doc/")))]
        issues: list[DriftIssue] = []
        checks = 0

        commands = self._doc_commands(docs)
        package_scripts = self._package_scripts(by_path.get("package.json"))
        make_targets = self._make_targets(next((r for r in records if r.path.lower() == "makefile"), None))
        top_level = {PurePosixPath(p).parts[0] for p in paths}
        for command in commands:
            text, doc, line = command["command"], command["doc"], command["line"]
            if re.match(r"^cd\s", text) or "&&" in text and text.lower().startswith("cd "):
                continue
            for problem in self._check_command(text, paths, dirs, top_level, package_scripts, make_targets):
                checks += 1
                if problem:
                    issues.append(DriftIssue("command", "medium", problem[0], problem[1], doc, line, text[:200]))

        # Environment variables: used in code vs documented.
        env_usage = env_usage or {}
        documented = self._documented_env(records, docs)
        config_text = "\n".join(r.text for r in records if r.kind == "text" and r.role in {"code", "config"} and not PurePosixPath(r.path).name.startswith(".env"))
        used = {name: refs for name, refs in env_usage.items() if name not in SYSTEM_ENV}
        for name, refs in sorted(used.items()):
            checks += 1
            if name not in documented:
                issues.append(DriftIssue("env_undocumented", "low", f"Environment variable {name} is read but not documented", f"Read in {', '.join(refs[:3])} but not mentioned in README/docs or .env.example.", None, None, name, refs[:5]))
        for name, (doc, line) in sorted(documented.items()):
            checks += 1
            if name in used or name in SYSTEM_ENV:
                continue
            if not re.search(rf"\b{re.escape(name)}\b", config_text):
                issues.append(DriftIssue("env_unused", "low", f"Documented environment variable {name} is never used", "Documented, but no code or config file references it.", doc, line, name))

        # API routes: documented vs implemented.
        implemented = {}
        for node in graph.get("nodes", []):
            if node.get("kind") == "route" and node.get("path"):
                implemented[(str(node.get("method") or "ANY").upper(), _normalize_route(str(node["path"])))] = node.get("file_path")
        documented_routes = self._documented_routes(docs)
        if implemented:
            implemented_paths = {path for _, path in implemented}
            for (method, route), (doc, line, raw) in documented_routes.items():
                checks += 1
                if (method, route) in implemented or ("ANY", route) in implemented:
                    continue
                if route in implemented_paths:
                    issues.append(DriftIssue("route_method", "medium", f"Documented {method} {raw} uses a different HTTP method in code", "The path exists but the documented method does not.", doc, line, f"{method} {raw}"))
                else:
                    issues.append(DriftIssue("route_missing", "medium", f"Documented endpoint {method} {raw} does not exist in code", "No route handler with this path was found.", doc, line, f"{method} {raw}"))
            if len(documented_routes) >= 2:
                documented_keys = {route for _, route in documented_routes}
                missing_docs = sorted(f"{m} {p}" for (m, p), _ in implemented.items() if p not in documented_keys)
                checks += 1
                if missing_docs:
                    issues.append(DriftIssue("route_undocumented", "low", f"{len(missing_docs)} implemented endpoint(s) are not documented", ", ".join(missing_docs[:12]), None, None, None, missing_docs[:20]))

        # Relative links and referenced paths in docs.
        for doc in docs:
            in_fence = False
            base = PurePosixPath(doc.path).parent
            for line_number, line in enumerate(doc.text.splitlines(), start=1):
                if FENCE_RE.match(line):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue
                for target in MD_LINK_RE.findall(line):
                    if re.match(r"^(https?:|mailto:|#|/|\{|<)", target):
                        continue
                    clean = unquote(target.split("#", 1)[0].split("?", 1)[0])
                    if not clean:
                        continue
                    resolved = posixpath.normpath(posixpath.join(str(base), clean))
                    checks += 1
                    if resolved not in paths and resolved.rstrip("/") not in dirs:
                        issues.append(DriftIssue("broken_link", "low", f"Broken relative link to {clean}", "The linked file or folder does not exist in the repository.", doc.path, line_number, target))

        score = round(1 - (len(issues) / checks), 3) if checks else None
        return {
            "checks": checks,
            "issue_count": len(issues),
            "docs_accuracy": score,
            "issues": [issue.to_dict() for issue in issues[:100]],
            "documented_env_vars": sorted(documented)[:100],
            "used_env_vars": sorted(used)[:100],
        }

    # --- extraction --------------------------------------------------------------------
    def _doc_commands(self, docs: list[FileRecord]) -> list[dict[str, Any]]:
        commands = []
        for doc in docs:
            in_fence = False
            fence_lang = ""
            continued = ""
            for line_number, raw in enumerate(doc.text.splitlines(), start=1):
                fence = FENCE_RE.match(raw)
                if fence:
                    in_fence = not in_fence
                    fence_lang = raw.strip()[3:].strip().lower() if in_fence else ""
                    continue
                if not in_fence or fence_lang not in {"", "bash", "sh", "shell", "console", "zsh", "powershell", "cmd", "text"}:
                    continue
                line = re.sub(r"^\s*(?:\$|>|PS>)\s+", "", raw).strip()
                if not line or line.startswith("#"):
                    continue
                if line.endswith("\\"):
                    continued += line[:-1] + " "
                    continue
                full = (continued + line).strip()
                continued = ""
                for part in re.split(r"\s*(?:&&|;)\s*", full):
                    if part:
                        commands.append({"command": part, "doc": doc.path, "line": line_number})
        return commands

    def _check_command(self, text: str, paths: set[str], dirs: set[str], top_level: set[str], scripts: set[str] | None, targets: set[str] | None) -> list[tuple[str, str] | None]:
        results: list[tuple[str, str] | None] = []

        def exists(path: str) -> bool:
            clean = path.strip("'\"").removeprefix("./")
            return clean in paths or clean in dirs

        match = LOCAL_RUNNERS.match(text)
        if match:
            results.append(None if exists(match.group(1)) else (f"Documented command runs missing file {match.group(1)}", "The script referenced by the docs is not in the repository."))
        match = MODULE_RUNNER.match(text)
        if match:
            module = match.group(1)
            first = module.split(".")[0]
            if first not in THIRD_PARTY_MODULES and (first in top_level or f"src/{first}" in dirs):
                as_path = module.replace(".", "/")
                ok = any(p in paths for p in (f"{as_path}.py", f"{as_path}/__main__.py", f"src/{as_path}.py", f"src/{as_path}/__main__.py"))
                results.append(None if ok else (f"Documented module {module} cannot be run", "`python -m` target has no module file or __main__.py."))
        match = SCRIPT_RUNNER.match(text)
        if match:
            script = match.group(1) or match.group(2)
            if script and not script.startswith("./configure") and "$" not in script:
                results.append(None if exists(script) else (f"Documented script {script} is missing", "The docs reference a script that is not in the repository."))
        match = FILE_RUNNERS.match(text)
        if match:
            results.append(None if exists(match.group(1)) else (f"Documented entry file {match.group(1)} is missing", "The docs start a file that does not exist."))
        match = UVICORN_RE.match(text)
        if match:
            module_path = match.group(1).replace(".", "/")
            ok = any(p in paths for p in (f"{module_path}.py", f"src/{module_path}.py", f"{module_path}/__init__.py"))
            results.append(None if ok else (f"ASGI/WSGI target {match.group(1)}:{match.group(2)} not found", "The server module in the docs does not exist."))
        match = REQ_RE.search(text)
        if match:
            results.append(None if exists(match.group(1)) else (f"Requirements file {match.group(1)} is missing", "`pip install -r` points at a file that does not exist."))
        match = NPM_SCRIPT_RE.match(text)
        if match and scripts is not None:
            script = match.group(1) or match.group(2)
            if script not in {"install", "add", "run", "test", "start", "build"} or match.group(1):
                results.append(None if script in scripts else (f"npm script '{script}' is not defined", "package.json has no script with this name."))
        match = MAKE_RE.match(text)
        if match and targets is not None:
            results.append(None if match.group(1) in targets else (f"Make target '{match.group(1)}' is not defined", "The Makefile has no such target."))
        match = COPY_RE.match(text)
        if match and not match.group(1).startswith(("/", "~", "$")):
            results.append(None if exists(match.group(1)) else (f"Documented copy source {match.group(1)} is missing", "`cp` source file does not exist."))
        return results

    def _package_scripts(self, record: FileRecord | None) -> set[str] | None:
        if record is None:
            return None
        try:
            data = json.loads(record.text)
        except json.JSONDecodeError:
            return None
        scripts = data.get("scripts") or {}
        return set(scripts) if isinstance(scripts, dict) else set()

    def _make_targets(self, record: FileRecord | None) -> set[str] | None:
        if record is None:
            return None
        return {match.group(1) for match in re.finditer(r"^([A-Za-z0-9_.-]+)\s*:(?!=)", record.text, re.MULTILINE)}

    def _documented_env(self, records: list[FileRecord], docs: list[FileRecord]) -> dict[str, tuple[str, int]]:
        documented: dict[str, tuple[str, int]] = {}
        for record in records:
            name = PurePosixPath(record.path).name.lower()
            if name.startswith(".env.") or name in {"docker-compose.yml", "docker-compose.yaml", "compose.yml", "compose.yaml", "app.json", "render.yaml", "fly.toml"}:
                for match in ENV_ASSIGN_RE.finditer(record.text):
                    line = record.text[: match.start()].count("\n") + 1
                    documented.setdefault(match.group(1), (record.path, line))
                for match in re.finditer(r"^\s*-?\s*([A-Z][A-Z0-9_]+)\s*[:=]", record.text, re.MULTILINE):
                    line = record.text[: match.start()].count("\n") + 1
                    documented.setdefault(match.group(1), (record.path, line))
        for doc in docs:
            for line_number, line in enumerate(doc.text.splitlines(), start=1):
                for name in ENV_NAME_RE.findall(line):
                    if "`" in line or "|" in line or "=" in line or "export" in line:
                        documented.setdefault(name, (doc.path, line_number))
        return documented

    def _documented_routes(self, docs: list[FileRecord]) -> dict[tuple[str, str], tuple[str, int, str]]:
        routes: dict[tuple[str, str], tuple[str, int, str]] = {}
        for doc in docs:
            for line_number, line in enumerate(doc.text.splitlines(), start=1):
                for method, path in DOC_ROUTE_RE.findall(line):
                    if path.startswith("//") or "." in PurePosixPath(path).name and not path.endswith("/"):
                        continue
                    routes.setdefault((method.upper(), _normalize_route(path)), (doc.path, line_number, path))
        return routes


def safe_split(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return command.split()
