from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

from app.models import FileRecord, Weakness
from app.utils.redaction import find_secret, redact

SETUP_RE = re.compile(r"\b(install|setup|quickstart|getting started|usage|how to run)\b", re.IGNORECASE)
LINT_CONFIGS = {".flake8", "ruff.toml", ".ruff.toml", ".eslintrc", ".eslintrc.json", ".eslintrc.js", ".eslintrc.cjs", "eslint.config.js", "eslint.config.mjs", ".prettierrc", "biome.json", ".golangci.yml", "rustfmt.toml", ".pylintrc", "tox.ini"}
CI_PATH_PREFIXES = (".github/workflows/", ".gitlab-ci.yml", ".circleci/", "azure-pipelines.yml", "jenkinsfile", ".travis.yml", "bitbucket-pipelines.yml")
LICENSE_NAMES = {"license", "license.md", "license.txt", "copying", "licence"}
ABSOLUTE_PATH_RE = re.compile(r"""['"](/home/[^'"\s]+|/Users/[^'"\s]+|[A-Za-z]:\\\\Users\\\\[^'"\s]+)['"]""")
SQL_KEYWORDS_RE = re.compile(r"\b(SELECT|UPDATE|DELETE|INSERT)\b", re.IGNORECASE)
JS_RULES = [
    (re.compile(r"(?<![\w.$])eval\s*\("), "high", "security", "Dynamic code execution (eval) in JavaScript", "eval executes arbitrary strings as code.", "Parse data with JSON.parse or use explicit dispatch."),
    (re.compile(r"\bnew\s+Function\s*\("), "high", "security", "Dynamic Function constructor", "new Function compiles strings into code.", "Avoid runtime code generation from strings."),
    (re.compile(r"\.innerHTML\s*=(?!=)"), "medium", "security", "innerHTML assignment", "Assigning unsanitized HTML enables XSS.", "Use textContent or a sanitizer such as DOMPurify."),
    (re.compile(r"dangerouslySetInnerHTML"), "medium", "security", "dangerouslySetInnerHTML usage", "Unsanitized HTML in React enables XSS.", "Sanitize the HTML or render structured content."),
    (re.compile(r"\bchild_process\b.*\bexec(?:Sync)?\s*\(|\bexec(?:Sync)?\s*\(\s*`"), "high", "security", "Shell command execution via child_process.exec", "exec runs through a shell and enables command injection.", "Use execFile/spawn with an argument array."),
    (re.compile(r"(?i)\bcors\(\s*\)|Access-Control-Allow-Origin['\"]?\s*[:,]\s*['\"]\*"), "low", "security", "Permissive CORS configuration", "Any origin can call the API with the user's browser.", "Restrict allowed origins."),
]
WEAKNESS_CATEGORIES = {"documentation", "testing", "architecture", "performance", "security", "maintainability", "deployment", "run_readiness", "dependency", "documentation_drift", "portability"}


class WeaknessDetector:
    def detect(self, records: list[FileRecord], facts: dict | None = None, graph: dict[str, Any] | None = None, repo_id: str | None = None) -> list[Weakness]:
        facts = facts or {}
        graph = graph or {}
        self._repo_id = repo_id or (records[0].repo_id if records else "unknown")
        findings: list[Weakness] = []
        names = {Path(record.path).name.lower() for record in records}
        lower_paths = [record.path.lower() for record in records]
        readmes = [record for record in records if Path(record.path).name.lower().startswith("readme") and record.path.count("/") == 0] or [
            record for record in records if Path(record.path).name.lower().startswith("readme")
        ]

        if not readmes:
            findings.append(self._w("high", "documentation", "Missing README", None, None, None, "Users cannot quickly understand setup, purpose, or usage.", "Add a README with overview, setup, run commands, and examples.", confidence="high"))
        elif max(len(record.text.strip()) for record in readmes) < 500:
            findings.append(self._w("medium", "documentation", "README is thin", readmes[0].path, None, None, "A weak README slows onboarding and makes the project harder to evaluate.", "Expand the README with architecture, setup, examples, and limitations.", confidence="high"))
        elif not any(SETUP_RE.search(record.text) for record in readmes):
            findings.append(self._w("medium", "documentation", "README lacks setup or run instructions", readmes[0].path, None, None, "A reader may understand the project but still be unable to run it.", "Add a quickstart section with installation, environment variables, and run commands."))

        if not facts.get("dependency_files") and not names & {"requirements.txt", "pyproject.toml", "package.json", "go.mod", "cargo.toml", "pom.xml", "build.gradle", "setup.py"}:
            findings.append(self._w("medium", "run_readiness", "No dependency manifest detected", None, None, None, "Reproducing the environment is harder without declared dependencies.", "Add a language-appropriate dependency manifest."))
        code_files = sum(1 for record in records if record.role == "code")
        test_files = sum(1 for record in records if record.role == "test")
        if not facts.get("has_tests") and test_files == 0:
            findings.append(self._w("medium", "testing", "No tests detected", None, None, None, "Changes can regress behavior without automated checks.", "Add focused unit tests for core logic and smoke tests for user flows.", confidence="high"))
        elif code_files >= 10 and test_files / max(1, code_files) < 0.1:
            findings.append(self._w("low", "testing", "Low test-to-code file ratio", None, None, f"{test_files} test files for {code_files} code files", "Most modules likely have no direct tests.", "Add tests for the most central modules first (see central files)."))
        if ".gitignore" not in names:
            findings.append(self._w("medium", "maintainability", "Missing .gitignore", None, None, None, "Generated files, secrets, or local artifacts may be committed accidentally.", "Add a .gitignore tailored to the stack."))
        if not names & LINT_CONFIGS and not self._pyproject_has_lint(records):
            findings.append(self._w("low", "maintainability", "No linting or formatting config detected", None, None, None, "Style and simple correctness issues may be caught late in review.", "Add Ruff/Black, ESLint/Prettier, or the stack's standard formatter configuration."))
        if not any(path.startswith(CI_PATH_PREFIXES) or path in CI_PATH_PREFIXES for path in lower_paths):
            findings.append(self._w("low", "testing", "No CI workflow detected", None, None, None, "Tests and linting may not run consistently before merge.", "Add a lightweight CI workflow that installs dependencies and runs tests."))
        if not names & LICENSE_NAMES:
            findings.append(self._w("low", "maintainability", "No license file detected", None, None, None, "Reviewers and users may not know how the code can be reused.", "Add an appropriate LICENSE file if the project is intended to be public."))
        if facts.get("api_route_files") and not facts.get("deployment_files"):
            findings.append(self._w("medium", "deployment", "No deployment artifact detected", None, None, None, "The repository exposes an app/API but lacks a clear deployable container or process file.", "Add a Dockerfile or deployment guide for the service."))
        if not facts.get("main_entrypoints") and code_files:
            findings.append(self._w("medium", "architecture", "Unclear entrypoint", None, None, None, "A reviewer may not know where execution starts.", "Document the main entrypoint and add a conventional CLI/app file if appropriate."))

        for cycle in graph.get("dependency_cycles", [])[:5]:
            findings.append(self._w("medium", "architecture", "Circular dependency between modules", cycle[0], None, " -> ".join(cycle[:6]), "Import cycles make modules hard to test and can cause import-time failures.", "Extract the shared pieces into a lower-level module or invert the dependency."))
        for item in graph.get("complex_functions", [])[:8]:
            if (item.get("complexity") or 0) >= 15 or item.get("lines", 0) >= 120:
                findings.append(self._w(
                    "low", "maintainability", "Highly complex function", item.get("file_path"), item.get("start_line"),
                    f"{item.get('symbol')}: complexity {item.get('complexity')}, {item.get('lines')} lines",
                    "Long, branch-heavy functions are hard to test and change safely.", "Split into smaller functions with focused tests.",
                ))

        for record in records:
            name = PurePosixPath(record.path).name.lower()
            if name == ".env" or (name.startswith(".env.") and not name.endswith((".example", ".sample", ".template", ".dist"))):
                findings.append(self._w("high", "security", "Environment file committed to the repository", record.path, None, None, "Committed .env files frequently contain live credentials.", "Remove it from git history, rotate its secrets, and add it to .gitignore.", confidence="high"))
            if record.kind != "text" or record.role == "generated":
                continue
            if record.size_bytes > 1_000_000:
                findings.append(self._w("medium", "performance", "Large text file committed", record.path, None, f"{record.size_bytes // 1024} KB", "Large files slow scanning, review, and repository operations.", "Move generated or bulky assets to external storage when possible."))
            findings.extend(self._secret_findings(record))
            if record.language == "python" and record.role in {"code", "test"}:
                findings.extend(self._python_findings(record))
            elif record.language in {"javascript", "typescript"} and record.role == "code":
                findings.extend(self._js_findings(record))
            if record.role in {"code", "config"}:
                for line_number, line in enumerate(record.text.splitlines(), start=1):
                    if ABSOLUTE_PATH_RE.search(line):
                        findings.append(self._w("medium", "portability", "Hardcoded absolute path detected", record.path, line_number, line.strip()[:180], "Machine-specific paths break for other users or CI.", "Use project-relative paths or configuration."))
        return self._dedupe(findings)

    # --- detectors --------------------------------------------------------------------
    def _secret_findings(self, record: FileRecord) -> list[Weakness]:
        findings = []
        for line_number, line in enumerate(record.text.splitlines(), start=1):
            if len(line) > 2000:
                continue
            if find_secret(line):
                in_test = record.role == "test"
                findings.append(self._w(
                    "medium" if in_test else "high", "security", "Possible secret in test fixture" if in_test else "High-confidence hardcoded secret",
                    record.path, line_number, redact(line.strip())[:220], "Secrets in source control can leak credentials.",
                    "Move the secret to an environment variable and rotate the exposed credential.", confidence="medium" if in_test else "high",
                ))
        return findings

    def _python_findings(self, record: FileRecord) -> list[Weakness]:
        try:
            tree = ast.parse(record.text)
        except (SyntaxError, ValueError):
            return []
        lines = record.text.splitlines()
        findings: list[Weakness] = []

        def evidence(node: ast.AST) -> str:
            line = getattr(node, "lineno", 1)
            return lines[line - 1].strip()[:180] if 0 < line <= len(lines) else ""

        is_test = record.role == "test"
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.id if isinstance(func, ast.Name) else (func.attr if isinstance(func, ast.Attribute) else "")
                owner = func.value.id if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) else ""
                if isinstance(func, ast.Name) and name in {"eval", "exec"} and not is_test:
                    constant = bool(node.args) and isinstance(node.args[0], ast.Constant)
                    findings.append(self._w("medium" if constant else "high", "security", f"Dynamic code execution ({name})", record.path, node.lineno, evidence(node), f"{name}() can execute attacker-controlled input.", "Replace dynamic execution with explicit parsing or dispatch tables.", confidence="high"))
                if any(kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in node.keywords):
                    findings.append(self._w("high", "security", "subprocess shell=True detected", record.path, node.lineno, evidence(node), "Shell invocation can enable command injection.", "Pass command arguments as a list and validate user input.", confidence="high"))
                if owner == "os" and name in {"system", "popen"} and not is_test:
                    findings.append(self._w("medium", "security", f"os.{name} shell execution", record.path, node.lineno, evidence(node), "os.system/popen run through the shell.", "Use subprocess.run with an argument list."))
                if owner in {"pickle", "cPickle", "dill", "joblib"} and name in {"load", "loads"} and not is_test:
                    findings.append(self._w("medium", "security", "Unsafe deserialization", record.path, node.lineno, evidence(node), f"{owner}.{name} can execute code when loading untrusted data.", "Only load trusted files; prefer JSON/safetensors for untrusted input."))
                if owner == "yaml" and name == "load" and not any(kw.arg == "Loader" for kw in node.keywords) and len(node.args) < 2:
                    findings.append(self._w("medium", "security", "yaml.load without a safe Loader", record.path, node.lineno, evidence(node), "yaml.load can construct arbitrary Python objects.", "Use yaml.safe_load."))
                if name in {"execute", "executemany", "raw", "text"} and node.args and self._is_formatted_sql(node.args[0]):
                    findings.append(self._w("high", "security", "SQL built with string formatting", record.path, node.lineno, evidence(node), "String-built SQL can lead to injection vulnerabilities.", "Use parameterized queries or an ORM.", confidence="high"))
                if name in {"get", "post", "put", "delete", "request"} and owner in {"requests", "httpx"} and any(kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in node.keywords):
                    findings.append(self._w("medium", "security", "TLS verification disabled", record.path, node.lineno, evidence(node), "verify=False allows man-in-the-middle attacks.", "Keep certificate verification on; configure a CA bundle if needed."))
            elif isinstance(node, ast.ExceptHandler) and not is_test:
                broad = node.type is None or (isinstance(node.type, ast.Name) and node.type.id in {"Exception", "BaseException"})
                silent = all(isinstance(stmt, (ast.Pass, ast.Continue)) or (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)) for stmt in node.body)
                if broad and silent:
                    findings.append(self._w("medium", "maintainability", "Exception silently swallowed", record.path, node.lineno, evidence(node), "Silent broad catches hide failures and make debugging hard.", "Catch specific exceptions and log actionable context."))
        return findings

    def _is_formatted_sql(self, node: ast.AST) -> bool:
        if isinstance(node, ast.JoinedStr):
            text = "".join(part.value for part in node.values if isinstance(part, ast.Constant) and isinstance(part.value, str))
            return bool(SQL_KEYWORDS_RE.search(text)) and any(isinstance(part, ast.FormattedValue) for part in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
            left = node.left
            if isinstance(left, ast.Constant) and isinstance(left.value, str) and SQL_KEYWORDS_RE.search(left.value):
                return True
            return self._is_formatted_sql(left)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            base = node.func.value
            return isinstance(base, ast.Constant) and isinstance(base.value, str) and bool(SQL_KEYWORDS_RE.search(base.value))
        return False

    def _js_findings(self, record: FileRecord) -> list[Weakness]:
        findings = []
        in_block_comment = False
        for line_number, line in enumerate(record.text.splitlines(), start=1):
            stripped = line.strip()
            if in_block_comment:
                in_block_comment = "*/" not in stripped
                continue
            if stripped.startswith("/*"):
                in_block_comment = "*/" not in stripped
                continue
            if stripped.startswith("//") or len(stripped) > 1000:
                continue
            for pattern, severity, category, title, why, fix in JS_RULES:
                if pattern.search(stripped):
                    findings.append(self._w(severity, category, title, record.path, line_number, stripped[:180], why, fix))
        return findings

    def _pyproject_has_lint(self, records: list[FileRecord]) -> bool:
        for record in records:
            if Path(record.path).name.lower() == "pyproject.toml" and re.search(r"^\[tool\.(ruff|black|flake8|pylint|isort)", record.text, re.MULTILINE):
                return True
            if Path(record.path).name.lower() == "package.json" and re.search(r'"(eslint|prettier|biome)"', record.text):
                return True
        return False

    # --- helpers ----------------------------------------------------------------------
    def _w(
        self,
        severity: str,
        category: str,
        title: str,
        file_path: str | None,
        line_number: int | None,
        evidence: str | None,
        why: str,
        fix: str,
        confidence: str = "medium",
    ) -> Weakness:
        repo_id = getattr(self, "_repo_id", "unknown")
        raw_id = f"{repo_id}:{category}:{title}:{file_path}:{line_number}:{evidence or ''}"
        finding_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        return Weakness(
            id=finding_id,
            severity=severity,
            category=category,
            confidence=confidence,
            title=title,
            file_path=file_path,
            line_number=line_number,
            evidence=evidence,
            why_it_matters=why,
            suggested_fix=fix,
        )

    def make(self, repo_id: str, **kwargs: Any) -> Weakness:
        """Public factory used by other analyzers (drift, dependency scan) so IDs stay repo-scoped."""
        self._repo_id = repo_id
        return self._w(**kwargs)

    def _dedupe(self, findings: list[Weakness]) -> list[Weakness]:
        seen = set()
        deduped = []
        severity_order = {"critical": -1, "high": 0, "medium": 1, "low": 2}
        for finding in sorted(findings, key=lambda item: (severity_order.get(item.severity, 9), item.category, item.title, item.file_path or "", item.line_number or 0)):
            if finding.id in seen:
                continue
            seen.add(finding.id)
            deduped.append(finding)
        return deduped
