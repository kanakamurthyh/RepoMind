from __future__ import annotations

import hashlib
import re
from pathlib import Path

from app.models import FileRecord, Weakness

AWS_KEY_RE = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
GITHUB_TOKEN_RE = re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{30,}\b")
OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")
GOOGLE_KEY_RE = re.compile(r"\bAIza[0-9A-Za-z_-]{25,}\b")
PEM_PRIVATE_RE = re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")
ASSIGNED_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|password|passwd|token|access[_-]?token|auth[_-]?token)\b\s*[:=]\s*['\"]([^'\"\s]{12,})['\"]"
)
RAW_SQL_RE = re.compile(r"(SELECT|UPDATE|DELETE|INSERT)\s+.+(%|\+|f['\"])", re.IGNORECASE)
ABSOLUTE_PATH_RE = re.compile(r"(?<![\w/])(/home/[^'\"\s]+|/users/[^'\"\s]+|c:\\\\users\\\\[^'\"\s]+)", re.IGNORECASE)
SETUP_RE = re.compile(r"\b(install|setup|quickstart|getting started|run|usage)\b", re.IGNORECASE)
LINT_CONFIGS = {".flake8", "ruff.toml", ".ruff.toml", "pyproject.toml", ".eslintrc", ".eslintrc.json", ".prettierrc", "biome.json"}
CI_PATH_MARKERS = {".github/workflows", ".gitlab-ci.yml", "circle.yml", ".circleci/config.yml"}
LICENSE_NAMES = {"license", "license.md", "license.txt", "copying"}


class WeaknessDetector:
    def detect(self, records: list[FileRecord], facts: dict | None = None) -> list[Weakness]:
        facts = facts or {}
        findings: list[Weakness] = []
        paths = {record.path.lower(): record for record in records}
        names = {Path(record.path).name.lower(): record for record in records}
        readmes = [record for record in records if Path(record.path).name.lower().startswith("readme")]

        if not readmes:
            findings.append(self._w("high", "documentation", "Missing README", None, None, None, "Users cannot quickly understand setup, purpose, or usage.", "Add a README with overview, setup, run commands, and examples."))
        elif max(len(record.text.strip()) for record in readmes) < 500:
                findings.append(self._w("medium", "documentation", "README is thin", readmes[0].path, None, None, "A weak README slows onboarding and makes the project harder to evaluate.", "Expand the README with architecture, setup, examples, and limitations.", confidence="high"))
        elif not any(SETUP_RE.search(record.text) for record in readmes):
            findings.append(self._w("medium", "documentation", "README lacks setup or run instructions", readmes[0].path, None, None, "A reader may understand the project but still be unable to run it.", "Add a quickstart section with installation, environment variables, and run commands."))

        if not facts.get("dependency_files") and not any(name in names for name in {"requirements.txt", "pyproject.toml", "package.json", "go.mod", "cargo.toml"}):
            findings.append(self._w("medium", "run_readiness", "No dependency manifest detected", None, None, None, "Reproducing the environment is harder without declared dependencies.", "Add a language-appropriate dependency manifest."))
        if not facts.get("has_tests") and not any(record.role == "test" for record in records):
            findings.append(self._w("medium", "testing", "No tests detected", None, None, None, "Changes can regress behavior without automated checks.", "Add focused unit tests for core logic and smoke tests for user flows."))
        if ".gitignore" not in names:
            findings.append(self._w("medium", "maintainability", "Missing .gitignore", None, None, None, "Generated files, secrets, or local artifacts may be committed accidentally.", "Add a .gitignore tailored to the stack."))
        if not any(name in names for name in LINT_CONFIGS):
            findings.append(self._w("low", "maintainability", "No linting or formatting config detected", None, None, None, "Style and simple correctness issues may be caught late in review.", "Add Ruff/Black, ESLint/Prettier, or the stack's standard formatter configuration."))
        if not any(path in paths or any(record.path.lower().startswith(f"{path}/") for record in records) for path in CI_PATH_MARKERS):
            findings.append(self._w("low", "testing", "No CI workflow detected", None, None, None, "Tests and linting may not run consistently before merge.", "Add a lightweight CI workflow that installs dependencies and runs tests."))
        if not any(name in names for name in LICENSE_NAMES):
            findings.append(self._w("low", "maintainability", "No license file detected", None, None, None, "Reviewers and users may not know how the code can be reused.", "Add an appropriate LICENSE file if the project is intended to be public."))
        if self._appears_deployable(records, facts) and not facts.get("deployment_files"):
            findings.append(self._w("medium", "deployment", "No deployment artifact detected", None, None, None, "The repository appears to expose an app/API but lacks a clear deployable container or process file.", "Add a Dockerfile or deployment guide for the service."))
        if not facts.get("main_entrypoints"):
            findings.append(self._w("medium", "architecture", "Unclear entrypoint", None, None, None, "A reviewer may not know where execution starts.", "Document the main entrypoint and add a conventional CLI/app file if appropriate."))

        for record in records:
            if record.size_bytes > 1_000_000:
                findings.append(self._w("medium", "performance", "Large text file committed", record.path, None, None, "Large files slow scanning, review, and repository operations.", "Move generated or bulky assets to external storage when possible."))
            findings.extend(self._security_findings(record))

        return self._dedupe(findings)

    def _security_findings(self, record: FileRecord) -> list[Weakness]:
        findings: list[Weakness] = []
        for line_number, line in enumerate(record.text.splitlines(), start=1):
            stripped = line.strip()
            secret_match = self._secret_match(stripped)
            if secret_match:
                findings.append(self._w("high", "security", "High-confidence hardcoded secret", record.path, line_number, self._redact(stripped), "Secrets in source control can leak credentials.", "Move the secret to an environment variable and rotate the exposed credential.", confidence="high"))
            lowered = stripped.lower()
            if record.language == "python":
                if "eval(" in lowered or "exec(" in lowered:
                    findings.append(self._w("high", "security", "Dynamic code execution detected", record.path, line_number, stripped[:180], "eval/exec can execute attacker-controlled input.", "Replace dynamic execution with explicit parsing or dispatch tables.", confidence="medium"))
                if "shell=true" in lowered:
                    findings.append(self._w("high", "security", "subprocess shell=True detected", record.path, line_number, stripped[:180], "Shell invocation can enable command injection.", "Pass command arguments as a list and validate user input.", confidence="medium"))
                if "pickle.load" in lowered or "pickle.loads" in lowered:
                    findings.append(self._w("medium", "security", "Pickle deserialization detected", record.path, line_number, stripped[:180], "Pickle can execute code when loading untrusted data.", "Use safer formats such as JSON for untrusted input.", confidence="medium"))
                if RAW_SQL_RE.search(stripped):
                    findings.append(self._w("high", "security", "Possible raw SQL string formatting", record.path, line_number, stripped[:180], "String-built SQL can lead to injection vulnerabilities.", "Use parameterized queries or an ORM.", confidence="medium"))
                if ("except:" in lowered or "except exception" in lowered) and "pass" in lowered:
                    findings.append(self._w("medium", "maintainability", "Broad exception handling may hide failures", record.path, line_number, stripped[:180], "Silent or broad catches make debugging harder.", "Catch specific exceptions and log actionable context."))
            if record.role in {"code", "config"} and ABSOLUTE_PATH_RE.search(stripped):
                findings.append(self._w("medium", "portability", "Hardcoded absolute path detected", record.path, line_number, stripped[:180], "Machine-specific paths can break for other users or CI.", "Use project-relative paths or configuration."))
        return findings

    def _secret_match(self, line: str) -> bool:
        if PEM_PRIVATE_RE.search(line) or AWS_KEY_RE.search(line) or GITHUB_TOKEN_RE.search(line) or OPENAI_KEY_RE.search(line) or GOOGLE_KEY_RE.search(line):
            return True
        assigned = ASSIGNED_SECRET_RE.search(line)
        if not assigned:
            return False
        value = assigned.group(2)
        return not any(placeholder in value.lower() for placeholder in ("example", "your_", "changeme", "placeholder", "dummy", "xxxx"))

    def _redact(self, line: str) -> str:
        return re.sub(r"(['\"])[^'\"]{8,}(['\"])", r"\1<redacted>\2", line[:220])

    def _appears_deployable(self, records: list[FileRecord], facts: dict) -> bool:
        if facts.get("api_route_files"):
            return True
        text = "\n".join(record.text[:3000].lower() for record in records if record.role in {"code", "config"})
        deployable_markers = ("from fastapi", "import fastapi", "from flask", "import flask", "django", "streamlit run", "express()")
        return any(marker in text for marker in deployable_markers)

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
        raw_id = f"{category}:{title}:{file_path}:{line_number}:{evidence or ''}"
        finding_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:12]
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

    def _dedupe(self, findings: list[Weakness]) -> list[Weakness]:
        seen = set()
        deduped = []
        severity_order = {"high": 0, "medium": 1, "low": 2}
        for finding in sorted(findings, key=lambda item: (severity_order.get(item.severity, 9), item.category, item.title)):
            key = (finding.category, finding.title, finding.file_path, finding.line_number, hashlib.sha1((finding.evidence or "").encode("utf-8")).hexdigest()[:10])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(finding)
        return deduped
