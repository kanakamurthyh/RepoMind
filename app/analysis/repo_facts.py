from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.models import FileRecord

DEPENDENCY_NAMES = {
    "requirements.txt",
    "pyproject.toml",
    "package.json",
    "go.mod",
    "cargo.toml",
    "pom.xml",
    "build.gradle",
}
PACKAGE_NAMES = DEPENDENCY_NAMES | {"setup.py", "setup.cfg", "manifest.in"}
DEPLOYMENT_NAMES = {"dockerfile", "docker-compose.yml", "compose.yml", "vercel.json", "render.yaml", "procfile"}
RUN_HINT_NAMES = DEPENDENCY_NAMES | DEPLOYMENT_NAMES | {"makefile", "justfile", "tox.ini", "noxfile.py"}
SCRIPT_EXTS = {".sh", ".bash", ".ps1"}
SOURCE_DIR_NAMES = {"src", "app", "lib", "package", "packages", "backend", "frontend"}
TEST_DIR_NAMES = {"test", "tests", "__tests__", "spec", "specs"}
DOC_DIR_NAMES = {"docs", "doc", "documentation"}
README_NOISE = re.compile(r"(screenshot|badge|!\[|https?://|<img|shields\.io)", re.IGNORECASE)
RUN_SECTION_RE = re.compile(
    r"\b(prerequisites?|install(?:ation)?|setup|quickstart|getting started|fresh vm|build|run|usage|deploy(?:ment)?|uninstall|troubleshoot(?:ing)?|acceptance checklist)\b",
    re.IGNORECASE,
)
SHELL_PROMPT_RE = re.compile(r"^\s*(?:\$|#)\s+")
COMMAND_START_RE = re.compile(
    r"^\s*(sudo\s+|python[3]?\s+|pip[3]?\s+|uv\s+|poetry\s+|npm\s+|pnpm\s+|yarn\s+|make\s+|docker\s+|docker-compose\s+|bash\s+|sh\s+|./|apt\s+|apt-get\s+|systemctl\s+|cp\s+|mkdir\s+|chmod\s+|export\s+|curl\s+|wget\s+)"
)
FRAMEWORK_PACKAGES = {
    "FastAPI": {"fastapi"},
    "Flask": {"flask"},
    "Django": {"django"},
    "Streamlit": {"streamlit"},
    "Gradio": {"gradio"},
    "SQLAlchemy": {"sqlalchemy"},
    "Pydantic": {"pydantic"},
    "Celery": {"celery"},
    "PyTorch": {"torch", "pytorch"},
    "TensorFlow": {"tensorflow"},
    "scikit-learn": {"sklearn", "scikit-learn"},
    "Hugging Face Transformers": {"transformers"},
    "LangChain": {"langchain", "langchain-core"},
    "LlamaIndex": {"llama-index", "llama_index"},
    "pandas": {"pandas"},
    "React": {"react"},
    "Next.js": {"next"},
    "Vue": {"vue"},
    "Angular": {"@angular/core"},
    "Svelte": {"svelte"},
    "Express": {"express"},
    "NestJS": {"@nestjs/core"},
    "Spring": {"org.springframework", "spring-boot-starter"},
    "Gin": {"github.com/gin-gonic/gin"},
    "Actix": {"actix-web"},
}
UI_MARKERS = re.compile(r"^\s*(?:import\s+streamlit|from\s+streamlit\b|import\s+gradio)", re.MULTILINE)
WEB_MARKERS = re.compile(r"\b(?:FastAPI|Flask|APIRouter|Blueprint)\s*\(|\bexpress\s*\(\s*\)|@(?:app|router)\.(?:get|post|put|delete|patch|route)\b|createServer\s*\(")
CLI_MARKERS = re.compile(r"\b(?:import\s+argparse|from\s+argparse|import\s+click|from\s+click|import\s+typer|from\s+typer|require\(['\"]commander['\"]\)|from\s+['\"]commander['\"]|flag\.Parse\(\)|clap::)")
# Headings whose whole text names the project's features ("🚀 Key Features", "2. What it does"), not e.g. "Engineered Features".
FEATURE_SECTION_RE = re.compile(r"^(?:key |main |core )?(?:features?(?: overview)?|what (?:it|this project) does|capabilities|highlights)$", re.IGNORECASE)
OVERVIEW_SECTION_RE = re.compile(r"^(?:overview|about|introduction|summary)$", re.IGNORECASE)
NON_PROSE_START_RE = re.compile(r"^(?:[|>#<!+-]|\*(?!\*)|\d+[.)]\s|-{3,}|\*{3,})")
PURPOSE_SECTION_RE = re.compile(r"\b(?:overview|about|introduction|summary|problem|purpose|motivation|why)\b", re.IGNORECASE)
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+)$")
OFFICIAL_DOC_NAMES = {"readme.md", "install.md", "installation.md", "setup.md", "quickstart.md", "getting_started.md", "getting-started.md"}


class RepoFactsCollector:
    """Collect deterministic facts that summaries and answers can trust."""

    def collect(self, repo_url: str, records: list[FileRecord], graph: dict[str, Any]) -> dict[str, Any]:
        by_name: dict[str, FileRecord] = {}
        for record in sorted(records, key=lambda item: (item.path.count("/"), item.path)):
            by_name.setdefault(Path(record.path).name.lower(), record)
        languages = Counter(record.language for record in records if record.language not in {"unknown", "text"})
        readmes = [record for record in records if Path(record.path).name.lower().startswith("readme")]
        folders = Counter(self._top_folder(record.path) for record in records)
        source_dirs = sorted(folder for folder in folders if folder.split("/", 1)[0] in SOURCE_DIR_NAMES)
        test_dirs = sorted(folder for folder in folders if folder.split("/", 1)[0] in TEST_DIR_NAMES)
        docs_dirs = sorted(folder for folder in folders if folder.split("/", 1)[0] in DOC_DIR_NAMES)
        dependency_files = self._files_by_name(records, DEPENDENCY_NAMES)
        package_files = self._files_by_name(records, PACKAGE_NAMES)
        deployment_files = self._files_by_name(records, DEPLOYMENT_NAMES)
        config_files = [record.path for record in records if record.role == "config"]
        scripts = self._scripts(records)
        official_docs = self._official_docs(records)
        markdown_instructions = self._extract_markdown_instructions(official_docs)
        helper_scripts = self._helper_scripts(scripts, official_docs)
        entrypoints = self._entrypoints(records, graph, scripts)
        api_route_files = self._api_route_files(graph)
        cli_files = self._cli_files(records)
        important_files = self._important_files(records, entrypoints, api_route_files, graph)
        run_instructions = self._run_instructions(by_name, markdown_instructions)
        title = self._readme_title(readmes) or repo_url.rstrip("/").split("/")[-1]

        facts = {
            "repo_name": repo_url.rstrip("/").split("/")[-1],
            "project_title": title,
            "file_tree": self._file_tree(records),
            "readme_paths": [record.path for record in readmes],
            "official_doc_paths": [record.path for record in official_docs],
            "install_doc_paths": [record.path for record in official_docs if self._doc_priority(record.path) == 1],
            "run_doc_paths": sorted({item["source"] for item in run_instructions if item.get("official")}),
            "run_doc_sections": markdown_instructions["sections"],
            "official_helper_commands": markdown_instructions["helper_commands"],
            "readme_summary": self._readme_summary(readmes),
            "readme_has_overview": self._readme_overview(readmes) is not None,
            "readme_features": self._readme_features(readmes),
            "languages": [{"language": lang, "files": count} for lang, count in languages.most_common()],
            "dependency_files": dependency_files,
            "config_files": sorted(config_files)[:30],
            "package_files": package_files,
            "source_dirs": source_dirs,
            "test_dirs": test_dirs,
            "docs_dirs": docs_dirs,
            "scripts": scripts,
            "executable_scripts": scripts,
            "helper_scripts": helper_scripts,
            "main_entrypoints": entrypoints,
            "important_modules": self._important_modules(records, graph),
            "cli_files": cli_files,
            "api_route_files": api_route_files,
            "deployment_files": deployment_files,
            "important_files": important_files,
            "run_instructions": run_instructions,
            "target_audience": self._target_audience(readmes, records),
            "frameworks": self._frameworks(records, graph),
            "top_folders": [{"path": folder, "files": count, "purpose": self._folder_purpose(folder)} for folder, count in folders.most_common(12)],
            "test_files": sorted(record.path for record in records if record.role == "test")[:300],
            "code_file_count": sum(1 for record in records if record.role == "code"),
            "test_file_count": sum(1 for record in records if record.role == "test"),
            "has_tests": bool(test_dirs or any(record.role == "test" for record in records)),
            "has_readme": bool(readmes),
            "has_run_instructions": any(item["confidence"] in {"high", "medium"} and item["command"] != "Not clearly documented." for item in run_instructions),
        }
        return facts

    def _files_by_name(self, records: list[FileRecord], names: set[str]) -> list[str]:
        return sorted(record.path for record in records if Path(record.path).name.lower() in names)

    def _top_folder(self, path: str) -> str:
        parts = Path(path).parts
        if len(parts) == 1:
            return "."
        return parts[0]

    def _file_tree(self, records: list[FileRecord]) -> list[str]:
        paths = sorted(record.path for record in records)
        shallow = [path for path in paths if len(Path(path).parts) <= 3]
        return (shallow[:80] if shallow else paths[:80])

    def _readme_title(self, readmes: list[FileRecord]) -> str | None:
        for record in readmes:
            for line in record.text.splitlines()[:80]:
                clean = line.strip()
                if clean.startswith("#"):
                    title = clean.lstrip("#").strip()
                    if title and not README_NOISE.search(title):
                        return title
        return None

    def _readme_summary(self, readmes: list[FileRecord]) -> str:
        overview = self._readme_overview(readmes)
        if overview:
            return overview
        title = self._readme_title(readmes)
        if title:
            return f"The README identifies the project as {title}, but does not provide a strong prose overview. RepoMind infers the purpose from files, entrypoints, and package metadata."
        return "No substantive README description was found; the project description is inferred from files and metadata."

    def _readme_overview(self, readmes: list[FileRecord]) -> str | None:
        """The README paragraph that best describes the project, if any."""
        for record in readmes:
            intro = self._intro_paragraph(record)
            if intro:
                return intro
        for record in readmes:
            candidates = []
            for index, paragraph in enumerate(re.split(r"\n\s*\n", record.text)):
                if "```" in paragraph:
                    continue
                clean = self._clean_markdown(paragraph)
                if len(clean) < 60 or README_NOISE.search(clean) or self._looks_like_command_block(clean) or self._looks_like_tag_line(clean):
                    continue
                lower = clean.lower()
                score = sum(word in lower for word in ("project", "tool", "agent", "package", "test", "analy", "run", "build", "cli", "validation", "verifies", "automated"))
                if index <= 2:
                    score += 5
                if lower.startswith(("setup", "dependencies", "usage", "log format", "llm configuration", "interpretation")):
                    score -= 4
                candidates.append((score, clean))
            if candidates:
                candidates.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
                return candidates[0][1][:600]
        return None

    def _intro_paragraph(self, record: FileRecord) -> str | None:
        """First prose paragraph under the title or an overview/purpose heading, where READMEs usually say what the project is."""
        for section in self._markdown_sections(record):
            heading_line = section["lines"][0] if section["lines"] else ""
            is_title = bool(re.match(r"^\s{0,3}#\s", heading_line))
            if not is_title and not PURPOSE_SECTION_RE.search(section["heading"]):
                continue
            body = "\n".join(section["lines"][1:] if re.match(r"^\s{0,3}#", heading_line) else section["lines"])
            for paragraph in re.split(r"\n\s*\n", body):
                first = paragraph.strip().splitlines()[0].strip() if paragraph.strip() else ""
                if not first or "```" in paragraph or NON_PROSE_START_RE.match(first):
                    continue
                if re.fullmatch(r"(\*\*|__|\*|_).+?\1", first):
                    continue  # bold/italic tagline such as "**Solar Plant Feasibility Analysis**"
                clean = self._clean_markdown(paragraph)
                if len(clean) >= 60 and not README_NOISE.search(clean) and not self._looks_like_command_block(clean) and not self._looks_like_tag_line(clean):
                    return clean[:600]
        return None

    def _looks_like_tag_line(self, text: str) -> bool:
        """Badge-style lines such as "`Python` · `XGBoost` · `Streamlit`" carry no prose."""
        prose = re.sub(r"`[^`]*`", "", text)
        return len(re.findall(r"[A-Za-z]{3,}", prose)) < 6

    def _readme_features(self, readmes: list[FileRecord], limit: int = 8) -> list[dict[str, Any]]:
        """Bullet points or table rows the README lists under a features heading, else under an overview heading."""
        for heading_re in (FEATURE_SECTION_RE, OVERVIEW_SECTION_RE):
            features = self._readme_list_items(readmes, heading_re, limit)
            if features:
                return features
        return []

    def _readme_list_items(self, readmes: list[FileRecord], heading_re: re.Pattern[str], limit: int) -> list[dict[str, Any]]:
        features: list[dict[str, Any]] = []
        for record in readmes:
            lines = record.text.splitlines()
            for section in self._markdown_sections(record):
                heading = re.sub(r"^[^A-Za-z]+", "", section["heading"]).strip().rstrip(":")
                if not heading_re.match(heading):
                    continue
                in_fence = False
                for line_number in range(section["start_line"] + 1, section.get("end_line", len(lines)) + 1):
                    line = lines[line_number - 1]
                    if line.strip().startswith("```"):
                        in_fence = not in_fence
                        continue
                    next_line = lines[line_number] if line_number < len(lines) else ""
                    is_table_header = line.strip().startswith("|") and bool(TABLE_SEPARATOR_RE.match(next_line))
                    text = "" if in_fence or is_table_header else self._feature_text(line)
                    if text:
                        features.append({"text": text, "source": record.path, "line": line_number})
                    if len(features) >= limit:
                        return features
            if features:
                return features
        return features

    def _feature_text(self, line: str) -> str:
        item = LIST_ITEM_RE.match(line)
        if item:
            text = item.group(1)
        elif line.strip().startswith("|"):
            if TABLE_SEPARATOR_RE.match(line):
                return ""
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            text = " — ".join(cell for cell in cells[:2] if cell)
        else:
            return ""
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
        text = re.sub(r"(\*\*|__|`)", "", text).strip()
        if len(text) < 8 or README_NOISE.search(text):
            return ""
        return text[:240]

    def _clean_markdown(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|") or stripped.startswith("!["):
                continue
            stripped = re.sub(r"^#+\s*", "", stripped)
            stripped = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", stripped)
            lines.append(stripped)
        return " ".join(lines).strip()

    def _looks_like_command_block(self, text: str) -> bool:
        command_markers = ("python ", "pip ", "npm ", "bash ", "docker ", "$ ", "sudo ", "./")
        words = text.split()
        return len(words) < 30 and any(marker in text.lower() for marker in command_markers)

    def _scripts(self, records: list[FileRecord]) -> list[str]:
        scripts = []
        for record in records:
            path = Path(record.path)
            text_head = record.text[:120]
            if path.suffix.lower() in SCRIPT_EXTS or text_head.startswith("#!"):
                scripts.append(record.path)
        return sorted(scripts)[:30]

    def _entrypoints(self, records: list[FileRecord], graph: dict[str, Any], scripts: list[str]) -> list[dict[str, str]]:
        entries: list[dict[str, str]] = []
        common_names = {"main.py", "app.py", "server.py", "manage.py", "wsgi.py", "asgi.py", "index.js", "index.ts", "server.js", "server.ts", "main.go", "main.rs", "cli.py", "__main__.py"}
        for record in records:
            name = Path(record.path).name.lower()
            if name in common_names and record.role == "code":
                reason = self._entrypoint_reason(record, name)
                entries.append({"path": record.path, "reason": reason, "kind": self._entrypoint_kind(record, name)})
        for node in graph.get("nodes", []):
            if node.get("kind") == "entrypoint" and node.get("file_path") and not self._is_test_path(node["file_path"], records):
                entries.append({"path": node["file_path"], "reason": "Runs directly because it contains a Python `if __name__ == '__main__'` guard.", "kind": "script"})
            if node.get("kind") == "route" and node.get("file_path") and not self._is_test_path(node["file_path"], records):
                entries.append({"path": node["file_path"], "reason": "Defines web/API route handlers that receive user or HTTP requests.", "kind": "web_api"})
        return self._dedupe_dicts(entries, "path")[:20]

    def _is_test_path(self, path: str, records: list[FileRecord]) -> bool:
        return any(record.path == path and record.role == "test" for record in records)

    def _entrypoint_kind(self, record: FileRecord, name: str) -> str:
        head = record.text[:20000]
        if UI_MARKERS.search(head):
            return "streamlit_ui"
        if WEB_MARKERS.search(head):
            return "web_api"
        if CLI_MARKERS.search(head) or name in {"cli.py", "__main__.py"}:
            return "cli"
        if re.search(r"^if\s+__name__\s*==\s*['\"]__main__['\"]", head, re.MULTILINE):
            return "script"
        if name in {"app.py", "server.py", "index.js"}:
            return "application"
        return "entrypoint"

    def _entrypoint_reason(self, record: FileRecord, name: str) -> str:
        kind = self._entrypoint_kind(record, name)
        if kind == "streamlit_ui":
            return "Likely starts the interactive UI because it imports or calls Streamlit."
        if kind == "web_api":
            return "Likely starts the web/API layer because it creates an app object or declares HTTP routes."
        if kind == "cli":
            return "Likely starts a command-line workflow because it uses CLI parsing conventions."
        if kind == "script":
            return "Can be run directly because it contains a Python `__main__` guard."
        if kind == "application":
            return f"Common application entry filename `{name}` with top-level application code."
        return f"Common executable entry filename `{name}`."

    def _api_route_files(self, graph: dict[str, Any]) -> list[str]:
        files = sorted({node.get("file_path") for node in graph.get("nodes", []) if node.get("kind") == "route" and node.get("file_path")})
        return [file for file in files if file]

    def _cli_files(self, records: list[FileRecord]) -> list[str]:
        cli_files = []
        for record in records:
            name = Path(record.path).name.lower()
            if record.role != "code" or record.language not in {"python", "javascript", "typescript", "go", "rust"}:
                continue
            if CLI_MARKERS.search(record.text[:20000]) or re.search(r"(^|[_-])cli([_.-]|$)", name):
                cli_files.append(record.path)
        return sorted(set(cli_files))[:20]

    def _official_docs(self, records: list[FileRecord]) -> list[FileRecord]:
        docs = []
        for record in records:
            name = Path(record.path).name.lower()
            lower = record.path.lower()
            top_level = "/" not in record.path
            if name in {"install.md", "installation.md", "setup.md", "quickstart.md", "getting_started.md", "getting-started.md"}:
                docs.append(record)
            elif name == "readme.md" and top_level:
                docs.append(record)
            elif lower.startswith("docs/") and RUN_SECTION_RE.search(name):
                docs.append(record)
        docs.sort(key=lambda record: (self._doc_priority(record.path), record.path.lower()))
        return docs

    def _doc_priority(self, path: str) -> int:
        name = Path(path).name.lower()
        lower = path.lower()
        if name in {"install.md", "installation.md"}:
            return 1
        if name == "readme.md":
            return 2
        if lower.startswith("docs/") and RUN_SECTION_RE.search(name):
            return 3
        if name in {"setup.md", "quickstart.md", "getting_started.md", "getting-started.md"}:
            return 3
        return 9

    def _extract_markdown_instructions(self, docs: list[FileRecord]) -> dict[str, list[dict[str, Any]]]:
        commands: list[dict[str, Any]] = []
        sections: list[dict[str, Any]] = []
        for record in docs:
            parsed_sections = self._markdown_sections(record)
            for section in parsed_sections:
                is_install_doc = self._doc_priority(record.path) == 1
                relevant = is_install_doc or RUN_SECTION_RE.search(section["heading"] or "") is not None
                if not relevant:
                    continue
                section_excerpt = self._clean_markdown("\n".join(section["lines"]))[:500]
                sections.append(
                    {
                        "source": record.path,
                        "heading": section["heading"] or Path(record.path).name,
                        "start_line": section["start_line"],
                        "end_line": section["end_line"],
                        "excerpt": section_excerpt,
                        "official": True,
                    }
                )
                for command in section["commands"]:
                    command_row = {
                        "command": command["command"],
                        "source": record.path,
                        "heading": section["heading"] or Path(record.path).name,
                        "start_line": command["start_line"],
                        "end_line": command["end_line"],
                        "confidence": "high" if is_install_doc else "medium",
                        "official": True,
                    }
                    command_row["helper"] = self._is_helper_doc_command(command_row)
                    commands.append(command_row)
        deduped = self._dedupe_dicts(commands, "command")
        return {
            "commands": [item for item in deduped if not item.get("helper")],
            "helper_commands": [item for item in deduped if item.get("helper")],
            "sections": sections,
        }

    def _is_helper_doc_command(self, command: dict[str, Any]) -> bool:
        heading = str(command.get("heading", "")).lower()
        text = str(command.get("command", "")).lower()
        if re.search(r"\b(optional|demo|benchmarks?|smoke|troubleshoot(?:ing)?|acceptance|profil(?:e|ing)|contribut(?:e|ing)|development)\b", heading):
            return True
        return bool(re.search(r"(^|[\s/])(benchmarks?|devtools|tools/dev|perf)/|smoke[_-]test|\bbenchmark", text))

    def _markdown_sections(self, record: FileRecord) -> list[dict[str, Any]]:
        lines = record.text.splitlines()
        sections: list[dict[str, Any]] = []
        current = {"heading": Path(record.path).name, "start_line": 1, "lines": [], "commands": []}
        in_fence = False
        fence_start = 0
        fence_lines: list[tuple[int, str]] = []
        for index, line in enumerate(lines, start=1):
            heading = re.match(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$", line)
            if heading and not in_fence:
                if current["lines"] or current["commands"]:
                    current["end_line"] = index - 1
                    sections.append(current)
                current = {"heading": heading.group(2).strip(), "start_line": index, "lines": [line], "commands": []}
                continue
            current["lines"].append(line)
            if line.strip().startswith("```"):
                if not in_fence:
                    in_fence = True
                    fence_start = index
                    fence_lines = []
                else:
                    current["commands"].extend(self._commands_from_block(fence_lines, fence_start, index))
                    in_fence = False
                continue
            if in_fence:
                fence_lines.append((index, line))
        if current["lines"] or current["commands"]:
            current["end_line"] = len(lines) or 1
            sections.append(current)
        return sections

    def _commands_from_block(self, lines: list[tuple[int, str]], start_line: int, end_line: int) -> list[dict[str, Any]]:
        commands = []
        current_parts: list[str] = []
        command_start = start_line
        for line_number, raw in lines:
            line = SHELL_PROMPT_RE.sub("", raw.strip())
            if not line or line.startswith("#"):
                continue
            if line.endswith("\\"):
                if not current_parts:
                    command_start = line_number
                current_parts.append(line[:-1].strip())
                continue
            if current_parts:
                current_parts.append(line)
                command = " ".join(current_parts).strip()
                current_parts = []
                if self._looks_like_shell_command(command):
                    commands.append({"command": command, "start_line": command_start, "end_line": line_number})
            elif self._looks_like_shell_command(line):
                commands.append({"command": line, "start_line": line_number, "end_line": line_number})
        return commands

    def _looks_like_shell_command(self, line: str) -> bool:
        if not line or line.startswith("{") or line.startswith("["):
            return False
        return COMMAND_START_RE.search(line) is not None

    def _important_files(self, records: list[FileRecord], entrypoints: list[dict[str, str]], api_route_files: list[str], graph: dict[str, Any]) -> list[dict[str, str]]:
        important: list[dict[str, str]] = []
        important_names = {
            "readme.md": "Project documentation and onboarding.",
            "install.md": "Official installation and setup instructions.",
            "installation.md": "Official installation and setup instructions.",
            "setup.md": "Setup instructions or environment guidance.",
            "quickstart.md": "Quickstart instructions.",
            "pyproject.toml": "Python packaging, dependencies, and tooling.",
            "requirements.txt": "Python runtime dependencies.",
            "package.json": "JavaScript package metadata and scripts.",
            "makefile": "Common build/run automation.",
            "dockerfile": "Container deployment entrypoint.",
            ".env.example": "Environment variable documentation.",
        }
        for record in sorted(records, key=lambda item: (item.path.count("/"), item.path)):
            name = Path(record.path).name.lower()
            if name in important_names and record.path.count("/") <= 1:
                important.append({"path": record.path, "role": record.role, "reason": important_names[name]})
        for item in entrypoints:
            important.append({"path": item["path"], "role": "entrypoint", "reason": item["reason"]})
        for path in api_route_files:
            important.append({"path": path, "role": "api", "reason": "Defines detected API routes."})
        for item in graph.get("central_files", [])[:8]:
            important.append({"path": item["path"], "role": "core", "reason": f"Central module: imported/called by {item['in_degree']} other file(s)."})
        return self._dedupe_dicts(important, "path")[:30]

    def _run_instructions(self, by_name: dict[str, FileRecord], markdown_instructions: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        official_commands = markdown_instructions["commands"]
        if official_commands:
            return official_commands[:25]

        instructions: list[dict[str, Any]] = []
        if "requirements.txt" in by_name:
            instructions.append({"command": "python -m pip install -r requirements.txt", "source": "requirements.txt", "heading": "Inferred from dependency file", "confidence": "medium", "official": False})
        if "pyproject.toml" in by_name:
            instructions.append({"command": "python -m pip install -e .", "source": "pyproject.toml", "heading": "Inferred from Python package metadata", "confidence": "medium", "official": False})
        if "package.json" in by_name:
            instructions.append({"command": "npm install", "source": "package.json", "heading": "Inferred from JavaScript package metadata", "confidence": "medium", "official": False})
            try:
                package = json.loads(by_name["package.json"].text)
                for script in package.get("scripts", {}):
                    instructions.append({"command": f"npm run {script}", "source": "package.json", "heading": "Inferred from package.json scripts", "confidence": "medium", "official": False})
            except json.JSONDecodeError:
                pass
        if "makefile" in by_name:
            for line in by_name["makefile"].text.splitlines():
                match = re.match(r"^([A-Za-z0-9_.-]+):", line)
                if match and not match.group(1).startswith("."):
                    instructions.append({"command": f"make {match.group(1)}", "source": "Makefile", "heading": "Inferred from Makefile target", "confidence": "medium", "official": False})
        if not instructions:
            instructions.append({"command": "Not clearly documented.", "source": "repo scan", "heading": "No official docs found", "confidence": "low", "official": False})
        return self._dedupe_dicts(instructions, "command")[:15]

    def _helper_scripts(self, scripts: list[str], official_docs: list[FileRecord]) -> list[dict[str, Any]]:
        official_text = "\n".join(record.text for record in official_docs).lower()
        helpers = []
        for script in scripts:
            lower = script.lower()
            mentioned = lower in official_text or Path(script).name.lower() in official_text
            category = "helper"
            if "benchmark" in lower or "measure" in lower:
                category = "benchmark"
            elif "smoke" in lower or "test" in lower:
                category = "testing helper"
            elif "devtool" in lower or "demo" in lower or "vm" in lower:
                category = "developer/VM helper"
            helpers.append(
                {
                    "path": script,
                    "category": category,
                    "mentioned_in_official_docs": mentioned,
                    "note": "Official docs mention this script." if mentioned else "Detected helper script; not treated as an official install/run command.",
                }
            )
        return helpers

    def _important_modules(self, records: list[FileRecord], graph: dict[str, Any]) -> list[dict[str, str | int]]:
        definition_counts: Counter[str] = Counter()
        import_counts: Counter[str] = Counter()
        route_counts: Counter[str] = Counter()
        dependency_counts: Counter[str] = Counter()
        record_by_path = {record.path: record for record in records}
        for edge in graph.get("edges", []):
            if edge.get("relation", "").startswith("defines_") and str(edge.get("source", "")).startswith("file:"):
                definition_counts[str(edge["source"]).removeprefix("file:")] += 1
            if edge.get("relation") == "imports" and str(edge.get("source", "")).startswith("file:"):
                import_counts[str(edge["source"]).removeprefix("file:")] += 1
            if edge.get("relation") == "declares_route" and str(edge.get("source", "")).startswith("file:"):
                route_counts[str(edge["source"]).removeprefix("file:")] += 1
            if edge.get("relation") == "depends_on_package" and str(edge.get("source", "")).startswith("file:"):
                dependency_counts[str(edge["source"]).removeprefix("file:")] += 1
        modules = []
        signal_counts = definition_counts + import_counts + route_counts + dependency_counts
        for path, count in signal_counts.most_common(20):
            record = record_by_path.get(path)
            if record and record.role not in {"code", "test"}:
                continue
            modules.append(
                {
                    "path": path,
                    "signals": count,
                    "reason": self._module_reason(path, record, definition_counts[path], import_counts[path], route_counts[path], dependency_counts[path]),
                }
            )
        if not modules:
            modules = [{"path": record.path, "signals": 1, "reason": self._module_reason(record.path, record, 0, 0, 0, 0)} for record in records if record.role == "code"][:12]
        return modules

    def _module_reason(
        self,
        path: str,
        record: FileRecord | None,
        definitions: int,
        imports: int,
        routes: int,
        dependencies: int,
    ) -> str:
        name = Path(path).name.lower()
        lower_path = path.lower()
        text = (record.text[:8000].lower() if record else "")
        details: list[str] = []
        if routes:
            details.append(f"defines {routes} web/API route handler(s)")
        if "evaluate" in lower_path or "metric" in lower_path:
            details.append("evaluates outputs or reports metrics")
        if re.search(r"^\s*(?:import|from)\s+(?:torch|tensorflow|keras|sklearn|xgboost|lightgbm|transformers)\b", text, re.MULTILINE) or re.search(r"\.(?:fit|predict)\(", text):
            details.append("contains model or ML inference/training logic")
        if "preprocess" in lower_path or "transform" in lower_path or "clean" in lower_path:
            details.append("prepares or transforms input data before core processing")
        if "config" in lower_path or name in {"settings.py", "config.py"}:
            details.append("centralizes configuration used by other modules")
        if name in {"api.py", "server.py", "app.py"}:
            details.append("connects the application surface to backend logic")
        if definitions:
            details.append(f"defines {definitions} function/class symbol(s)")
        if imports:
            details.append(f"imports {imports} module/dependency reference(s)")
        if dependencies:
            details.append(f"declares {dependencies} package dependency signal(s)")
        if not details:
            details.append("source file detected with analyzable code")
        return "; ".join(details[:3]) + "."

    def _frameworks(self, records: list[FileRecord], graph: dict[str, Any]) -> list[str]:
        """Frameworks come from declared/imported dependencies, never from raw substrings."""
        dependencies = {str(dep).lower() for dep in graph.get("dependencies", [])}
        dependencies |= {str(node.get("label", "")).lower() for node in graph.get("nodes", []) if node.get("kind") == "dependency"}
        dependencies |= {str(item.get("dependency", "")).lower() for item in graph.get("top_dependencies", [])}
        found = [label for label, packages in FRAMEWORK_PACKAGES.items() if dependencies & packages]
        if any(Path(record.path).name.lower() == "dockerfile" for record in records):
            found.append("Docker")
        return found

    def _target_audience(self, readmes: list[FileRecord], records: list[FileRecord]) -> str:
        text = " ".join(record.text[:4000] for record in readmes)
        code_head = "\n".join(record.text[:4000] for record in records if record.role == "code")
        if re.search(r"\b(?:CLI|command[- ]line)\b", text) or CLI_MARKERS.search(code_head):
            return "developers or operators using command-line tooling"
        if re.search(r"\b(?:API|REST|endpoints?|HTTP)\b", text) or WEB_MARKERS.search(code_head):
            return "developers integrating or operating an API service"
        if re.search(r"\b(?:library|SDK|package)\b", text, re.IGNORECASE):
            return "developers consuming a library or SDK"
        if UI_MARKERS.search(code_head):
            return "end users of an interactive data/ML app"
        return "developers or maintainers of the repository"

    def _folder_purpose(self, folder: str) -> str:
        root = folder.split("/", 1)[0]
        purposes = {
            ".": "top-level docs and project metadata",
            "src": "source package code",
            "app": "application source code",
            "tests": "automated tests",
            "docs": "documentation",
            "scripts": "automation scripts",
            "packaging": "packaging or distribution assets",
            "lib": "library code",
            "packages": "workspace packages",
            "examples": "usage examples",
            ".github": "CI/CD and repository automation",
            "migrations": "database migrations",
        }
        return purposes.get(root, "project module or supporting assets")

    def _dedupe_dicts(self, items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
        seen = set()
        deduped = []
        for item in items:
            value = item.get(key)
            if value in seen:
                continue
            seen.add(value)
            deduped.append(item)
        return deduped
