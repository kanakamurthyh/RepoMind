from __future__ import annotations

from typing import Any

from app.analysis.quality import QualityScorer
from app.models import FileRecord, Weakness


class SummaryGenerator:
    def __init__(self) -> None:
        self.quality = QualityScorer()

    def generate(
        self,
        repo_url: str,
        records: list[FileRecord],
        graph: dict[str, Any],
        weaknesses: list[Weakness],
        facts: dict[str, Any],
        insights: dict[str, Any] | None = None,
        coverage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        insights = insights or {}
        quality_score = self.quality.score(facts, weaknesses, graph, insights)
        summary = {
            "project_identity": {
                "project_name": facts["project_title"],
                "repo_name": facts["repo_name"],
                "one_line_summary": self._one_line(facts),
                "what_this_repo_is_for": self._purpose(facts),
                "target": facts["target_audience"],
                "about": self._about(facts),
            },
            "tech_stack": self._tech_stack(facts),
            "main_structure": facts["top_folders"],
            "important_files": facts["important_files"],
            "how_it_likely_works": self._how_it_works(facts, graph),
            "run_instructions": facts["run_instructions"],
            "architecture": {
                "explanation": self._architecture_explanation(facts, graph),
                "entrypoint_flow": self._entrypoint_flow(facts),
                "entrypoint_flow_basis": "suggested reading order inferred from file roles, not a verified runtime call chain",
                "important_modules": facts["important_modules"][:12],
                "mermaid": self._architecture_mermaid(facts),
                "mermaid_basis": "conceptual overview from file roles; arrows are not verified call relationships",
                "dependency_mermaid": graph.get("mermaid", ""),
                "central_files": graph.get("central_files", [])[:10],
                "dependency_cycles": graph.get("dependency_cycles", [])[:5],
            },
            "strengths": self._strengths(facts, graph),
            "risks": [w.to_dict() for w in weaknesses if w.severity in {"high", "medium"}][:10],
            "suggested_improvements": self._improvements(weaknesses, quality_score),
            "quality_score": quality_score,
            "suggested_questions": [
                "Explain this project like I am new to the codebase.",
                "How do I run this repository locally?",
                "Which files should I read first?",
                "What are the main risks or weaknesses?",
                "How does the main execution flow work?",
            ],
            "repo_facts": facts,
            "graph_metrics": {
                "nodes": graph.get("node_count", 0),
                "edges": graph.get("edge_count", 0),
                "node_kinds": graph.get("node_kinds", {}),
                "edge_relations": graph.get("edge_relations", {}),
                "top_dependencies": graph.get("top_dependencies", []),
                "unresolved_calls": graph.get("unresolved_calls", 0),
                "parser_backends": graph.get("parser_backends", {}),
            },
            "insights": self._insights(insights, graph),
            "legacy": self._legacy_fields(facts, weaknesses, quality_score),
        }
        if coverage is not None:
            # Which files the scan/chunk caps left out; see app/ingestion/coverage.py.
            summary["coverage"] = coverage
        return summary

    def _insights(self, insights: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
        drift = insights.get("drift") or {}
        deps = insights.get("dependencies") or {}
        history = insights.get("history") or {}
        return {
            "docs_drift": {
                "docs_accuracy": drift.get("docs_accuracy"),
                "issue_count": drift.get("issue_count", 0),
                "top_issues": drift.get("issues", [])[:8],
            },
            "dependencies": {
                "scan_status": deps.get("scan_status", "skipped"),
                "packages_scanned": deps.get("packages_scanned", 0),
                "inventory_packages": deps.get("inventory_packages", 0),
                "inventory_truncated": deps.get("inventory_truncated", False),
                "inventory_warnings": deps.get("inventory_warnings", []),
                "vulnerabilities": deps.get("vulnerabilities", [])[:15],
                "unpinned_count": deps.get("unpinned_count", len(deps.get("unpinned", []))),
                "note": deps.get("note", ""),
            },
            "history": {
                "status": history.get("status", "unavailable"),
                "commits_analyzed": history.get("commits_analyzed", 0),
                "bus_factor": history.get("bus_factor"),
                "hotspots": history.get("hotspots", [])[:10],
                "note": history.get("note", ""),
            },
            "code_health": {
                "complex_functions": graph.get("complex_functions", [])[:10],
                "unreferenced_private_functions": graph.get("unreferenced_private_functions", [])[:10],
                "dependency_cycles": graph.get("dependency_cycles", [])[:5],
            },
        }

    def _one_line(self, facts: dict[str, Any]) -> str:
        languages = ", ".join(self._display_name(item["language"]) for item in facts.get("languages", [])[:3])
        role = "repository"
        if facts.get("cli_files"):
            role = "CLI/tooling repository"
        elif facts.get("api_route_files"):
            role = "API application"
        elif facts.get("frameworks"):
            role = f"{facts['frameworks'][0]} project"
        if languages:
            article = "an" if role[0].lower() in {"a", "e", "i", "o", "u"} else "a"
            return f"{facts['project_title']} appears to be {article} {role} built mainly with {languages}."
        return f"{facts['project_title']} appears to be a software repository with limited detected source metadata."

    def _purpose(self, facts: dict[str, Any]) -> str:
        readme_summary = facts.get("readme_summary", "")
        inferred = []
        if facts.get("cli_files"):
            inferred.append("command-line workflows")
        if facts.get("api_route_files"):
            inferred.append("API endpoints")
        if facts.get("package_files"):
            inferred.append("installable/package-managed code")
        if facts.get("test_dirs"):
            inferred.append("testable project components")
        inferred_text = ", ".join(inferred) if inferred else "the files and project metadata found in the repo"
        return f"{readme_summary} Based on the repository structure, it appears to target {inferred_text}."

    def _about(self, facts: dict[str, Any]) -> dict[str, Any]:
        """Plain-language project description; README statements are kept apart from structural inference."""
        readme_paths = facts.get("readme_paths", [])
        readme_features = facts.get("readme_features", [])
        has_readme_purpose = bool(readme_paths) and facts.get("readme_has_overview", False)
        return {
            "what_it_is": self._one_line(facts),
            "why_it_exists": facts.get("readme_summary", "") if has_readme_purpose else "",
            "why_it_exists_source": readme_paths[0] if has_readme_purpose else None,
            "what_it_does": [{"text": item["text"], "source": f"{item['source']}:{item['line']}"} for item in readme_features]
            or [{"text": text, "source": "inferred from repository structure"} for text in self._inferred_capabilities(facts)],
            "what_it_does_basis": "readme" if readme_features else "inferred",
            "who_it_is_for": facts.get("target_audience", ""),
        }

    def _inferred_capabilities(self, facts: dict[str, Any]) -> list[str]:
        capabilities = []
        if facts.get("api_route_files"):
            capabilities.append(f"Serves an HTTP API (routes defined in {self._paths(facts['api_route_files'])}).")
        if facts.get("cli_files"):
            capabilities.append(f"Provides a command-line interface ({self._paths(facts['cli_files'])}).")
        ui_frameworks = [name for name in facts.get("frameworks", []) if name in {"Streamlit", "Gradio", "React", "Next.js", "Vue", "Angular", "Svelte"}]
        if ui_frameworks:
            capabilities.append(f"Ships a user interface built with {', '.join(ui_frameworks)}.")
        ml_frameworks = [name for name in facts.get("frameworks", []) if name in {"PyTorch", "TensorFlow", "scikit-learn", "Hugging Face Transformers", "LangChain", "LlamaIndex"}]
        if ml_frameworks:
            capabilities.append(f"Uses machine-learning / AI tooling ({', '.join(ml_frameworks)}).")
        if facts.get("main_entrypoints") and not capabilities:
            capabilities.append(f"Runs from {self._paths([entry['path'] for entry in facts['main_entrypoints']])}.")
        if facts.get("deployment_files"):
            capabilities.append(f"Includes deployment configuration ({self._paths(facts['deployment_files'])}).")
        if facts.get("has_tests"):
            capabilities.append(f"Has an automated test suite ({facts.get('test_file_count', 0)} test file(s)).")
        return capabilities

    def _paths(self, paths: list[str], limit: int = 3) -> str:
        return ", ".join(f"`{path}`" for path in paths[:limit])

    def _tech_stack(self, facts: dict[str, Any]) -> list[dict[str, str | int]]:
        rows: list[dict[str, str | int]] = []
        for item in facts.get("languages", []):
            rows.append({"name": self._display_name(item["language"]), "type": "language", "evidence": f"{item['files']} file(s)"})
        for framework in facts.get("frameworks", []):
            rows.append({"name": framework, "type": "framework/tooling", "evidence": "detected in code, docs, or config"})
        for path in facts.get("dependency_files", []):
            rows.append({"name": path, "type": "dependency file", "evidence": "package/dependency metadata"})
        return rows or [{"name": "Not clearly detected", "type": "unknown", "evidence": "repo has little analyzable source metadata"}]

    def _how_it_works(self, facts: dict[str, Any], graph: dict[str, Any]) -> list[str]:
        steps = []
        if facts.get("main_entrypoints"):
            entry = facts["main_entrypoints"][0]
            steps.append(f"Execution likely starts around `{entry['path']}` because {entry['reason'].rstrip('.')}.")
        if facts.get("cli_files"):
            steps.append(f"CLI-oriented logic appears in {', '.join(f'`{path}`' for path in facts['cli_files'][:3])}.")
        if facts.get("important_modules"):
            modules = ", ".join(f"`{item['path']}`" for item in facts["important_modules"][:4])
            steps.append(f"The main implementation is concentrated in {modules}.")
        if facts.get("dependency_files"):
            steps.append(f"Dependencies and package metadata are declared in {', '.join(f'`{p}`' for p in facts['dependency_files'][:4])}.")
        if facts.get("test_dirs"):
            steps.append(f"Tests are present under {', '.join(f'`{p}`' for p in facts['test_dirs'][:3])}.")
        if not steps:
            steps.append("The execution flow is not clearly documented; start from the README and the largest source files.")
        return steps

    def _architecture_explanation(self, facts: dict[str, Any], graph: dict[str, Any]) -> str:
        folders = facts.get("top_folders", [])[:5]
        folder_text = ", ".join(f"`{item['path']}` ({item['purpose']})" for item in folders)
        node_count = graph.get("node_count", 0)
        edge_count = graph.get("edge_count", 0)
        return f"The repository is organized around {folder_text or 'top-level files'}. The code graph contains {node_count} nodes and {edge_count} relationships, which RepoMind uses internally to connect files, imports, definitions, routes, and entry points."

    def _entrypoint_flow(self, facts: dict[str, Any]) -> list[str]:
        flow: list[str] = []
        official_docs = facts.get("run_doc_paths") or facts.get("install_doc_paths") or facts.get("readme_paths")
        if official_docs:
            flow.append(f"1. Read `{official_docs[0]}` for the documented setup/run path before inferring commands from code.")
        if facts.get("dependency_files"):
            deps = ", ".join(f"`{path}`" for path in facts["dependency_files"][:3])
            flow.append(f"2. Install dependencies declared in {deps}.")

        entries = self._prioritized_entrypoints(facts)
        if entries:
            for index, entry in enumerate(entries[:3], start=len(flow) + 1):
                flow.append(f"{index}. Start from `{entry['path']}`: {entry['reason']}")

        support_modules = self._support_modules(facts, {entry["path"] for entry in entries})
        if support_modules:
            module_text = "; ".join(f"`{item['path']}` {item['reason'].rstrip('.')}" for item in support_modules[:3])
            flow.append(f"{len(flow) + 1}. Inspect these supporting modules: {module_text}.")

        test_like = [path for path in facts.get("file_tree", []) if any(marker in path.lower() for marker in ("test", "evaluate", "eval"))]
        if test_like:
            flow.append(f"{len(flow) + 1}. Validate behavior with `{test_like[0]}` and related test/evaluation files.")
        return flow or ["No clear execution flow was detected from docs, entrypoint filenames, routes, or source relationships."]

    def _prioritized_entrypoints(self, facts: dict[str, Any]) -> list[dict[str, Any]]:
        priority = {"streamlit_ui": 0, "web_api": 1, "cli": 2, "script": 3, "application": 4, "entrypoint": 5}
        entries = list(facts.get("main_entrypoints", []))
        entries.sort(key=lambda item: (priority.get(item.get("kind", "entrypoint"), 9), item.get("path", "")))
        deduped: list[dict[str, Any]] = []
        seen = set()
        for entry in entries:
            path = entry.get("path")
            if not path or path in seen:
                continue
            seen.add(path)
            deduped.append(entry)
        return deduped

    def _support_modules(self, facts: dict[str, Any], excluded_paths: set[str]) -> list[dict[str, Any]]:
        modules = []
        for module in facts.get("important_modules", []):
            path = module.get("path")
            if path and path not in excluded_paths:
                modules.append(module)
        return modules

    def _architecture_mermaid(self, facts: dict[str, Any]) -> str:
        lines = ["flowchart TD", '    User["User / Developer"]']
        entrypoints = facts.get("main_entrypoints", [])[:3]
        modules = facts.get("important_modules", [])[:5]
        deps = facts.get("dependency_files", [])[:3]
        tests = facts.get("test_dirs", [])[:2]
        previous = "User"
        if entrypoints:
            for idx, entry in enumerate(entrypoints):
                node = f"Entry{idx}"
                lines.append(f'    {previous} --> {node}["{self._label(entry["path"])}"]')
                previous = node
        elif modules:
            lines.append('    User --> Source["Source files"]')
            previous = "Source"
        for idx, module in enumerate(modules):
            node = f"Module{idx}"
            lines.append(f'    {previous} --> {node}["{self._label(module["path"])}"]')
        for idx, dep in enumerate(deps):
            lines.append(f'    Config{idx}["{self._label(dep)}"] --> {previous}')
        for idx, test_dir in enumerate(tests):
            lines.append(f'    Tests{idx}["{self._label(test_dir)}"] --> {previous}')
        return "\n".join(lines)

    def _strengths(self, facts: dict[str, Any], graph: dict[str, Any]) -> list[str]:
        strengths = []
        if facts.get("has_readme"):
            strengths.append("Includes README documentation for onboarding.")
        if facts.get("dependency_files"):
            strengths.append("Declares dependencies or package metadata.")
        if facts.get("has_tests"):
            strengths.append("Contains test files or test directories.")
        if facts.get("main_entrypoints"):
            strengths.append("Has identifiable entrypoint files.")
        if graph.get("edge_count", 0) > 0:
            strengths.append("RepoMind could extract useful code relationships for graph-aware retrieval.")
        return strengths or ["The repository is small enough to inspect manually, but it needs stronger documentation signals."]

    def _improvements(self, weaknesses: list[Weakness], quality_score: dict[str, Any]) -> dict[str, list[str]]:
        categories = {
            "documentation": [],
            "testing": [],
            "architecture": [],
            "performance": [],
            "security": [],
            "maintainability": [],
            "deployment": [],
            "run_readiness": [],
        }
        mapping = {"run_readiness": "run_readiness", "portability": "maintainability"}
        for weakness in weaknesses:
            bucket = mapping.get(weakness.category, weakness.category)
            if bucket not in categories:
                bucket = "maintainability"
            categories[bucket].append(f"{weakness.title}: {weakness.suggested_fix}")
        categories["top_5"] = quality_score.get("top_improvements", [])
        return categories

    def _legacy_fields(self, facts: dict[str, Any], weaknesses: list[Weakness], quality_score: dict[str, Any]) -> dict[str, Any]:
        return {
            "project_name": facts["project_title"],
            "one_line_summary": self._one_line(facts),
            "what_it_does": self._purpose(facts),
            "tech_stack_detected": {
                "languages": {item["language"]: item["files"] for item in facts.get("languages", [])},
                "frameworks": facts.get("frameworks", []),
                "dependency_files": facts.get("dependency_files", []),
            },
            "main_folders": facts.get("top_folders", []),
            "main_entry_points": facts.get("main_entrypoints", []),
            "how_to_run": [item["command"] for item in facts.get("run_instructions", [])],
            "architecture_overview": self._architecture_explanation(facts, {}),
            "tests": "Test signals detected." if facts.get("has_tests") else "No test files were detected.",
            "weaknesses_or_risks": [w.to_dict() for w in weaknesses[:10]],
            "good_questions": [
                "Explain this project like I am new to the codebase.",
                "How do I run this repository locally?",
                "Which files should I read first?",
                "What are the main risks or weaknesses?",
            ],
        }

    def _label(self, value: str) -> str:
        return value.replace('"', "'")[:48]

    def _display_name(self, language: str) -> str:
        names = {"html": "HTML", "yaml": "YAML", "json": "JSON", "javascript": "JavaScript", "typescript": "TypeScript"}
        return names.get(language.lower(), language.title())
