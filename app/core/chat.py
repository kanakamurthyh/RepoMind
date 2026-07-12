from __future__ import annotations

import logging
from typing import Any

from app.config import Settings, settings
from app.core.repo_analyzer import RepoAnalyzer
from app.rag.llm import GeminiClient, GroundedPromptBuilder, LLMError
from app.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, analyzer: RepoAnalyzer, app_settings: Settings = settings) -> None:
        self.analyzer = analyzer
        self.settings = app_settings
        self.retriever = HybridRetriever(analyzer.vector_store)
        self.prompt_builder = GroundedPromptBuilder()

    def answer(self, repo_id: str, question: str) -> dict[str, Any]:
        analysis = self.analyzer.load(repo_id)
        facts = analysis.summary.get("repo_facts", {})
        weaknesses = [weakness.to_dict() for weakness in analysis.weaknesses]
        context = self.retriever.retrieve(repo_id, question, analysis.graph, facts, weaknesses)
        if not self.settings.local_model_mode and not (self.settings.openai_api_key or self.settings.gemini_api_key):
            return {
                "answer": "No external LLM key is configured and LOCAL_MODEL_MODE is disabled. Enable local mode or set OPENAI_API_KEY/GEMINI_API_KEY.",
                "citations": self._public_citations(context, facts),
            }
        fallback_answer = self._grounded_answer(question, analysis.summary, context)
        answer, llm_status = self._llm_answer(question, analysis.summary, context, fallback_answer)
        citations = self._public_citations(context, facts)
        try:
            self.analyzer.metadata.record_chat(repo_id, question, answer, citations)
        except Exception as exc:
            logger.warning("Could not persist chat response for %s: %s", repo_id, exc)
        return {"answer": answer, "citations": citations, "llm": llm_status}

    def _llm_answer(self, question: str, summary: dict[str, Any], context: dict[str, Any], fallback_answer: str) -> tuple[str, dict[str, Any]]:
        provider = self._effective_provider()
        if provider != "gemini":
            return fallback_answer, {"provider": "local", "used": False, "reason": "No Gemini key/provider configured."}
        if not self.settings.gemini_api_key:
            return (
                fallback_answer + "\n\nNotes / uncertainty:\n- Gemini is selected, but `GEMINI_API_KEY` is not configured, so RepoMind used the local grounded fallback.",
                {"provider": "gemini", "used": False, "reason": "GEMINI_API_KEY is not configured."},
            )
        prompt = self.prompt_builder.build(question, summary, context, fallback_answer)
        try:
            client = GeminiClient(self.settings.gemini_api_key, self.settings.gemini_model)
            answer = client.generate(prompt)
            return answer, {
                "provider": "gemini",
                "model": getattr(client, "last_model_used", None) or self.settings.gemini_model,
                "used": True,
            }
        except LLMError as exc:
            return (
                fallback_answer + f"\n\nNotes / uncertainty:\n- Gemini answer synthesis failed, so RepoMind used the local grounded fallback. Error: {exc}",
                {"provider": "gemini", "model": self.settings.gemini_model, "used": False, "reason": str(exc)},
            )

    def _effective_provider(self) -> str:
        provider = self.settings.llm_provider.strip().lower()
        if provider in {"off", "disabled", "none"}:
            return "local"
        if self.settings.gemini_api_key and provider in {"auto", "local", "gemini", ""}:
            return "gemini"
        return provider

    def _grounded_answer(self, question: str, summary: dict[str, Any], context: dict[str, Any]) -> str:
        intents = context.get("intents", [])
        facts = context.get("repo_facts", {})
        chunks = context.get("selected_chunks", [])
        weaknesses = context.get("weakness_facts", [])
        if "run_instructions" in intents:
            return self._run_answer(question, facts, chunks)
        if any(intent in intents for intent in ("weakness_question", "improvement_question", "security_question")):
            return self._weakness_answer(weaknesses, summary, context)
        if "file_explanation" in intents:
            return self._files_answer(facts, chunks, context)
        if "architecture" in intents:
            return self._architecture_answer(summary, facts, context)
        return self._overview_answer(summary, facts, chunks, context)

    def _overview_answer(self, summary: dict[str, Any], facts: dict[str, Any], chunks: list[dict], context: dict[str, Any]) -> str:
        identity = summary.get("project_identity", {})
        lines = [
            "Direct answer:",
            f"{identity.get('one_line_summary', summary.get('one_line_summary', 'This repository contains a software project.'))} {identity.get('what_this_repo_is_for', '')}",
            "",
            "Explanation:",
        ]
        for step in summary.get("how_it_likely_works", [])[:5]:
            lines.append(f"- {step}")
        lines.extend(["", "Important files:"])
        for item in facts.get("important_files", [])[:6]:
            lines.append(f"- `{item['path']}`: {item['reason']}")
        lines.extend(self._evidence_section(context))
        return "\n".join(lines)

    def _run_answer(self, question: str, facts: dict[str, Any], chunks: list[dict]) -> str:
        instructions = self._filter_run_instructions_for_question(question, facts.get("run_instructions", []))
        official_docs = facts.get("run_doc_paths") or facts.get("install_doc_paths") or facts.get("official_doc_paths", [])
        official = [item for item in instructions if item.get("official")]
        clear = [item for item in instructions if item.get("command") != "Not clearly documented."]
        lines = ["Direct answer:"]
        if official:
            lines.append(f"Use the official repository documentation first: {', '.join(f'`{path}`' for path in official_docs[:3])}. The commands below come from those docs.")
        elif clear:
            lines.append("I did not find official install documentation, so these commands are inferred from package/config files and should be verified.")
        else:
            lines.append("The repository does not clearly document a reliable run command.")

        lines.extend(["", "Steps:"])
        current_heading = None
        for item in instructions[:8]:
            heading = item.get("heading") or "Instructions"
            if heading != current_heading:
                lines.append(f"- {heading}:")
                current_heading = heading
            source = item.get("source", "unknown")
            line_range = ""
            if item.get("start_line") and item.get("end_line"):
                line_range = f":{item['start_line']}-{item['end_line']}"
            lines.append(f"  - `{item['command']}` from `{source}{line_range}` ({item['confidence']} confidence).")
        if not clear:
            lines.append("- Suggested fix: add a README quickstart with install, environment, and run commands.")

        lines.extend(["", "Commands:"])
        if clear:
            lines.append("```bash")
            lines.extend(item["command"] for item in instructions[:8] if item.get("command") != "Not clearly documented.")
            lines.append("```")
        else:
            lines.append("- No official command block was found.")

        lines.extend(["", "Official documentation:"])
        for path in official_docs[:6]:
            lines.append(f"- `{path}`")

        helper_commands = facts.get("official_helper_commands", [])
        if helper_commands:
            lines.extend(["", "Helper / developer scripts:"])
            for command in helper_commands[:6]:
                source = command.get("source", "unknown")
                line_range = f":{command['start_line']}-{command['end_line']}" if command.get("start_line") and command.get("end_line") else ""
                lines.append(f"- `{command['command']}` from `{source}{line_range}`: optional/helper command from official docs, not the main install command.")
        lines.extend(self._evidence_section({"source_references": self._run_refs_from_chunks(chunks, facts)}))
        return "\n".join(lines)

    def _filter_run_instructions_for_question(self, question: str, instructions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        lowered = question.lower()
        wants_uninstall = "uninstall" in lowered or "remove" in lowered
        wants_benchmark = "benchmark" in lowered or "smoke" in lowered
        filtered = []
        for item in instructions:
            heading = str(item.get("heading", "")).lower()
            command = str(item.get("command", "")).lower()
            if not wants_uninstall and "uninstall" in heading:
                continue
            if not wants_benchmark and any(marker in heading or marker in command for marker in ("benchmark", "smoke_test", "measure_")):
                continue
            filtered.append(item)
        return filtered or instructions

    def _weakness_answer(self, weaknesses: list[dict], summary: dict[str, Any], context: dict[str, Any]) -> str:
        lines = ["Direct answer:"]
        if weaknesses:
            lines.append("The main risks are concrete project hygiene, run-readiness, testing, or security findings detected from repository evidence.")
        else:
            lines.append("I did not find high-confidence weaknesses in the retrieved evidence, though this is static analysis rather than a full audit.")
        lines.extend(["", "Explanation:"])
        for weakness in weaknesses[:8]:
            location = weakness.get("file_path") or "repository-wide"
            if weakness.get("line_number"):
                location = f"{location}:{weakness['line_number']}"
            evidence = f" Evidence: `{weakness['evidence']}`" if weakness.get("evidence") else ""
            lines.append(f"- {weakness['severity'].upper()} {weakness['category']} - {weakness['title']} ({location}). {weakness['suggested_fix']}{evidence}")
        if not weaknesses:
            for item in summary.get("suggested_improvements", {}).get("top_5", [])[:5]:
                lines.append(f"- {item}")
        lines.extend(["", "Important files:"])
        for ref in context.get("source_references", [])[:6]:
            lines.append(f"- `{self._format_ref(ref)}`: supporting evidence.")
        return "\n".join(lines)

    def _files_answer(self, facts: dict[str, Any], chunks: list[dict], context: dict[str, Any]) -> str:
        lines = [
            "Direct answer:",
            "The most important files are the entrypoints, package/config files, README, and modules with many graph relationships.",
            "",
            "Explanation:",
        ]
        for item in facts.get("important_files", [])[:10]:
            lines.append(f"- `{item['path']}`: {item['reason']}")
        lines.extend(self._evidence_section(context))
        return "\n".join(lines)

    def _architecture_answer(self, summary: dict[str, Any], facts: dict[str, Any], context: dict[str, Any]) -> str:
        arch = summary.get("architecture", {})
        lines = [
            "Direct answer:",
            arch.get("explanation", "The architecture is inferred from folders, entrypoints, imports, and code definitions."),
            "",
            "Explanation:",
        ]
        for step in arch.get("entrypoint_flow", [])[:8]:
            lines.append(f"- {step}")
        lines.extend(["", "Important files:"])
        for item in facts.get("main_entrypoints", [])[:5]:
            lines.append(f"- `{item['path']}`: {item['reason']}")
        for item in facts.get("important_modules", [])[:5]:
            lines.append(f"- `{item['path']}`: {item['reason']}")
        lines.extend(self._evidence_section(context))
        return "\n".join(lines)

    def _evidence_section(self, context: dict[str, Any]) -> list[str]:
        refs = context.get("source_references", [])
        lines = ["", "Evidence:"]
        if refs:
            for ref in refs[:8]:
                lines.append(f"- `{self._format_ref(ref)}`")
        else:
            lines.append("- No precise source chunk was retrieved; this answer relies on repository-level facts.")
        lines.extend(["", "Notes / uncertainty:"])
        lines.append("- This is static repository analysis. If run commands are not documented, RepoMind reports inferred commands as uncertain.")
        return lines

    def _refs_from_chunks(self, chunks: list[dict]) -> list[dict[str, Any]]:
        return [{"file_path": chunk["file_path"], "start_line": chunk["start_line"], "end_line": chunk["end_line"]} for chunk in chunks]

    def _run_refs_from_chunks(self, chunks: list[dict], facts: dict[str, Any]) -> list[dict[str, Any]]:
        preferred = set(facts.get("run_doc_paths", [])) | set(facts.get("install_doc_paths", [])) | set(facts.get("readme_paths", [])) | set(facts.get("dependency_files", []))
        refs = [chunk for chunk in chunks if chunk.get("file_path") in preferred]
        return self._refs_from_chunks(refs or chunks[:4])

    def _format_ref(self, ref: dict[str, Any]) -> str:
        path = ref.get("file_path", "")
        start = ref.get("start_line")
        end = ref.get("end_line")
        if start and end:
            return f"{path}:{start}-{end}"
        return path

    def _public_citations(self, context: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
        if "run_instructions" not in context.get("intents", []):
            return context["source_references"]
        preferred = set(facts.get("run_doc_paths", [])) | set(facts.get("install_doc_paths", [])) | set(facts.get("readme_paths", [])) | set(facts.get("dependency_files", []))
        filtered = [ref for ref in context["source_references"] if ref.get("file_path") in preferred]
        return filtered or context["source_references"][:4]
