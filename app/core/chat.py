from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from app.config import Settings, settings
from app.core.snapshots import snapshot_read
from app.rag.grounding import GroundedPromptBuilder, annotate_invalid, verify_citations
from app.rag.llm import LLMError, build_llm
from app.rag.rerank import build_reranker
from app.rag.retriever import HybridRetriever
from app.utils.redaction import redact, redact_data

logger = logging.getLogger(__name__)

# Intents whose deterministic answer comes from repository-wide analysis (docs, static findings),
# so weak term evidence adds a caveat instead of replacing the answer.
ANALYSIS_INTENTS = ("run_instructions", "weakness_question", "improvement_question", "security_question")
UNKNOWN_EVIDENCE: dict[str, Any] = {"level": "unknown", "reason": "Evidence strength was not assessed.", "basis": "unassessed"}


class ChatService:
    def __init__(
        self,
        analyzer: Any,
        app_settings: Settings = settings,
        llm_factory: Callable[[Settings], tuple[Any, str]] = build_llm,
        cache_size: int = 256,
    ) -> None:
        self.analyzer = analyzer
        self.settings = app_settings
        self.retriever = HybridRetriever(
            analyzer.vector_store,
            build_reranker(app_settings.reranker_provider, app_settings.reranker_model),
            min_coverage=app_settings.evidence_min_coverage,
            rerank_relevance_threshold=app_settings.evidence_rerank_threshold,
        )
        self.prompt_builder = GroundedPromptBuilder()
        self.llm_factory = llm_factory
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self.cache_size = cache_size

    # --- public ---------------------------------------------------------------------------
    @snapshot_read
    def retrieve(self, repo_id: str, question: str, history: list[dict[str, str]] | None = None, top_k: int = 8) -> dict[str, Any]:
        analysis = self.analyzer.load(repo_id)
        facts = analysis.summary.get("repo_facts", {})
        weaknesses = [weakness.to_dict() for weakness in analysis.weaknesses]
        graph_index = self.analyzer.graph_index(repo_id) if hasattr(self.analyzer, "graph_index") else None
        context = self.retriever.retrieve(repo_id, question, analysis.graph, facts, weaknesses, top_k=top_k, history=history, graph_index=graph_index)
        context["_summary"] = analysis.summary
        context["_commit"] = analysis.summary.get("ingestion", {}).get("run_id") or analysis.commit_sha
        context["snapshot"] = {"commit_sha": analysis.commit_sha, "run_id": analysis.summary.get("ingestion", {}).get("run_id")}
        for ref in context["source_references"]:
            ref.update(context["snapshot"])
            if analysis.commit_sha and analysis.summary.get("ingestion", {}).get("source_kind", "github") == "github":
                ref["source_url"] = f"{analysis.repo_url}/blob/{analysis.commit_sha}/{quote(ref['file_path'], safe='/')}#L{ref['start_line']}-L{ref['end_line']}"
        context["coverage"] = analysis.summary.get("coverage", {})
        return redact_data(context)

    @snapshot_read
    def answer(self, repo_id: str, question: str, history: list[dict[str, str]] | None = None, debug: bool = False) -> dict[str, Any]:
        history = [turn for turn in (history or []) if turn.get("role") in {"user", "assistant"}][-6:]
        analysis = self.analyzer.load(repo_id)
        run_key = analysis.summary.get("ingestion", {}).get("run_id") or analysis.commit_sha
        llm, unavailable_reason = self.llm_factory(self.settings)
        provider_key = f"{getattr(llm, 'name', 'local')}:{getattr(llm, 'model', '')}"
        cache_key = self._cache_key(repo_id, run_key, question, history, provider_key)
        cached = self._cache_get(cache_key)
        if cached is not None and not debug:
            return {**cached, "cached": True}
        context = self.retrieve(repo_id, question, history)
        summary = context.pop("_summary")
        context.pop("_commit")
        facts = context.get("repo_facts", {})
        citations = self._public_citations(context, facts)
        evidence = dict(context.get("evidence") or UNKNOWN_EVIDENCE)

        if not self.settings.local_model_mode and not (self.settings.openai_api_key or self.settings.gemini_api_key):
            return {
                "answer": "No external LLM key is configured and LOCAL_MODEL_MODE is disabled. Enable local mode or set GEMINI_API_KEY/OPENAI_API_KEY.",
                "citations": citations,
                "llm": {"provider": "none", "used": False, "reason": "not_configured"},
                "evidence_strength": evidence["level"],
                "evidence": evidence,
            }

        fallback_answer = self._grounded_answer(question, summary, context)
        prompt = self.prompt_builder.build(question, summary, context, history)
        evidence_note = self._evidence_instruction(evidence)
        if evidence_note:
            prompt.user = f"{prompt.user}\n\n{evidence_note}"
        if llm is None:
            answer_text = fallback_answer
            llm_status: dict[str, Any] = {"provider": "local", "used": False, "reason": unavailable_reason}
        else:
            try:
                response = llm.complete(prompt.user, system=prompt.system)
                answer_text = response.text
                llm_status = {"provider": response.provider, "model": response.model, "used": True, "latency_ms": response.latency_ms, "usage": response.usage}
            except LLMError as exc:
                logger.warning("LLM synthesis failed for %s (%s): %s", repo_id, exc.code, exc.detail)
                answer_text = fallback_answer + f"\n\nNotes / uncertainty:\n- The LLM was unavailable ({exc}). This is RepoMind's deterministic, evidence-only answer."
                llm_status = {"provider": getattr(llm, "name", "llm"), "model": getattr(llm, "model", None), "used": False, "reason": str(exc), "error_code": exc.code}

        answer_text = redact(answer_text)
        report = verify_citations(answer_text, prompt.source_ids)
        if llm_status.get("used"):
            answer_text = annotate_invalid(answer_text, report)
            if not report.valid_ids and not report.path_refs_verified:
                answer_text += "\n\n> ⚠ This answer does not cite any retrieved source. Treat it as unverified."
        cited = set(report.valid_ids)
        for citation in citations:
            citation["cited"] = citation.get("id") in cited
        result: dict[str, Any] = {
            "answer": answer_text,
            "citations": citations,
            "llm": llm_status,
            "grounding": report.to_dict(),
            "intents": context.get("intents", []),
            "evidence_strength": evidence["level"],
            "evidence": evidence,
            "snapshot": context.get("snapshot"),
            "coverage": context.get("coverage"),
        }
        if debug:
            result["debug"] = {"search_query": context.get("search_query"), "retrieval_trace": context.get("retrieval_trace"), "graph_paths": context.get("graph_paths")}
        try:
            self.analyzer.metadata.record_chat(repo_id, redact(question), answer_text, citations, {"llm": llm_status, "grounding": report.grounding_score, "evidence_strength": evidence["level"]})
        except Exception as exc:  # chat history is best-effort
            logger.warning("Could not persist chat response for %s: %s", repo_id, exc)
        if llm is None or llm_status.get("used"):
            self._cache_put(cache_key, {k: v for k, v in result.items() if k != "debug"})
        return result

    # --- cache --------------------------------------------------------------------------------
    def _cache_key(self, repo_id: str, commit: str | None, question: str, history: list[dict[str, str]], provider: str) -> str:
        config = [self.settings.reranker_provider, self.settings.reranker_model, self.settings.evidence_min_coverage,
                  self.settings.evidence_rerank_threshold, self.prompt_builder.max_context_chars]
        payload = json.dumps([repo_id, commit, question, history, provider, config], sort_keys=True)
        return hashlib.sha1(payload.encode()).hexdigest()

    def _cache_get(self, key: str) -> dict[str, Any] | None:
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return json.loads(json.dumps(self._cache[key]))
        return None

    def _cache_put(self, key: str, value: dict[str, Any]) -> None:
        with self._cache_lock:
            self._cache[key] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # --- evidence strength / abstention ------------------------------------------------------------
    @staticmethod
    def _evidence_instruction(evidence: dict[str, Any]) -> str:
        """Extra prompt block telling the LLM how much the retrieved evidence can be trusted."""
        level = evidence.get("level", "strong")
        absent = evidence.get("absent_from_repo") or []
        absent_line = f"Question terms that occur nowhere in the indexed repository: {', '.join(absent)}." if absent else ""
        if level == "strong":
            if not absent_line:
                return ""
            return f"<evidence_check level=\"strong\">\n{absent_line}\nIf the answer depends on those terms, say the repository does not mention them instead of guessing.\n</evidence_check>"
        strength = "NO" if level == "none" else "only WEAK"
        return "\n".join(
            [
                f'<evidence_check level="{level}">',
                f"RepoMind's retrieval check found {strength} evidence in this repository for the question. {evidence.get('reason', '')}",
                *([absent_line] if absent_line else []),
                "The sources are only the closest matches and may be unrelated. Do not guess or fill gaps from general knowledge.",
                "Begin the direct answer by stating that the repository does not appear to contain (or only partially covers) what was asked,",
                "then say what is missing, and list any genuinely related sources as \"possibly related\" with citations.",
                "</evidence_check>",
            ]
        )

    def _abstain_answer(self, context: dict[str, Any], evidence: dict[str, Any]) -> str:
        level = evidence.get("level")
        lines = ["Direct answer:"]
        if level == "none":
            lines.append("The repository does not appear to contain evidence for this question, so RepoMind will not guess an answer.")
        else:
            lines.append("RepoMind found only weak evidence for this question in the repository, so it will not present a confident answer.")
        lines.append(f"Evidence check: {evidence.get('reason', 'no strong match')}")
        refs = context.get("source_references", [])
        if refs:
            lines.extend(["", "Possibly related (closest matches, not a confirmed answer):"])
            lines.extend(self._ref_lines(refs[:5]))
        notes = [f"Evidence strength: {level}."]
        absent = evidence.get("absent_from_repo") or []
        if absent:
            notes.append(f"Not found anywhere in the indexed repository: {', '.join(absent)}.")
        notes.append("It may live in another repository, an unindexed or generated file, or under a different name; try asking with an exact symbol or file name.")
        return "\n".join(lines + self._uncertainty(notes))

    @staticmethod
    def _evidence_caveat(evidence: dict[str, Any]) -> list[str]:
        level = evidence.get("level")
        prefix = "No direct evidence" if level == "none" else "Only weak evidence"
        return [f"> Evidence check ({level}): {prefix} for the specific question was found. {evidence.get('reason', '')}", "> The answer below is the closest repository-level information, not a confirmed answer.", ""]

    @staticmethod
    def _absent_notes(context: dict[str, Any]) -> list[str]:
        absent = (context.get("evidence") or {}).get("absent_from_repo") or []
        return [f"These question terms do not appear anywhere in the repository: {', '.join(absent)}."] if absent else []

    # --- deterministic answers ------------------------------------------------------------------
    def _grounded_answer(self, question: str, summary: dict[str, Any], context: dict[str, Any]) -> str:
        intents = context.get("intents", [])
        evidence = context.get("evidence") or UNKNOWN_EVIDENCE
        if evidence.get("level") in {"none", "weak", "unknown"}:
            if not any(intent in intents for intent in ANALYSIS_INTENTS):
                return self._abstain_answer(context, evidence)
            return "\n".join(self._evidence_caveat(evidence)) + "\n" + self._routed_answer(question, summary, context)
        return self._routed_answer(question, summary, context)

    def _routed_answer(self, question: str, summary: dict[str, Any], context: dict[str, Any]) -> str:
        intents = context.get("intents", [])
        facts = context.get("repo_facts", {})
        chunks = context.get("selected_chunks", [])
        if "run_instructions" in intents:
            return self._run_answer(question, facts, chunks, context)
        if any(intent in intents for intent in ("weakness_question", "improvement_question", "security_question")):
            return self._weakness_answer(context.get("weakness_facts", []), summary, context)
        if any(intent in intents for intent in ("function_explanation", "call_flow", "general_code_question")) and chunks:
            return self._code_answer(context)
        if "file_explanation" in intents:
            return self._files_answer(facts, context)
        if "architecture" in intents:
            return self._architecture_answer(summary, facts, context)
        return self._overview_answer(summary, facts, context)

    def _code_answer(self, context: dict[str, Any]) -> str:
        refs = context.get("source_references", [])
        lines = ["Direct answer:"]
        top = refs[0] if refs else None
        if top:
            symbol = f" (`{top['symbol']}`)" if top.get("symbol") else ""
            lines.append(f"The most relevant code is in `{top['file_path']}`{symbol}, lines {top['start_line']}-{top['end_line']} [{top['id']}].")
        lines.extend(["", "Explanation:"])
        lines.extend(self._ref_lines(refs[:5]))
        if context.get("graph_paths"):
            lines.extend(["", "Code relationships:"])
            lines.extend(f"- {path}" for path in context["graph_paths"][:6])
        lines.extend(self._uncertainty(["Deterministic mode lists the most relevant code; enable an LLM provider for a synthesized explanation.", *self._absent_notes(context)]))
        return "\n".join(lines)

    @staticmethod
    def _ref_lines(refs: list[dict[str, Any]]) -> list[str]:
        lines = []
        for ref in refs:
            symbol = f"`{ref['symbol']}` in " if ref.get("symbol") else ""
            first_line = next((line.strip() for line in ref.get("excerpt", "").splitlines() if line.strip()), "")
            lines.append(f"- {symbol}`{ref['file_path']}:{ref['start_line']}-{ref['end_line']}` [{ref['id']}]: `{first_line[:100]}`")
        return lines

    def _overview_answer(self, summary: dict[str, Any], facts: dict[str, Any], context: dict[str, Any]) -> str:
        identity = summary.get("project_identity", {})
        lines = [
            "Direct answer:",
            f"{identity.get('one_line_summary', 'This repository contains a software project.')} {identity.get('what_this_repo_is_for', '')}".strip(),
            "",
            "Explanation:",
        ]
        lines.extend(f"- {step}" for step in summary.get("how_it_likely_works", [])[:5])
        lines.extend(["", "Important files:"])
        lines.extend(f"- `{item['path']}`: {item['reason']}" for item in facts.get("important_files", [])[:6])
        lines.extend(self._evidence(context))
        return "\n".join(lines)

    def _run_answer(self, question: str, facts: dict[str, Any], chunks: list[dict], context: dict[str, Any]) -> str:
        instructions = self._filter_run_instructions_for_question(question, facts.get("run_instructions", []))
        official_docs = facts.get("run_doc_paths") or facts.get("install_doc_paths") or facts.get("official_doc_paths", [])
        official = [item for item in instructions if item.get("official")]
        clear = [item for item in instructions if item.get("command") != "Not clearly documented."]
        lines = ["Direct answer:"]
        if official:
            lines.append(f"Follow the official documentation first: {', '.join(f'`{path}`' for path in official_docs[:3])}. The commands below come from those docs.")
        elif clear:
            lines.append("No official install documentation was found, so these commands are inferred from package/config files and should be verified.")
        else:
            lines.append("The repository does not clearly document a reliable run command.")
        lines.extend(["", "Steps:"])
        current_heading = None
        for item in instructions[:10]:
            heading = item.get("heading") or "Instructions"
            if heading != current_heading:
                lines.append(f"- {heading}:")
                current_heading = heading
            line_range = f":{item['start_line']}-{item['end_line']}" if item.get("start_line") and item.get("end_line") else ""
            lines.append(f"  - `{item['command']}` from `{item.get('source', 'unknown')}{line_range}` ({item.get('confidence', 'medium')} confidence).")
        if clear:
            lines.extend(["", "Commands:", "```bash", *[item["command"] for item in instructions[:10] if item.get("command") != "Not clearly documented."], "```"])
        else:
            lines.append("- Suggested fix: add a README quickstart with install, environment, and run commands.")
        helper_commands = facts.get("official_helper_commands", [])
        if helper_commands:
            lines.extend(["", "Helper / developer scripts (optional):"])
            lines.extend(f"- `{command['command']}` from `{command.get('source', 'unknown')}`" for command in helper_commands[:6])
        lines.extend(self._evidence(context))
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
            if not wants_benchmark and any(marker in heading or marker in command for marker in ("benchmark", "smoke_test")):
                continue
            filtered.append(item)
        return filtered or instructions

    def _weakness_answer(self, weaknesses: list[dict], summary: dict[str, Any], context: dict[str, Any]) -> str:
        lines = ["Direct answer:"]
        if weaknesses:
            lines.append("The main risks below come from static analysis, docs-vs-code checks and dependency scanning of this repository.")
        else:
            lines.append("No high-confidence weaknesses were found, though this is static analysis rather than a full audit.")
        lines.extend(["", "Explanation:"])
        for weakness in weaknesses[:8]:
            location = weakness.get("file_path") or "repository-wide"
            if weakness.get("line_number"):
                location = f"{location}:{weakness['line_number']}"
            evidence = f" Evidence: `{weakness['evidence']}`" if weakness.get("evidence") else ""
            lines.append(f"- **{weakness['severity'].upper()}** {weakness['category']} - {weakness['title']} (`{location}`). {weakness['suggested_fix']}{evidence}")
        if not weaknesses:
            lines.extend(f"- {item}" for item in summary.get("suggested_improvements", {}).get("top_5", [])[:5])
        lines.extend(self._evidence(context))
        return "\n".join(lines)

    def _files_answer(self, facts: dict[str, Any], context: dict[str, Any]) -> str:
        lines = ["Direct answer:", "Start with the entrypoints, the most central modules, and the package/config files.", "", "Explanation:"]
        lines.extend(f"- `{item['path']}`: {item['reason']}" for item in facts.get("important_files", [])[:10])
        lines.extend(self._evidence(context))
        return "\n".join(lines)

    def _architecture_answer(self, summary: dict[str, Any], facts: dict[str, Any], context: dict[str, Any]) -> str:
        arch = summary.get("architecture", {})
        lines = ["Direct answer:", arch.get("explanation", "The architecture is inferred from folders, entrypoints, imports, and code definitions."), "", "Explanation:"]
        lines.extend(f"- {step}" for step in arch.get("entrypoint_flow", [])[:8])
        central = arch.get("central_files", [])
        if central:
            lines.extend(["", "Most central modules (dependency graph):"])
            lines.extend(f"- `{item['path']}` (used by {item['in_degree']} file(s))" for item in central[:6])
        lines.extend(self._evidence(context))
        return "\n".join(lines)

    def _evidence(self, context: dict[str, Any]) -> list[str]:
        refs = context.get("source_references", [])
        lines = ["", "Evidence:"]
        if refs:
            lines.extend(f"- [{ref['id']}] `{self._format_ref(ref)}`" for ref in refs[:8])
        else:
            lines.append("- No precise source chunk was retrieved; this answer relies on repository-level facts.")
        return lines + self._uncertainty(["This is static repository analysis; inferred commands and flows are marked as such.", *self._absent_notes(context)])

    def _uncertainty(self, notes: list[str]) -> list[str]:
        return ["", "Notes / uncertainty:", *[f"- {note}" for note in notes]]

    def _format_ref(self, ref: dict[str, Any]) -> str:
        path = ref.get("file_path", "")
        start, end = ref.get("start_line"), ref.get("end_line")
        return f"{path}:{start}-{end}" if start and end else path

    def _public_citations(self, context: dict[str, Any], facts: dict[str, Any]) -> list[dict[str, Any]]:
        refs = [dict(ref) for ref in context.get("source_references", [])]
        if "run_instructions" not in context.get("intents", []):
            return refs
        preferred = set(facts.get("run_doc_paths", [])) | set(facts.get("install_doc_paths", [])) | set(facts.get("readme_paths", [])) | set(facts.get("dependency_files", []))
        filtered = [ref for ref in refs if ref.get("file_path") in preferred]
        return filtered or refs[:4]
