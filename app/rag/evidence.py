"""Evidence strength: does the retrieved evidence actually support an answer?"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Any

from app.models import Chunk
from app.rag.tokenize import STOPWORDS, WORD_RE, code_tokens, query_identifiers, split_identifier, stems

# --- evidence strength (relevance floor) ------------------------------------------------------
# Retrieval always returns top_k chunks, so "something came back" says nothing about whether the
# repository actually contains the answer. The assessment below is deterministic and explainable:
# it checks whether the *content terms* of the question (not stopwords or intent vocabulary such
# as "how do I run") occur in the top-ranked evidence, whether an exact identifier/file the user
# named was found, and - when a cross-encoder is active - whether its calibrated relevance is high.
EVIDENCE_LEVELS = ("strong", "weak", "none")
DEFAULT_MIN_COVERAGE = 0.6  # share of content terms one top chunk must cover for "strong"
DEFAULT_RERANK_RELEVANCE = 0.5  # cross-encoder sigmoid relevance that counts as "strong" on its own
EVIDENCE_TOP_N = 3  # how many top-ranked chunks may provide the single-chunk coverage
MIN_PREFIX = 4  # shortest prefix accepted for prefix matching ("auth" ~ "authentication")
REPO_LEVEL_INTENTS = {
    "repo_overview", "run_instructions", "architecture", "file_explanation", "dependency_question",
    "weakness_question", "improvement_question", "security_question", "testing_question",
}
# Question scaffolding and intent vocabulary: these words route the question but say nothing about
# whether the repository contains the specific thing being asked about.
GENERIC_TERMS = {
    "handle", "handled", "handles", "handling", "define", "defined", "defines", "implement", "implemented",
    "implements", "implementation", "happen", "happens", "happened", "doing", "done", "did", "make", "makes",
    "made", "need", "needs", "want", "like", "look", "looks", "whose", "some", "any", "all", "here", "way",
    "ways", "thing", "things", "should", "would", "could", "will", "shall", "might", "must", "has", "have",
    "had", "was", "were", "been", "being", "than", "then", "also", "just", "only", "more", "most", "other",
    "onto", "over", "under", "via", "per", "each", "every", "not", "yes", "exist", "exists", "located",
    "location", "find", "found", "see", "give", "list", "describe", "example", "examples", "configure",
    "configured", "configures", "run", "running", "runs", "install", "installed", "installing", "installation",
    "setup", "set", "start", "started", "starting", "launch", "getting", "quickstart", "locally", "overview",
    "summary", "summarize", "summarise", "purpose", "architecture", "structure", "design", "diagram", "module",
    "modules", "component", "components", "layer", "layers", "flow", "flows", "weakness", "weaknesses", "risk",
    "risks", "issue", "issues", "problem", "problems", "improve", "improvement", "improvements", "suggest",
    "suggestion", "suggestions", "recommend", "recommendation", "recommendations", "better", "refactor", "call",
    "calls", "called", "caller", "callers", "trace", "important", "first", "read", "main", "entry", "entrypoint",
    "entrypoints", "point", "points", "new", "codebase", "app", "application", "system", "tool", "library",
    "does", "work", "working", "happening", "anything", "something", "everything", "there", "whether",
    "deploy", "deployed", "deploying", "deployment",
}
# Common code abbreviations so natural-language questions still match terse identifiers.
ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "authentication": ("auth", "authn"), "authenticate": ("auth",), "authorization": ("auth", "authz", "perm"),
    "configuration": ("config", "cfg", "conf"), "database": ("db",), "repository": ("repo",),
    "environment": ("env",), "message": ("msg",), "messages": ("msg", "msgs"), "password": ("pwd", "passwd"),
    "passwords": ("pwd", "passwd"), "kubernetes": ("k8s",), "initialize": ("init",), "initialization": ("init",),
    "administrator": ("admin",), "directory": ("dir",), "documentation": ("docs", "doc"),
    "dependency": ("dep", "deps"), "dependencies": ("dep", "deps"), "parameter": ("param", "params"),
    "parameters": ("param", "params"), "argument": ("arg", "args"), "arguments": ("arg", "args"),
    "asynchronous": ("async",), "synchronous": ("sync",), "temporary": ("tmp", "temp"),
    "specification": ("spec",), "information": ("info",), "development": ("dev",), "production": ("prod",),
    "utilities": ("utils", "util"), "utility": ("util", "utils"), "statistics": ("stats",),
    "request": ("req",), "requests": ("req",), "response": ("resp", "res"), "error": ("err",),
    "errors": ("err",), "image": ("img",), "number": ("num",), "library": ("lib",), "calculate": ("calc",),
    "generate": ("gen",), "context": ("ctx",), "connection": ("conn",), "database_url": ("db_url",),
}


@dataclass(slots=True)
class Vocabulary:
    """Token set with prefix lookups (sorted list + bisect), so matching stays cheap on big repos."""

    tokens: set[str]
    ordered: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ordered = sorted(self.tokens)

    def has(self, form: str) -> bool:
        if form in self.tokens:
            return True
        if len(form) < MIN_PREFIX:
            return False
        position = bisect.bisect_left(self.ordered, form)  # a token that extends the form: auth -> authentication
        if position < len(self.ordered) and self.ordered[position].startswith(form):
            return True
        floor = max(MIN_PREFIX + 1, int(0.6 * len(form)))  # a token the form extends: configured -> config
        return any(form[:size] in self.tokens for size in range(floor, len(form)))


@dataclass(frozen=True, slots=True)
class TermGroup:
    label: str
    forms: frozenset[str]
    parts: tuple[str, ...] = ()

    def matched_by(self, vocabulary: Vocabulary) -> bool:
        if any(vocabulary.has(form) for form in self.forms):
            return True
        return bool(self.parts) and all(vocabulary.has(part) for part in self.parts)


def content_term_groups(query: str) -> list[TermGroup]:
    """Distinct content words of a question, each with its stem/abbreviation variants."""
    groups: dict[str, TermGroup] = {}
    for match in WORD_RE.finditer(query):
        raw = match.group(0)
        lowered = raw.lower()
        if lowered in groups or lowered.isdigit() or len(lowered) < 3 or lowered in STOPWORDS or lowered in GENERIC_TERMS:
            continue
        forms = {lowered, *stems(lowered), *ABBREVIATIONS.get(lowered, ())}
        for stem in stems(lowered):
            forms.update(ABBREVIATIONS.get(stem, ()))
        parts = tuple(part for part in split_identifier(raw) if len(part) > 1 and part not in STOPWORDS)
        groups[lowered] = TermGroup(lowered, frozenset(forms), parts if len(parts) > 1 else ())
    return list(groups.values())


@dataclass(slots=True)
class EvidenceAssessment:
    """How strongly the retrieved evidence supports answering the question from this repository."""

    level: str  # "strong" | "weak" | "none"
    reason: str
    basis: str  # which rule decided: identifier_match, mentioned_file, term_coverage, rerank_relevance, repo_level, ...
    coverage: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    missing_terms: list[str] = field(default_factory=list)
    absent_from_repo: list[str] = field(default_factory=list)
    entity_found: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "reason": self.reason,
            "basis": self.basis,
            "coverage": round(self.coverage, 3),
            "matched_terms": self.matched_terms,
            "missing_terms": self.missing_terms,
            "absent_from_repo": self.absent_from_repo,
            "answer_support": "unassessed",
            "assessment_scope": "retrieval relevance; not a claim-support probability",
            "entity_found": self.entity_found,
        }


def _quoted(terms: list[str], limit: int = 6) -> str:
    return ", ".join(f"'{term}'" for term in terms[:limit]) + (" ..." if len(terms) > limit else "")


def assess_evidence(
    query: str,
    intents: list[str],
    selected: list[Chunk],
    provenance: dict[str, dict[str, int]],
    identifiers: set[str] | None = None,
    absent_from_repo: list[str] | None = None,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    rerank_relevance_threshold: float = DEFAULT_RERANK_RELEVANCE,
) -> EvidenceAssessment:
    """Decide whether the ranked evidence actually supports an answer (rules applied in order).

    1. Nothing retrieved -> none.
    2. No content terms (pure intent question such as "how do I run this?") -> strong for
       repo-level intents answered from analysis facts, weak otherwise.
    3. An exact identifier from the question is in a top chunk, or a named file exists -> strong.
    4. One top-ranked chunk covers >= min_coverage of the content terms -> strong.
    5. A cross-encoder (when active) rates the top chunk >= its relevance threshold -> strong.
    6. Some content terms appear in the retrieved evidence -> weak; none of them -> none.
    """
    identifiers = identifiers if identifiers is not None else query_identifiers(query)
    groups = content_term_groups(query)
    absent = absent_from_repo or []
    if not selected:
        return EvidenceAssessment("none", "No source excerpts were retrieved for this question.", "empty", absent_from_repo=absent)
    if not groups:
        if any(intent in REPO_LEVEL_INTENTS for intent in intents):
            return EvidenceAssessment("strong", "Repository-level question answered from repository analysis facts and top-ranked files.", "repo_level")
        return EvidenceAssessment("weak", "The question has no specific terms to check against the repository.", "no_content_terms")

    top = selected[:EVIDENCE_TOP_N]
    top_vocabs = [Vocabulary(set(code_tokens(chunk.index_text))) for chunk in top]
    matched_per_chunk = [[group.label for group in groups if group.matched_by(vocab)] for vocab in top_vocabs]
    union_vocab = Vocabulary(set().union(*(set(code_tokens(chunk.index_text)) for chunk in selected)))
    matched = [group.label for group in groups if group.matched_by(union_vocab)]
    missing = [group.label for group in groups if group.label not in matched]
    best = max(matched_per_chunk, key=len)
    coverage = len(best) / len(groups)

    entity_found = False

    def result(level: str, reason: str, basis: str) -> EvidenceAssessment:
        return EvidenceAssessment(level, reason, basis, coverage, matched, missing, absent, entity_found)

    lowered_ids = {identifier.lower() for identifier in identifiers}
    for chunk, vocab in zip(top, top_vocabs, strict=True):
        symbol = (chunk.symbol or "").split("#")[0].lower()
        hits = sorted(i for i in lowered_ids if i in vocab.tokens or i == symbol or i == symbol.rsplit(".", 1)[-1])
        entity_found = entity_found or bool(hits)
        if hits and coverage >= min_coverage and not absent:
            return result("strong", f"Exact identifier `{hits[0]}` found in {chunk.file_path}:{chunk.start_line}-{chunk.end_line}.", "identifier_match")
    mentioned = next((chunk for chunk in selected if "mentioned" in provenance.get(chunk.id, {})), None)
    entity_found = entity_found or mentioned is not None
    if mentioned is not None and coverage >= min_coverage and not absent:
        return result("strong", f"The question names {mentioned.file_path}, which exists in the repository.", "mentioned_file")
    if best and coverage >= min_coverage:
        chunk = top[matched_per_chunk.index(best)]
        return result("strong", f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line} matches {_quoted(best)} ({len(best)}/{len(groups)} key terms).", "term_coverage")
    relevance = (top[0].metadata or {}).get("rerank_relevance")
    if isinstance(relevance, (int, float)) and relevance >= rerank_relevance_threshold:
        return result("strong", f"The reranker rates {top[0].file_path} as relevant ({relevance:.2f}).", "rerank_relevance")
    if matched:
        gap = f"; not found anywhere in the repository: {_quoted(absent)}" if absent else f"; not found in the top matches: {_quoted(missing)}"
        return result("weak", f"Only partial matches: {_quoted(matched)} ({len(best)}/{len(groups)} key terms in the best match){gap}.", "partial_terms")
    if absent and len(absent) == len(groups):
        return result("none", f"None of the key terms ({_quoted(absent)}) appear anywhere in the indexed repository.", "terms_absent")
    return result("none", f"None of the key terms ({_quoted(missing)}) appear in the retrieved evidence.", "no_term_overlap")
