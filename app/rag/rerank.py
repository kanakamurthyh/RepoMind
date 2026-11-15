"""Second-stage reranking: optional cross-encoder with a deterministic local fallback."""

from __future__ import annotations

import logging
import math

from app.models import Chunk
from app.rag.model_identity import model_revision
from app.rag.tokenize import code_tokens, query_identifiers

logger = logging.getLogger(__name__)

ROLE_PRIORS = {
    "run_instructions": {"docs": 0.25, "config": 0.15, "code": -0.05, "test": -0.2},
    "repo_overview": {"docs": 0.2, "config": 0.05},
    "testing_question": {"test": 0.3},
    "function_explanation": {"code": 0.15, "test": -0.05},
    "general_code_question": {"code": 0.1},
    "architecture": {"code": 0.1, "docs": 0.05},
    "dependency_question": {"config": 0.2},
}


class LocalReranker:
    """Feature-based reranker: query-term coverage, identifier/symbol hits, path hits, role priors."""

    name = "local-features"

    def rerank(self, query: str, chunks: list[Chunk], intents: list[str]) -> list[Chunk]:
        terms = set(code_tokens(query))
        identifiers = {identifier.lower() for identifier in query_identifiers(query)}
        if not chunks:
            return []
        max_fused = max((chunk.score or 0.0) for chunk in chunks) or 1.0
        rescored = []
        for chunk in chunks:
            text_tokens = set(code_tokens(chunk.index_text))
            coverage = len(terms & text_tokens) / len(terms) if terms else 0.0
            symbol = (chunk.symbol or "").lower()
            symbol_parts = set(code_tokens(chunk.symbol or ""))
            identifier_hit = 1.0 if identifiers and (symbol.split("#")[0] in identifiers or symbol.split(".")[-1].split("#")[0] in identifiers) else 0.0
            symbol_hit = len(terms & symbol_parts) / max(1, len(symbol_parts)) if symbol_parts else 0.0
            path_hit = len(terms & set(code_tokens(chunk.file_path))) / max(1, len(terms)) if terms else 0.0
            prior = sum(ROLE_PRIORS.get(intent, {}).get(chunk.role, 0.0) for intent in intents)
            base = (chunk.score or 0.0) / max_fused
            score = 0.45 * base + 0.3 * coverage + 0.35 * identifier_hit + 0.3 * symbol_hit + 0.1 * path_hit + prior
            clone = Chunk.from_dict(chunk.to_dict())
            clone.score = round(score, 6)
            clone.metadata = {**clone.metadata, "fused_score": chunk.score}
            rescored.append(clone)
        rescored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return rescored


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, model_name: str, device: str | None = None) -> None:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as exc:
            raise RuntimeError("sentence-transformers is not installed") from exc
        try:
            self.model = CrossEncoder(model_name, device=device)
        except Exception as exc:
            raise RuntimeError(f"could not load reranker {model_name}: {exc}") from exc
        self.fallback = LocalReranker()
        self.device = str(self.model.device)
        self.revision = model_revision(model_name, getattr(self.model.model, "config", None))
        self.model_id = f"{model_name}@{self.revision or 'unversioned'}"

    def rerank(self, query: str, chunks: list[Chunk], intents: list[str]) -> list[Chunk]:
        if not chunks:
            return []
        local = {chunk.id: chunk.score or 0.0 for chunk in self.fallback.rerank(query, chunks, intents)}
        raw = self.model.predict([(query, chunk.index_text[:2000]) for chunk in chunks])
        rescored = []
        for chunk, value in zip(chunks, raw, strict=True):
            relevance = 1 / (1 + math.exp(-max(-80, min(80, float(value)))))
            clone = Chunk.from_dict(chunk.to_dict())
            clone.score = round(relevance * 0.7 + 0.3 * local.get(chunk.id, 0.0), 6)
            # A sigmoid score for ranking, not a calibrated answer-support probability.
            clone.metadata = {**clone.metadata, "fused_score": chunk.score, "rerank_relevance": round(relevance, 4)}
            rescored.append(clone)
        rescored.sort(key=lambda item: item.score or 0.0, reverse=True)
        return rescored


def build_reranker(provider: str, model_name: str) -> LocalReranker | CrossEncoderReranker:
    if provider.lower() in {"cross-encoder", "cross_encoder", "auto"}:
        try:
            return CrossEncoderReranker(model_name)
        except RuntimeError as exc:
            logger.warning("Cross-encoder reranker unavailable (%s); using local reranker.", exc)
    return LocalReranker()
