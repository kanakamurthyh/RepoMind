"""Retrieval metrics with explicit hit, fractional recall and completeness semantics."""

from __future__ import annotations


def retrieval_metrics(rankings: list[list[str]], gold: list[set[str]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    return alternative_retrieval_metrics(rankings, [[g] if g else [] for g in gold], ks)


def alternative_retrieval_metrics(rankings: list[list[str]], gold: list[list[set[str]]], ks: tuple[int, ...] = (1, 3, 5)) -> dict[str, float]:
    if len(rankings) != len(gold):
        raise ValueError("Rankings and gold labels must have equal lengths")
    answerable = [(list(dict.fromkeys(r)), g) for r, g in zip(rankings, gold, strict=True) if g]
    if any(not alternative for _, alternatives in answerable for alternative in alternatives):
        raise ValueError("Gold alternatives cannot contain empty evidence sets")
    if not answerable:
        return {}
    count = len(answerable)
    result = {}
    for k in ks:
        if k < 1:
            raise ValueError("k must be positive")
        result[f"hit@{k}"] = sum(any(set(r[:k]) & g for g in alternatives) for r, alternatives in answerable) / count
        result[f"recall@{k}"] = sum(max(len(set(r[:k]) & g) / len(g) for g in alternatives) for r, alternatives in answerable) / count
        result[f"complete@{k}"] = sum(any(g <= set(r[:k]) for g in alternatives) for r, alternatives in answerable) / count
    result["mrr"] = sum(next((1 / i for i, path in enumerate(r, 1) if any(path in g for g in alternatives)), 0) for r, alternatives in answerable) / count
    return {key: round(value, 4) for key, value in result.items()}
