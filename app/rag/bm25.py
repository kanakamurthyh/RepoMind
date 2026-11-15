from __future__ import annotations

import math
from collections import Counter

from app.rag.tokenize import code_tokens


class BM25Index:
    """Okapi BM25 over code-aware tokens (identifiers + snake/camel sub-words)."""

    def __init__(self, documents: list[str], k1: float = 1.4, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.term_freqs: list[Counter[str]] = [Counter(code_tokens(doc)) for doc in documents]
        self.doc_lengths = [sum(tf.values()) for tf in self.term_freqs]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0
        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            df.update(tf.keys())
        count = len(documents)
        self.idf = {term: math.log(1 + (count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        self.postings: dict[str, list[int]] = {}
        for index, tf in enumerate(self.term_freqs):
            for term in tf:
                self.postings.setdefault(term, []).append(index)

    def scores(self, query: str) -> dict[int, float]:
        terms = sorted(term for term in set(code_tokens(query)) if term in self.idf)
        scores: dict[int, float] = {}
        for term in terms:
            idf = self.idf[term]
            for index in self.postings.get(term, []):
                freq = self.term_freqs[index][term]
                length_norm = 1 - self.b + self.b * (self.doc_lengths[index] / (self.avg_length or 1.0))
                scores[index] = scores.get(index, 0.0) + idf * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)
        return scores

    def top(self, query: str, k: int, allowed: set[int] | None = None) -> list[tuple[int, float]]:
        scored = self.scores(query)
        items = [(index, score) for index, score in scored.items() if allowed is None or index in allowed]
        items.sort(key=lambda item: (-item[1], item[0]))
        return items[:k]
