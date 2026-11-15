"""Code-aware tokenization shared by BM25, hashing embeddings, and reranking."""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\b")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how", "i", "in", "is", "it",
    "its", "me", "my", "of", "on", "or", "the", "this", "that", "to", "what", "when", "where", "which", "who", "why",
    "with", "you", "your", "we", "our", "there", "these", "those", "into", "about", "work", "works", "used", "use",
    "using", "file", "files", "code", "repo", "repository", "project", "explain", "show", "tell", "please", "get",
    "self", "return", "def", "import", "none", "true", "false", "const", "let", "var", "function", "class", "new",
}


def stems(word: str) -> list[str]:
    """Light suffix stripping (plural / past / gerund) so 'passwords' ~ 'password', 'hashed' ~ 'hash'."""
    if len(word) <= 4 or not word.isalpha():
        return []
    if word.endswith("ies"):
        return [word[:-3] + "y"]
    if word.endswith("ied"):
        return [word[:-3] + "y"]
    if word.endswith("ing") and len(word) > 6:
        return [word[:-3], word[:-3] + "e"]
    if word.endswith("ed") and len(word) > 5:
        return [word[:-2], word[:-1]]
    if word.endswith(("sses", "xes", "zes", "ches", "shes")):
        return [word[:-2]]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return [word[:-1]]
    return []


def split_identifier(token: str) -> list[str]:
    parts: list[str] = []
    for piece in token.split("_"):
        if piece:
            parts.extend(match.group(0).lower() for match in CAMEL_RE.finditer(piece))
    return parts


def code_tokens(text: str, keep_stopwords: bool = False) -> list[str]:
    """Lowercased tokens: whole identifiers plus their snake/camel sub-words."""
    tokens: list[str] = []
    for match in WORD_RE.finditer(text):
        raw = match.group(0)
        lowered = raw.lower()
        if len(lowered) < 2 and not lowered.isdigit():
            continue
        subparts = split_identifier(raw)
        if keep_stopwords or lowered not in STOPWORDS:
            tokens.append(lowered)
            tokens.extend(stems(lowered))
        if len(subparts) > 1:
            for part in subparts:
                if len(part) > 1 and (keep_stopwords or part not in STOPWORDS):
                    tokens.append(part)
                    tokens.extend(stems(part))
    return tokens


def query_identifiers(query: str) -> set[str]:
    """Identifiers the user probably means literally (snake_case, CamelCase, dotted, or code-quoted)."""
    identifiers: set[str] = set()
    for quoted in re.findall(r"`([^`]+)`", query):
        identifiers.update(IDENTIFIER_RE.findall(quoted))
    for match in IDENTIFIER_RE.finditer(query):
        token = match.group(0)
        if "_" in token or "." in token or re.search(r"[a-z][A-Z]", token) or re.match(r"^[A-Z][a-z]+[A-Z]", token):
            identifiers.add(token)
    expanded: set[str] = set()
    for identifier in identifiers:
        expanded.add(identifier)
        expanded.add(identifier.split(".")[-1])
    return {item for item in expanded if len(item) > 2}
