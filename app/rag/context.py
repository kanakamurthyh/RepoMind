"""Source packing that preserves exact, complete visible line ranges."""

from __future__ import annotations


def visible_lines(text: str, max_chars: int, max_bytes: int | None = None) -> str:
    """Keep complete lines only, within character and conservative byte-token budgets.

    UTF-8 bytes are a conservative upper bound for byte-level tokenizers. This
    works offline without pretending a character estimate is a model tokenizer.
    An oversized first line contributes no source evidence.
    """
    selected: list[str] = []
    chars = byte_count = 0
    for line in text.splitlines():
        separator = 1 if selected else 0
        extra_chars = len(line) + separator
        extra_bytes = len(line.encode("utf-8")) + separator
        if chars + extra_chars > max_chars or (max_bytes is not None and byte_count + extra_bytes > max_bytes):
            break
        selected.append(line)
        chars += extra_chars
        byte_count += extra_bytes
    return "\n".join(selected)
