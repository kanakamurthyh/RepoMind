from __future__ import annotations

from app.models import Chunk


def reciprocal_rank_fusion(
    channels: dict[str, list[Chunk]],
    weights: dict[str, float] | None = None,
    k: int = 60,
) -> tuple[list[Chunk], dict[str, dict[str, int]]]:
    """Fuse ranked lists with weighted RRF.

    Returns fused chunks (score = fused RRF score) and, per chunk id, the rank it had in
    each channel - kept for debugging and evaluation, never shown to end users.
    """
    weights = weights or {}
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    provenance: dict[str, dict[str, int]] = {}
    for channel, chunks in channels.items():
        weight = weights.get(channel, 1.0)
        if weight <= 0:
            continue
        seen: set[str] = set()
        for rank, chunk in enumerate(chunks, start=1):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            fused[chunk.id] = fused.get(chunk.id, 0.0) + weight / (k + rank)
            chunk_by_id.setdefault(chunk.id, chunk)
            provenance.setdefault(chunk.id, {})[channel] = rank
    ordered = sorted(fused, key=lambda chunk_id: fused[chunk_id], reverse=True)
    results = []
    for chunk_id in ordered:
        chunk = Chunk.from_dict(chunk_by_id[chunk_id].to_dict())
        chunk.score = round(fused[chunk_id], 6)
        results.append(chunk)
    return results, provenance
