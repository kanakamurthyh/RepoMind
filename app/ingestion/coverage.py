"""Index coverage reporting: what the scan/chunk caps left out of an analysis, and why.

The ``coverage`` object is part of the public API (``summary["coverage"]``), so its keys
are stable and every value is JSON-friendly. Older cached analyses that predate it are
upgraded on read via :func:`ensure_coverage`.
"""

from __future__ import annotations

from typing import Any

COVERAGE_VERSION = 1

_REASON_TEXT = {
    "max_files_per_repo": "the scanner stopped at the per-repository file limit (MAX_FILES_PER_REPO={limit})",
    "max_total_repo_bytes": "the scanner stopped at the per-repository byte limit (MAX_TOTAL_REPO_BYTES={limit})",
    "scan_limit": "the scanner stopped at a per-repository limit (file count or total bytes); this older analysis did not record how many files were skipped",
    "chunk_cap": "the index reached the chunk limit (MAX_CHUNKS_PER_REPO={limit}); lowest-priority files (tests, config, deep paths) were left out first",
}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def build_coverage(
    scan_stats: dict[str, Any],
    chunk_stats: dict[str, Any],
    limits: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Combine scanner and chunker stats into the stable ``coverage`` object."""
    limits = limits or {}
    files_skipped_by_cap = _int(scan_stats.get("files_skipped_by_cap"))
    files_dropped_by_chunk_cap = _int(chunk_stats.get("files_dropped_by_chunk_cap", chunk_stats.get("files_dropped")))
    files_scanned = _int(scan_stats.get("files_scanned", scan_stats.get("files_seen")))
    files_seen = _int(scan_stats.get("files_seen")) or files_scanned + files_skipped_by_cap

    reasons: list[dict[str, Any]] = []
    stop_reason = scan_stats.get("stop_reason")
    if stop_reason or files_skipped_by_cap:
        code = stop_reason or "max_files_per_repo"
        limit = limits.get(code)
        reasons.append({
            "code": code,
            "limit": limit,
            "files_affected": files_skipped_by_cap,
            "message": _REASON_TEXT.get(code, code).format(limit=limit if limit is not None else "?"),
        })
    if files_dropped_by_chunk_cap or chunk_stats.get("chunk_cap_reached"):
        limit = chunk_stats.get("chunk_cap", limits.get("chunk_cap"))
        reasons.append({
            "code": "chunk_cap",
            "limit": limit,
            "files_affected": files_dropped_by_chunk_cap,
            "message": _REASON_TEXT["chunk_cap"].format(limit=limit if limit is not None else "?"),
        })
    incomplete = {"truncated_files": _int(scan_stats.get("truncated")), "unreadable_files": _int(scan_stats.get("skipped")),
                  "parse_failures": _int(scan_stats.get("parse_failures"))}
    for code, count in incomplete.items():
        if count:
            reasons.append({"code": code, "limit": None, "files_affected": count,
                            "message": f"{count} file(s) affected by {code.replace('_', ' ')}"})

    not_analyzed = files_skipped_by_cap + files_dropped_by_chunk_cap
    analyzed_fraction = round(max(0.0, (files_seen - not_analyzed) / files_seen), 4) if files_seen else 1.0
    sample = list(scan_stats.get("skipped_by_cap_sample") or [])[:10] + list(chunk_stats.get("dropped_paths_sample") or [])[:10]
    return {
        "version": COVERAGE_VERSION,
        "partial": bool(reasons),
        "analyzed_fraction": analyzed_fraction,
        "fraction_basis": "file presence after repository caps; not source-content or semantic coverage",
        **incomplete,
        "files_seen": files_seen,
        "files_scanned": files_scanned,
        "files_indexed": _int(chunk_stats.get("files_indexed")),
        "files_skipped_by_cap": files_skipped_by_cap,
        "files_skipped_count_exact": bool(scan_stats.get("skipped_by_cap_exact", True)),
        "files_dropped_by_chunk_cap": files_dropped_by_chunk_cap,
        "duplicates_collapsed": _int(chunk_stats.get("duplicates_collapsed")),
        "duplicate_groups_sample": list(chunk_stats.get("duplicate_groups_sample") or [])[:10],
        "sample_dropped_paths": sample,
        "reasons": reasons,
    }


def coverage_message(coverage: dict[str, Any]) -> str:
    """One human-readable sentence describing partial coverage ("" when complete)."""
    if not coverage.get("partial"):
        return ""
    fraction = coverage.get("analyzed_fraction")
    share = f"{fraction:.0%} of" if isinstance(fraction, (int, float)) else "part of"
    why = "; ".join(reason.get("message", reason.get("code", "")) for reason in coverage.get("reasons", []))
    return f"Partial analysis: {share} {coverage.get('files_seen', '?')} files were analyzed because {why}."


def ensure_coverage(summary: dict[str, Any]) -> dict[str, Any]:
    """Return ``summary`` with a ``coverage`` object, deriving one for analyses cached before it existed."""
    if not isinstance(summary, dict) or isinstance(summary.get("coverage"), dict):
        return summary
    ingestion = summary.get("ingestion") or {}
    scan_stats = dict(ingestion)
    if ingestion.get("limit_reached") and not ingestion.get("stop_reason"):
        # Legacy scans only recorded that *a* scan cap was hit, not which one or how many files it skipped.
        scan_stats["stop_reason"] = "scan_limit"
        scan_stats["skipped_by_cap_exact"] = False
    coverage = build_coverage(scan_stats, ingestion)
    coverage["derived_from_legacy_stats"] = True
    return {**summary, "coverage": coverage}
