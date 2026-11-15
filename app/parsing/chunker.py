from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

from app.models import Chunk, CodeEntity, FileRecord

logger = logging.getLogger(__name__)

ROLE_PRIORITY = {"docs": 0, "code": 1, "config": 2, "test": 3, "other": 4}
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
SAMPLE_SIZE = 20
DUPLICATE_PATHS_PER_CHUNK = 20


def _chunk_id(repo_id: str, file_path: str, start: int, end: int, symbol: str | None) -> str:
    digest = hashlib.sha1(f"{repo_id}:{file_path}:{start}:{end}:{symbol or ''}".encode()).hexdigest()[:16]
    return f"{repo_id}:{digest}"


def _make_chunk(record: FileRecord, start: int, end: int, lines: list[str], symbol: str | None = None, header: str | None = None, kind: str = "block") -> Chunk:
    text = "\n".join(lines).rstrip()
    return Chunk(
        id=_chunk_id(record.repo_id, record.path, start, end, symbol),
        repo_id=record.repo_id,
        file_path=record.path,
        language=record.language,
        role=record.role,
        start_line=start,
        end_line=end,
        text=text,
        symbol=symbol,
        metadata={
            "header": header or (f"{record.path} › {symbol}" if symbol else record.path),
            "chunk_kind": kind,
            "file_kind": record.kind,
            "mime_type": record.mime_type,
            "warnings": list(record.warnings),
            "truncated": record.truncated,
        },
    )


class Chunker:
    """Role-aware chunking.

    Code is chunked along parsed symbol boundaries (Python AST or tree-sitter entities):
    small definitions become one chunk, large containers are split into a header chunk
    plus one chunk per child definition, and anything still too long is windowed with
    overlap. No line of a file is dropped and nested definitions are never duplicated.
    """

    def __init__(self, max_lines: int = 80, overlap: int = 10, collapse_duplicates: bool = True) -> None:
        self.max_lines = max_lines
        self.overlap = overlap
        self.collapse_duplicates = collapse_duplicates
        self.last_stats: dict[str, Any] = {}

    def chunk_files(
        self,
        records: list[FileRecord],
        max_chunks: int,
        entities_by_file: dict[str, list[CodeEntity]] | None = None,
    ) -> list[Chunk]:
        """Chunk every indexable file, highest-priority roles first, up to ``max_chunks``.

        Nothing is dropped silently: ``last_stats`` records how many files the chunk cap
        excluded (with a sample of their paths) and which byte-identical files were
        collapsed onto a single indexed copy.
        """
        entities_by_file = entities_by_file or {}
        ordered = sorted(
            (record for record in records if record.kind == "text" and record.role != "generated"),
            key=lambda record: (ROLE_PRIORITY.get(record.role, 5), record.path.count("/"), record.path),
        )
        canonical_by_digest: dict[str, FileRecord] = {}
        duplicates_of: dict[str, list[str]] = {}
        to_chunk: list[FileRecord] = []
        for record in ordered:
            key = self._duplicate_key(record)
            if key is not None and key in canonical_by_digest:
                duplicates_of.setdefault(canonical_by_digest[key].path, []).append(record.path)
                continue
            if key is not None:
                canonical_by_digest[key] = record
            to_chunk.append(record)

        chunks: list[Chunk] = []
        dropped_files: list[str] = []
        for record in to_chunk:
            file_chunks = self.chunk_file(record, entities_by_file.get(record.path))
            if len(chunks) + len(file_chunks) > max_chunks:
                dropped_files.append(record.path)
                continue
            copies = duplicates_of.get(record.path)
            if copies:
                for chunk in file_chunks:
                    chunk.metadata["duplicate_paths"] = copies[:DUPLICATE_PATHS_PER_CHUNK]
                    chunk.metadata["duplicate_count"] = len(copies)
            chunks.extend(file_chunks)

        # Duplicates of a file the cap dropped are dropped with it.
        dropped_set = set(dropped_files)
        dropped_with_canonical = sum(len(copies) for path, copies in duplicates_of.items() if path in dropped_set)
        duplicates_collapsed = sum(len(copies) for copies in duplicates_of.values()) - dropped_with_canonical
        files_dropped = len(dropped_files) + dropped_with_canonical
        if dropped_files:
            logger.warning(
                "Chunk cap %d reached; %d file(s) were not indexed (lowest priority first, e.g. %s)",
                max_chunks, files_dropped, ", ".join(dropped_files[:3]),
            )
        if duplicates_collapsed:
            logger.info("Collapsed %d byte-identical duplicate file(s) onto one indexed copy each", duplicates_collapsed)
        self.last_stats = {
            "chunks": len(chunks),
            "chunk_cap": max_chunks,
            "chunk_cap_reached": bool(dropped_files),
            "files_considered": len(ordered),
            "files_indexed": len(to_chunk) - len(dropped_files),
            # `files_dropped` is the long-standing name; `files_dropped_by_chunk_cap` is the explicit one.
            "files_dropped": files_dropped,
            "files_dropped_by_chunk_cap": files_dropped,
            "dropped_paths_sample": dropped_files[:SAMPLE_SIZE],
            "duplicates_collapsed": duplicates_collapsed,
            "duplicate_groups_sample": [
                {"indexed_path": path, "duplicate_paths": copies[:DUPLICATE_PATHS_PER_CHUNK], "count": len(copies)}
                for path, copies in list(duplicates_of.items())[:SAMPLE_SIZE]
                if path not in dropped_set
            ],
        }
        return chunks

    def _duplicate_key(self, record: FileRecord) -> str | None:
        """Only complete (untruncated), non-trivial files with a content hash are collapsed."""
        if not self.collapse_duplicates or not record.sha256 or record.truncated or not record.text.strip():
            return None
        return record.sha256

    def chunk_file(self, record: FileRecord, entities: list[CodeEntity] | None = None) -> list[Chunk]:
        if not record.text.strip() or record.role == "generated":
            return []
        if record.role == "docs" and record.language in {"markdown", "rst", "text"}:
            return self._chunk_docs(record)
        if record.role in {"code", "test"}:
            if entities is None and record.language == "python":
                from app.parsing.python_parser import PythonParser

                entities = PythonParser().parse(record).entities
            symbol_entities = [e for e in (entities or []) if e.kind in {"function", "class"}]
            if symbol_entities:
                return self._chunk_by_symbols(record, symbol_entities)
        if record.role == "config":
            return self._chunk_lines(record, 1, len(record.text.splitlines()), max_lines=40, overlap=5)
        return self._chunk_lines(record, 1, len(record.text.splitlines()), self.max_lines, self.overlap)

    # --- code ---------------------------------------------------------------------------
    def _chunk_by_symbols(self, record: FileRecord, entities: list[CodeEntity]) -> list[Chunk]:
        lines = record.text.splitlines()
        total = len(lines)
        spans = self._normalize_spans(entities, total)
        chunks: list[Chunk] = []
        covered: set[int] = set()
        self._emit_spans(record, lines, spans, chunks, covered)
        chunks.extend(self._module_chunks(record, lines, covered))
        chunks.sort(key=lambda chunk: (chunk.start_line, chunk.end_line))
        return [chunk for chunk in chunks if chunk.text.strip()]

    def _normalize_spans(self, entities: list[CodeEntity], total: int) -> list[tuple[int, int, CodeEntity]]:
        ordered = sorted(entities, key=lambda e: (e.start_line, -e.end_line))
        spans: list[tuple[int, int, CodeEntity]] = []
        for index, entity in enumerate(ordered):
            start = max(1, entity.start_line)
            end = min(total, entity.end_line)
            if end <= start:
                # Regex backends only know the declaration line: extend to the next declaration.
                following = [e.start_line for e in ordered[index + 1 :] if e.start_line > start]
                end = min(total, (following[0] - 1) if following else total)
            spans.append((start, max(start, end), entity))
        return spans

    def _emit_spans(self, record: FileRecord, lines: list[str], spans: list[tuple[int, int, CodeEntity]], chunks: list[Chunk], covered: set[int]) -> None:
        top_level = [span for span in spans if not any(o is not span and o[0] <= span[0] and span[1] <= o[1] and (o[0], o[1]) != (span[0], span[1]) for o in spans)]
        seen_ranges: set[tuple[int, int]] = set()
        for start, end, entity in top_level:
            if (start, end) in seen_ranges or start in covered:
                continue
            seen_ranges.add((start, end))
            qualname = entity.qualname
            header = f"{record.path} › {entity.kind} {qualname}"
            size = end - start + 1
            children = [s for s in spans if s[0] >= start and s[1] <= end and (s[0], s[1]) != (start, end)]
            if size <= int(self.max_lines * 1.5):
                chunks.append(_make_chunk(record, start, end, lines[start - 1 : end], qualname, header, "symbol"))
                covered.update(range(start, end + 1))
            elif children:
                first_child = min(child[0] for child in children)
                if first_child > start:
                    head_end = first_child - 1
                    chunks.extend(self._windows(record, lines, start, head_end, qualname, header + " (header)"))
                    covered.update(range(start, head_end + 1))
                self._emit_spans(record, lines, children, chunks, covered)
                remaining = [line for line in range(start, end + 1) if line not in covered]
                for block_start, block_end in self._contiguous(remaining):
                    chunks.extend(self._windows(record, lines, block_start, block_end, qualname, header))
                    covered.update(range(block_start, block_end + 1))
            else:
                chunks.extend(self._windows(record, lines, start, end, qualname, header))
                covered.update(range(start, end + 1))

    def _module_chunks(self, record: FileRecord, lines: list[str], covered: set[int]) -> list[Chunk]:
        remaining = [index for index in range(1, len(lines) + 1) if index not in covered and lines[index - 1].strip()]
        chunks: list[Chunk] = []
        # Merge nearby module-level lines (imports, constants, app wiring) into compact chunks.
        for block_start, block_end in self._contiguous(remaining, gap=3):
            chunks.extend(self._windows(record, lines, block_start, block_end, None, f"{record.path} › module scope", max_lines=40, overlap=0))
        return chunks

    def _windows(self, record: FileRecord, lines: list[str], start: int, end: int, symbol: str | None, header: str, max_lines: int | None = None, overlap: int | None = None) -> list[Chunk]:
        max_lines = max_lines or self.max_lines
        overlap = self.overlap if overlap is None else overlap
        step = max(1, max_lines - overlap)
        chunks = []
        cursor = start
        while cursor <= end:
            window_end = min(end, cursor + max_lines - 1)
            window = lines[cursor - 1 : window_end]
            if any(line.strip() for line in window):
                part_symbol = symbol if (cursor == start and window_end == end) or symbol is None else f"{symbol}#L{cursor}"
                chunks.append(_make_chunk(record, cursor, window_end, window, part_symbol if symbol else None, header, "window" if symbol is None else "symbol"))
            if window_end == end:
                break
            cursor += step
        return chunks

    def _contiguous(self, numbers: list[int], gap: int = 1) -> list[tuple[int, int]]:
        blocks: list[tuple[int, int]] = []
        for number in numbers:
            if blocks and number - blocks[-1][1] <= gap:
                blocks[-1] = (blocks[-1][0], number)
            else:
                blocks.append((number, number))
        return blocks

    # --- docs -----------------------------------------------------------------------------
    def _chunk_docs(self, record: FileRecord) -> list[Chunk]:
        lines = record.text.splitlines()
        chunks: list[Chunk] = []
        current: list[str] = []
        start_line = 1
        heading_path: list[tuple[int, str]] = []
        in_fence = False

        def flush(end_line: int) -> None:
            if current and any(line.strip() for line in current):
                trail = " › ".join(title for _, title in heading_path) or record.path
                header = f"{record.path} › {trail}" if heading_path else record.path
                section = heading_path[-1][1] if heading_path else None
                chunks.extend(
                    self._windows(record, lines, start_line, end_line, section, header, max_lines=self.max_lines, overlap=5)
                    if end_line - start_line + 1 > self.max_lines
                    else [_make_chunk(record, start_line, end_line, lines[start_line - 1 : end_line], section, header, "section")]
                )

        for index, line in enumerate(lines, start=1):
            if FENCE_RE.match(line):
                in_fence = not in_fence
            heading = None if in_fence else HEADING_RE.match(line)
            if heading:
                flush(index - 1)
                level = len(heading.group(1))
                heading_path = [(lvl, title) for lvl, title in heading_path if lvl < level] + [(level, heading.group(2).strip())]
                current = [line]
                start_line = index
                continue
            if not current:
                start_line = index
            current.append(line)
        flush(len(lines))
        return [chunk for chunk in chunks if chunk.text.strip()]

    # --- plain lines --------------------------------------------------------------------------
    def _chunk_lines(self, record: FileRecord, start: int, end: int, max_lines: int, overlap: int) -> list[Chunk]:
        lines = record.text.splitlines()
        return self._windows(record, lines, start, end, None, record.path, max_lines=max_lines, overlap=overlap)
