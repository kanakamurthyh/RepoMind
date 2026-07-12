from __future__ import annotations

import ast
import hashlib
import re

from app.models import Chunk, FileRecord


def _chunk_id(repo_id: str, file_path: str, start: int, end: int, text: str) -> str:
    digest = hashlib.sha1(f"{repo_id}:{file_path}:{start}:{end}:{text[:80]}".encode("utf-8")).hexdigest()[:16]
    return f"{repo_id}:{digest}"


def _make_chunk(record: FileRecord, start: int, end: int, lines: list[str]) -> Chunk:
    text = "\n".join(lines).strip()
    return Chunk(
        id=_chunk_id(record.repo_id, record.path, start, end, text),
        repo_id=record.repo_id,
        file_path=record.path,
        language=record.language,
        role=record.role,
        start_line=start,
        end_line=end,
        text=text,
        metadata={
            "file_kind": record.kind,
            "mime_type": record.mime_type,
            "warnings": list(record.warnings),
            "truncated": record.truncated,
        },
    )


class Chunker:
    def __init__(self, max_lines: int = 80, overlap: int = 10) -> None:
        self.max_lines = max_lines
        self.overlap = overlap

    def chunk_files(self, records: list[FileRecord], max_chunks: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for record in records:
            chunks.extend(self.chunk_file(record))
            if len(chunks) >= max_chunks:
                return chunks[:max_chunks]
        return chunks

    def chunk_file(self, record: FileRecord) -> list[Chunk]:
        if not record.text.strip():
            return []
        if record.language == "python" and record.role in {"code", "test"}:
            chunks = self._chunk_python(record)
            if chunks:
                return chunks
        if record.language in {"javascript", "typescript"} and record.role in {"code", "test"}:
            chunks = self._chunk_javascript(record)
            if chunks:
                return chunks
        if record.role == "docs":
            return self._chunk_docs(record)
        if record.role == "config":
            return self._chunk_lines(record, max_lines=40, overlap=5)
        return self._chunk_lines(record, max_lines=self.max_lines, overlap=self.overlap)

    def _chunk_python(self, record: FileRecord) -> list[Chunk]:
        lines = record.text.splitlines()
        try:
            tree = ast.parse(record.text)
        except SyntaxError:
            return self._chunk_lines(record, self.max_lines, self.overlap)

        ranges: list[tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = getattr(node, "lineno", 1)
                end = getattr(node, "end_lineno", start)
                ranges.append((start, end))
        ranges.sort()

        chunks: list[Chunk] = []
        covered: set[int] = set()
        for start, end in ranges:
            if end - start + 1 > self.max_lines * 2:
                for sub_start in range(start, end + 1, self.max_lines - self.overlap):
                    sub_end = min(end, sub_start + self.max_lines - 1)
                    chunks.append(_make_chunk(record, sub_start, sub_end, lines[sub_start - 1 : sub_end]))
            else:
                chunks.append(_make_chunk(record, start, end, lines[start - 1 : end]))
            covered.update(range(start, end + 1))

        module_lines = [(idx, line) for idx, line in enumerate(lines, start=1) if idx not in covered and line.strip()]
        if module_lines:
            grouped: list[str] = []
            start = module_lines[0][0]
            last = start
            for idx, line in module_lines:
                if len(grouped) >= 35 or idx > last + 3:
                    chunks.append(_make_chunk(record, start, last, grouped))
                    grouped = []
                    start = idx
                grouped.append(line)
                last = idx
            if grouped:
                chunks.append(_make_chunk(record, start, last, grouped))
        return [chunk for chunk in chunks if chunk.text]

    def _chunk_javascript(self, record: FileRecord) -> list[Chunk]:
        lines = record.text.splitlines()
        starts: list[int] = []
        pattern = re.compile(
            r"^\s*(export\s+)?(async\s+)?(function\s+[\w$]+|class\s+[\w$]+|const\s+[\w$]+\s*=\s*(async\s*)?\(|[\w$]+\s*:\s*(async\s*)?\()"
        )
        for index, line in enumerate(lines, start=1):
            if pattern.search(line):
                starts.append(index)
        if not starts:
            return self._chunk_lines(record, self.max_lines, self.overlap)

        chunks: list[Chunk] = []
        for pos, start in enumerate(starts):
            next_start = starts[pos + 1] if pos + 1 < len(starts) else len(lines) + 1
            end = min(next_start - 1, start + self.max_lines - 1)
            chunks.append(_make_chunk(record, start, end, lines[start - 1 : end]))

        top_level = lines[: starts[0] - 1]
        if any(line.strip() for line in top_level):
            chunks.insert(0, _make_chunk(record, 1, starts[0] - 1, top_level))
        return [chunk for chunk in chunks if chunk.text]

    def _chunk_docs(self, record: FileRecord) -> list[Chunk]:
        lines = record.text.splitlines()
        chunks: list[Chunk] = []
        current: list[str] = []
        start_line = 1
        for index, line in enumerate(lines, start=1):
            heading = re.match(r"^\s{0,3}#{1,6}\s+", line)
            if heading and current:
                chunks.append(_make_chunk(record, start_line, index - 1, current))
                current = []
                start_line = index
            if not current and line.strip():
                start_line = index
            current.append(line)
            if len(current) >= self.max_lines:
                chunks.append(_make_chunk(record, start_line, index, current))
                current = []
        if current and any(line.strip() for line in current):
            chunks.append(_make_chunk(record, start_line, len(lines), current))
        return [chunk for chunk in chunks if chunk.text]

    def _chunk_lines(self, record: FileRecord, max_lines: int, overlap: int) -> list[Chunk]:
        lines = record.text.splitlines()
        chunks: list[Chunk] = []
        step = max(1, max_lines - overlap)
        for start_index in range(0, len(lines), step):
            end_index = min(len(lines), start_index + max_lines)
            chunk_lines = lines[start_index:end_index]
            if any(line.strip() for line in chunk_lines):
                chunks.append(_make_chunk(record, start_index + 1, end_index, chunk_lines))
            if end_index == len(lines):
                break
        return chunks
