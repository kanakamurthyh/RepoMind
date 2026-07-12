from __future__ import annotations

from app.ingestion.scanner import FileScanner


def test_scanner_ignores_common_generated_directories(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n\nA small project.", encoding="utf-8")
    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "ignored.js").write_text("console.log('x')", encoding="utf-8")
    (repo / "app.py").write_text("print('hello')", encoding="utf-8")

    records = FileScanner(max_file_size_bytes=1000).scan("demo", repo)
    paths = {record.path for record in records}

    assert "README.md" in paths
    assert "app.py" in paths
    assert "node_modules/ignored.js" not in paths
    assert next(record for record in records if record.path == "README.md").role == "docs"


def test_scanner_truncates_large_text_files(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "large.txt").write_text("a" * 100, encoding="utf-8")

    records = FileScanner(max_file_size_bytes=10).scan("demo", repo)

    assert records[0].truncated is True
    assert len(records[0].text) == 10


def test_scanner_captures_binary_metadata_without_embedding_content(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 20)

    records = FileScanner(max_file_size_bytes=1000).scan("demo", repo)

    assert len(records) == 1
    assert records[0].kind == "binary"
    assert records[0].text == ""
    assert records[0].size_bytes > 0
    assert "content was not embedded" in records[0].warnings[0]


def test_scanner_records_symlink_warning_without_following(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (repo / "outside.txt").symlink_to(outside)

    records = FileScanner(max_file_size_bytes=1000).scan("demo", repo)

    assert records[0].kind == "symlink"
    assert records[0].text == ""
    assert "Symlink skipped" in records[0].warnings[0]
