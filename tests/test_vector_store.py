from __future__ import annotations

import pytest

from app.config import Settings
from app.models import Chunk
from app.rag.embeddings import LocalHashingEmbedder
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import IndexStaleError, VectorStore

CHUNKS = [
    Chunk(f"demo:{i}", "demo", f"src/mod{i}.py", "python", "code", 1, 5, f"def helper_{i}():\n    return {i}", symbol=f"helper_{i}")
    for i in range(6)
] + [Chunk("demo:auth", "demo", "src/auth.py", "python", "code", 1, 4, "def verify_jwt_token(token):\n    return decode(token)", symbol="verify_jwt_token")]


def test_dense_lexical_and_range_channels(tmp_path):
    store = VectorStore(tmp_path, LocalHashingEmbedder(), app_settings=Settings(data_dir=tmp_path))
    store.build("demo", CHUNKS)

    assert store.search("demo", "verify jwt token", top_k=3)[0].file_path == "src/auth.py"
    assert store.lexical_search("demo", "verify_jwt_token", top_k=3)[0].file_path == "src/auth.py"
    assert [c.file_path for c in store.chunks_for_ranges("demo", [("src/auth.py", 2, 2)])] == ["src/auth.py"]
    assert store.chunks_for_files("demo", {"src/mod1.py"}, query="helper")[0].symbol == "helper_1"


def test_index_built_with_other_embedder_is_detected_as_stale(tmp_path):
    settings = Settings(data_dir=tmp_path)
    VectorStore(tmp_path, LocalHashingEmbedder(dimensions=512), app_settings=settings).build("demo", CHUNKS)
    other = VectorStore(tmp_path, LocalHashingEmbedder(dimensions=384), app_settings=settings)

    assert other.exists("demo") is False
    with pytest.raises(IndexStaleError):
        other.search("demo", "anything")


def test_chroma_backend_returns_distinct_ids(tmp_path):
    pytest.importorskip("chromadb")
    settings = Settings(data_dir=tmp_path, vector_store_provider="chroma")
    store = VectorStore(tmp_path, LocalHashingEmbedder(), app_settings=settings)
    assert store.backend_name == "chroma"
    store.build("owner-repo-0123456789", [Chunk(c.id.replace("demo", "r"), "r", c.file_path, c.language, c.role, c.start_line, c.end_line, c.text, symbol=c.symbol) for c in CHUNKS])

    results = store.search("owner-repo-0123456789", "helper function", top_k=5)
    assert len({chunk.id for chunk in results}) == len(results) == 5
    context = HybridRetriever(store).retrieve("owner-repo-0123456789", "helper function", {"facts": [], "nodes": [], "edges": []})
    assert len(context["selected_chunks"]) > 1
