from __future__ import annotations

from app.models import Chunk
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore


def test_vector_store_and_hybrid_retriever_return_citations(tmp_path):
    store = VectorStore(tmp_path)
    chunks = [
        Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 5, "This project uses FastAPI for a repository analysis API."),
        Chunk("c2", "demo", "tests/test_app.py", "python", "test", 1, 3, "Tests validate repository scanning."),
    ]
    store.build("demo", chunks)

    context = HybridRetriever(store).retrieve(
        "demo",
        "Where is the health route defined?",
        {"facts": ["app/main.py declares route /health handled near lines 1-5"]},
    )

    assert context["chunks"]
    assert context["source_references"][0]["file_path"] in {"README.md", "tests/test_app.py"}
    assert context["graph_facts"]


def test_hybrid_retriever_boosts_files_from_graph_edges(tmp_path):
    store = VectorStore(tmp_path)
    chunks = [
        Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 4, "General overview without route details."),
        Chunk("c2", "demo", "api.py", "python", "code", 10, 14, "@app.get('/health')\ndef health():\n    return {'ok': True}"),
    ]
    store.build("demo", chunks)

    context = HybridRetriever(store).retrieve(
        "demo",
        "Where is the health route declared?",
        {
            "facts": ["api.py declares route app.get('/health') handled near lines 10-14"],
            "nodes": [{"id": "file:api.py", "kind": "file", "label": "api.py"}],
            "edges": [{"source": "file:api.py", "target": "route:api.py:app.get('/health')", "relation": "declares_route"}],
        },
    )

    assert "api.py" in context["relevant_files"]
    assert context["graph_paths"]
    assert any(chunk["file_path"] == "api.py" for chunk in context["chunks"])


def test_query_intent_classifier_uses_product_intents(tmp_path):
    retriever = HybridRetriever(VectorStore(tmp_path))

    assert retriever.classify("how do I run this repo?") == ["run_instructions"]
    assert retriever.classify("how do I install this in Linux?") == ["run_instructions"]
    assert "weakness_question" in retriever.classify("what are the weaknesses?")


def test_retrieval_context_pack_has_clean_public_fields(tmp_path):
    store = VectorStore(tmp_path)
    store.build("demo", [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "A useful README.")])

    context = HybridRetriever(store).retrieve(
        "demo",
        "explain this project",
        {"facts": [], "nodes": [], "edges": []},
        {"readme_paths": ["README.md"], "important_files": [{"path": "README.md", "reason": "docs"}]},
    )

    assert "repo_facts" in context
    assert "selected_chunks" in context
    assert "source_references" in context


def test_run_retrieval_prioritizes_install_doc_over_helper_scripts(tmp_path):
    store = VectorStore(tmp_path)
    store.build(
        "demo",
        [
            Chunk("c1", "demo", "INSTALL.md", "markdown", "docs", 1, 8, "## Install\n```bash\nsudo apt-get install demo\n```"),
            Chunk("c2", "demo", "benchmarks/smoke_test.sh", "shell", "code", 1, 3, "bash benchmarks/smoke_test.sh"),
        ],
    )

    context = HybridRetriever(store).retrieve(
        "demo",
        "how do I install this in Linux?",
        {"facts": [], "nodes": [], "edges": []},
        {
            "install_doc_paths": ["INSTALL.md"],
            "official_doc_paths": ["INSTALL.md", "README.md"],
            "readme_paths": ["README.md"],
            "helper_scripts": [{"path": "benchmarks/smoke_test.sh", "mentioned_in_official_docs": False}],
        },
    )

    assert context["selected_chunks"][0]["file_path"] == "INSTALL.md"
    assert "benchmarks/smoke_test.sh" not in context["relevant_files"]
