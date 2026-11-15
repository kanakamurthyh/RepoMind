from __future__ import annotations

from app.config import Settings
from app.core.chat import ChatService
from app.models import AnalysisResult, Chunk
from app.rag.embeddings import LocalHashingEmbedder
from app.rag.llm import LLMError, LLMResponse
from app.rag.vector_store import VectorStore


def _analyzer(tmp_path, chunks, summary):
    settings = Settings(data_dir=tmp_path)
    store = VectorStore(tmp_path / "indexes", LocalHashingEmbedder(), app_settings=settings)
    store.build("demo", chunks)

    class Metadata:
        def __init__(self):
            self.chats = []

        def record_chat(self, repo_id, question, answer, citations, meta=None):
            self.chats.append((question, answer))

    class Analyzer:
        vector_store = store
        metadata = Metadata()

        def load(self, repo_id):
            return AnalysisResult(repo_id, "https://github.com/example/demo", summary, [], {"facts": [], "nodes": [], "edges": []}, [], ["completed"])

    return Analyzer()


RUN_SUMMARY = {
    "project_identity": {"one_line_summary": "Demo project."},
    "repo_facts": {
        "run_instructions": [{"command": "python app.py", "source": "README.md", "start_line": 1, "end_line": 3, "official": True, "confidence": "high"}],
        "readme_paths": ["README.md"],
        "official_doc_paths": ["README.md"],
    },
}


def test_local_mode_answer_is_deterministic_with_citations(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Run with python app.py.")], RUN_SUMMARY)

    response = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (None, "LLM calls disabled")).answer("demo", "how do I run this?")

    assert "python app.py" in response["answer"]
    assert response["llm"] == {"provider": "local", "used": False, "reason": "LLM calls disabled"}
    assert response["citations"][0]["file_path"] == "README.md"
    assert "context" not in response
    assert response["evidence_strength"] == "strong"
    assert response["evidence"]["level"] == "strong"


def test_llm_answer_is_grounded_and_invalid_citations_are_flagged(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Run with python app.py.")], RUN_SUMMARY)
    prompts = []

    class FakeLLM:
        name = "fake"
        model = "fake-1"

        def complete(self, prompt, system=None, json_mode=False, max_tokens=1400):
            prompts.append((system, prompt))
            return LLMResponse("Direct answer:\nRun `python app.py` [S1]. Also see [S9] and README.md:1-3.", "fake", "fake-1")

    response = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (FakeLLM(), "")).answer("demo", "how do I run this?")

    assert response["llm"]["used"] is True
    assert response["grounding"]["cited_sources"] == ["S1"]
    assert response["grounding"]["invalid_source_ids"] == ["S9"]
    assert "S9⚠unverified" in response["answer"]
    assert response["citations"][0]["cited"] is True
    system, user = prompts[0]
    assert "untrusted data" in system and '<source id="S1"' in user


def test_llm_failure_falls_back_without_leaking_details(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Run with python app.py.")], RUN_SUMMARY)

    class BrokenLLM:
        name = "gemini"
        model = "m"

        def complete(self, *args, **kwargs):
            raise LLMError("network", "https://host/?key=AIzaSECRETSECRETSECRETSECRET123")

    response = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (BrokenLLM(), "")).answer("demo", "how do I run this?")

    assert response["llm"]["used"] is False
    assert response["llm"]["error_code"] == "network"
    assert "AIza" not in str(response)
    assert "python app.py" in response["answer"]


def test_answers_are_cached_per_question(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Demo README.")], RUN_SUMMARY)
    service = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (None, "off"))

    first = service.answer("demo", "What is this project?")
    second = service.answer("demo", "What is this project?")

    assert "cached" not in first and second.get("cached") is True


def test_cache_preserves_case_and_all_effective_history_and_skips_repeat_retrieval(tmp_path, monkeypatch):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 1, "Demo README.")], RUN_SUMMARY)
    service = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (None, "off"))
    retrieve = service.retrieve
    calls = []
    def tracked(*args, **kwargs):
        calls.append(args)
        return retrieve(*args, **kwargs)
    monkeypatch.setattr(service, "retrieve", tracked)
    service.answer("demo", "What is Foo?")
    service.answer("demo", "What is foo?")
    history = [{"role": "user", "content": "old Foo"}, {"role": "assistant", "content": "previous"},
               {"role": "user", "content": "latest"}]
    service.answer("demo", "Explain it", history)
    changed = [{**history[0], "content": "different old Foo"}, *history[1:]]
    service.answer("demo", "Explain it", changed)
    assert len(calls) == 4
    assert service.answer("demo", "Explain it", changed)["cached"]
    assert len(calls) == 4


def test_follow_up_question_is_rewritten_with_history(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "pay.py", "python", "code", 1, 3, "def charge_card(): retry()")], {"project_identity": {}, "repo_facts": {}})
    service = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (None, "off"))

    context = service.retrieve("demo", "and what calls it?", history=[{"role": "user", "content": "explain charge_card"}])

    assert "charge_card" in context["search_query"]


def test_local_mode_abstains_when_repo_lacks_the_topic(tmp_path):
    analyzer = _analyzer(tmp_path, [Chunk("c1", "demo", "pay.py", "python", "code", 1, 3, "def charge_card(): retry()")], {"project_identity": {}, "repo_facts": {}})
    service = ChatService(analyzer, Settings(data_dir=tmp_path), llm_factory=lambda s: (None, "off"))

    response = service.answer("demo", "How is the GraphQL schema federated?")

    assert response["evidence_strength"] == "none"
    assert "does not appear to contain evidence" in response["answer"]
    assert "Possibly related" in response["answer"]
