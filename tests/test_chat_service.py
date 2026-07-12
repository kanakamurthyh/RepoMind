from __future__ import annotations

from app.config import Settings
from app.core.chat import ChatService
from app.models import AnalysisResult, Chunk
from app.rag.embeddings import LocalHashingEmbedder
from app.rag.vector_store import LocalJsonVectorStore


def test_chat_service_public_response_does_not_expose_retrieval_context(tmp_path):
    store = LocalJsonVectorStore(tmp_path / "indexes", LocalHashingEmbedder())
    chunks = [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Demo project README.")]
    store.build("demo", chunks)

    class Metadata:
        def record_chat(self, repo_id: str, question: str, answer: str, citations: list[dict]) -> None:
            pass

    class Analyzer:
        vector_store = store
        metadata = Metadata()

        def load(self, repo_id: str) -> AnalysisResult:
            return AnalysisResult(
                repo_id=repo_id,
                repo_url="https://github.com/example/demo",
                summary={"project_identity": {"one_line_summary": "Demo project."}, "repo_facts": {}},
                weaknesses=[],
                graph={"facts": [], "nodes": [], "edges": []},
                chunks=chunks,
                statuses=["completed"],
            )

    response = ChatService(Analyzer(), Settings(data_dir=tmp_path, llm_provider="off", gemini_api_key=None)).answer("demo", "what is this?")

    assert "answer" in response
    assert "citations" in response
    assert "context" not in response


def test_chat_service_uses_gemini_when_key_is_present_even_with_old_local_provider(tmp_path, monkeypatch):
    store = LocalJsonVectorStore(tmp_path / "indexes", LocalHashingEmbedder())
    chunks = [Chunk("c1", "demo", "README.md", "markdown", "docs", 1, 3, "Run with python app.py.")]
    store.build("demo", chunks)

    class FakeGeminiClient:
        def __init__(self, api_key: str, model: str) -> None:
            self.api_key = api_key
            self.model = model

        def generate(self, prompt: str) -> str:
            return "Direct answer:\nGemini says run `python app.py`."

    class Metadata:
        def record_chat(self, repo_id: str, question: str, answer: str, citations: list[dict]) -> None:
            pass

    class Analyzer:
        vector_store = store
        metadata = Metadata()

        def load(self, repo_id: str) -> AnalysisResult:
            return AnalysisResult(
                repo_id=repo_id,
                repo_url="https://github.com/example/demo",
                summary={
                    "project_identity": {"one_line_summary": "Demo project."},
                    "repo_facts": {
                        "run_instructions": [
                            {
                                "command": "python app.py",
                                "source": "README.md",
                                "start_line": 1,
                                "end_line": 3,
                                "official": True,
                                "confidence": "high",
                            }
                        ],
                        "readme_paths": ["README.md"],
                        "official_doc_paths": ["README.md"],
                    },
                },
                weaknesses=[],
                graph={"facts": [], "nodes": [], "edges": []},
                chunks=chunks,
                statuses=["completed"],
            )

    monkeypatch.setattr("app.core.chat.GeminiClient", FakeGeminiClient)

    response = ChatService(
        Analyzer(),
        Settings(data_dir=tmp_path, llm_provider="local", gemini_api_key="test-key", gemini_model="gemini-test"),
    ).answer("demo", "how do I run this?")

    assert "Gemini says" in response["answer"]
    assert "context" not in response
