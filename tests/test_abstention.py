"""Evidence strength and abstention: RepoMind should say so when the repo lacks the answer."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.agent.detective import RepoDetective
from app.config import Settings
from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.models import Chunk
from app.rag.evidence import Vocabulary, content_term_groups
from app.rag.llm import LLMResponse
from app.rag.retriever import HybridRetriever
from app.rag.vector_store import VectorStore
from evals.run_eval import LocalGitClient

TINY_REPO = {
    "README.md": "# Billing\n\nA tiny invoicing library.\n\n## Usage\n\n```bash\npython -m billing.cli\n```\n",
    "billing/__init__.py": '"""Billing package."""\n',
    "billing/invoice.py": (
        "from billing.tax import sales_tax\n\n\n"
        "def compute_invoice_total(lines):\n"
        '    """Sum invoice line amounts and add sales tax."""\n'
        "    subtotal = sum(line['amount'] for line in lines)\n"
        "    return subtotal + sales_tax(subtotal)\n"
    ),
    "billing/tax.py": (
        "TAX_RATE = 0.08\n\n\n"
        "def sales_tax(amount):\n"
        '    """Calculate sales tax for an amount."""\n'
        "    return round(amount * TAX_RATE, 2)\n"
    ),
    "billing/cli.py": "from billing.invoice import compute_invoice_total\n\nif __name__ == '__main__':\n    print(compute_invoice_total([{'amount': 10}]))\n",
    "tests/test_invoice.py": "from billing.invoice import compute_invoice_total\n\n\ndef test_total():\n    assert compute_invoice_total([{'amount': 100}]) == 108\n",
}
ABSENT_QUESTION = "How does the Kubernetes Helm chart configure autoscaling?"


@pytest.fixture(scope="module")
def tiny_env(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    repo = root / "repo"
    for relative, content in TINY_REPO.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    settings = replace(
        Settings(), data_dir=root / "data", osv_scan_enabled=False, github_api_checks_enabled=False, llm_provider="off",
        embedding_provider="local", vector_store_provider="local", reranker_provider="local", database_url="",
    )
    analyzer = RepoAnalyzer(settings, git_client=LocalGitClient(repo))
    repo_id = analyzer.analyze("https://github.com/example/billing").repo_id
    return analyzer, settings, repo_id


def _chat(tiny_env, llm=None):
    analyzer, settings, _ = tiny_env
    return ChatService(analyzer, settings, llm_factory=lambda s: (llm, "LLM calls disabled"))


def test_absent_topic_has_no_evidence_and_deterministic_answer_abstains(tiny_env):
    repo_id = tiny_env[2]
    response = _chat(tiny_env).answer(repo_id, ABSENT_QUESTION)

    assert response["evidence_strength"] == "none"
    evidence = response["evidence"]
    assert {"kubernetes", "helm", "autoscaling"} <= set(evidence["absent_from_repo"])
    assert "configure" not in evidence["missing_terms"]  # intent/scaffolding words are not key terms
    answer = response["answer"]
    assert "does not appear to contain evidence" in answer
    assert "Possibly related" in answer
    assert "The most relevant code is" not in answer
    assert "kubernetes" in answer.split("Notes / uncertainty:")[1]


def test_real_symbol_question_has_strong_evidence(tiny_env):
    repo_id = tiny_env[2]
    response = _chat(tiny_env).answer(repo_id, "What does compute_invoice_total do?")

    assert response["evidence_strength"] == "strong"
    assert response["evidence"]["basis"] == "identifier_match"
    assert "billing/invoice.py" in response["answer"]
    assert "does not appear" not in response["answer"]


def test_natural_language_question_with_matching_terms_is_strong(tiny_env):
    context = _chat(tiny_env).retrieve(tiny_env[2], "How is sales tax calculated?")

    assert context["evidence_strength"] == "strong"
    assert context["evidence"]["basis"] == "term_coverage"
    assert context["evidence"]["coverage"] >= 0.6


def test_partially_covered_question_is_weak_and_names_missing_terms(tiny_env):
    response = _chat(tiny_env).answer(tiny_env[2], "How are invoice totals synced to Salesforce?")

    assert response["evidence_strength"] == "weak"
    assert "salesforce" in response["evidence"]["absent_from_repo"]
    assert "only weak evidence" in response["answer"]


def test_repo_level_question_without_key_terms_is_not_abstained(tiny_env):
    response = _chat(tiny_env).answer(tiny_env[2], "how do I run this?")

    assert response["evidence_strength"] == "strong"
    assert response["evidence"]["basis"] == "repo_level"
    assert "Evidence check" not in response["answer"]


def test_run_question_about_absent_platform_keeps_run_answer_with_caveat(tiny_env):
    response = _chat(tiny_env).answer(tiny_env[2], "How do I deploy this to Kubernetes?")

    assert response["evidence_strength"] == "none"
    assert response["answer"].startswith("> Evidence check (none)")
    assert "Direct answer:" in response["answer"]


def test_llm_prompt_warns_about_missing_evidence_only_when_weak(tiny_env):
    prompts: list[str] = []

    class FakeLLM:
        name = "fake"
        model = "fake-1"

        def complete(self, prompt, system=None, json_mode=False, max_tokens=1400):
            prompts.append(prompt)
            return LLMResponse("Direct answer:\nThe repository does not contain this.", "fake", "fake-1")

    chat = _chat(tiny_env, FakeLLM())
    absent = chat.answer(tiny_env[2], ABSENT_QUESTION)
    chat.answer(tiny_env[2], "What does compute_invoice_total do?")

    assert absent["evidence_strength"] == "none"
    assert '<evidence_check level="none">' in prompts[0]
    assert "Do not guess" in prompts[0] and "kubernetes" in prompts[0]
    assert "<evidence_check" not in prompts[1]


def test_detective_deterministic_answer_marks_weak_matches_as_possibly_related(tiny_env):
    analyzer, _settings, repo_id = tiny_env
    result = RepoDetective(analyzer, _chat(tiny_env)).investigate(repo_id, ABSENT_QUESTION)

    assert result["mode"] == "deterministic"
    assert "does not appear to contain strong evidence" in result["answer"]
    assert "Relevant:" not in result["answer"]


def test_cross_encoder_relevance_can_establish_strong_evidence(tmp_path):
    store = VectorStore(tmp_path)
    store.build("demo", [Chunk("c1", "demo", "auth/login.py", "python", "code", 1, 4, "def check_credentials(user, secret):\n    return verify(user, secret)")])

    class SemanticReranker:
        def rerank(self, query, chunks, intents):
            ranked = []
            for chunk in chunks:
                clone = Chunk.from_dict(chunk.to_dict())
                clone.metadata = {**clone.metadata, "rerank_relevance": 0.93}
                ranked.append(clone)
            return ranked

    question = "Where do we validate sign-in?"
    plain = HybridRetriever(store).retrieve("demo", question, {})
    semantic = HybridRetriever(store, SemanticReranker()).retrieve("demo", question, {})

    assert plain["evidence_strength"] in {"weak", "none"}
    assert semantic["evidence_strength"] == "strong"
    assert semantic["evidence"]["basis"] == "rerank_relevance"


def test_content_terms_drop_scaffolding_and_match_abbreviations():
    groups = {group.label: group for group in content_term_groups("How does the Kubernetes Helm chart configure autoscaling?")}
    assert set(groups) == {"kubernetes", "helm", "chart", "autoscaling"}

    auth = content_term_groups("Where is authentication handled?")
    assert [group.label for group in auth] == ["authentication"]
    assert auth[0].matched_by(Vocabulary({"auth", "login"}))
    assert not groups["autoscaling"].matched_by(Vocabulary({"auto", "scale"}))
    assert content_term_groups("explain HelmChart")[0].matched_by(Vocabulary({"helm", "chart"}))
