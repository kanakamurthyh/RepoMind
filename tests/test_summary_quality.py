from __future__ import annotations

from app.analysis.repo_facts import RepoFactsCollector
from app.analysis.summary import SummaryGenerator
from app.graph.code_graph import CodeGraphBuilder
from app.ingestion.scanner import FileScanner


def test_summary_entry_flow_and_module_reasons_are_specific(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text(
        "# Solar Sense\n\n"
        "## Setup\n"
        "```bash\n"
        "pip install -r requirements.txt\n"
        "streamlit run app/app.py\n"
        "python src/evaluate.py\n"
        "```\n",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("streamlit\nscikit-learn\n", encoding="utf-8")
    (repo / "app").mkdir()
    (repo / "app" / "app.py").write_text("import streamlit as st\nfrom src.model import predict\nst.title('Solar')\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "model.py").write_text("from sklearn.linear_model import LinearRegression\ndef train(): pass\ndef predict(): pass\n", encoding="utf-8")
    (repo / "src" / "utils.py").write_text("def preprocess(data): return data\n", encoding="utf-8")
    (repo / "src" / "evaluate.py").write_text("from src.model import predict\ndef evaluate(): return predict()\n", encoding="utf-8")

    records = FileScanner(max_file_size_bytes=10000).scan("demo", repo)
    graph = CodeGraphBuilder().build(records)
    facts = RepoFactsCollector().collect("https://github.com/example/solar-sense", records, graph)
    summary = SummaryGenerator().generate("https://github.com/example/solar-sense", records, graph, [], facts)

    reasons = [item["reason"] for item in facts["important_modules"]]
    assert not any(reason == "Contains many definitions/import relationships." for reason in reasons)
    assert any("model or ML" in reason for reason in reasons)
    assert any("evaluates outputs" in reason for reason in reasons)
    assert any("Start from `app/app.py`" in step for step in summary["architecture"]["entrypoint_flow"])
    assert any("Install dependencies" in step for step in summary["architecture"]["entrypoint_flow"])


def _summarize(tmp_path, files: dict[str, str]) -> dict:
    repo = tmp_path / "repo"
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text, encoding="utf-8")
    records = FileScanner(max_file_size_bytes=10000).scan("demo", repo)
    graph = CodeGraphBuilder().build(records)
    facts = RepoFactsCollector().collect("https://github.com/example/demo", records, graph)
    return SummaryGenerator().generate("https://github.com/example/demo", records, graph, [], facts)


def test_about_uses_readme_purpose_and_feature_list(tmp_path):
    summary = _summarize(tmp_path, {
        "README.md": (
            "# Ledger\n\n"
            "Ledger is a small tool that reconciles bank exports against an accounting journal so bookkeepers can close the month faster.\n\n"
            "## Features\n\n"
            "| Capability | How |\n"
            "| --- | --- |\n"
            "| **CSV import** | Parses bank exports |\n\n"
            "- Fuzzy matching of transactions by amount and date\n"
            "```bash\n- not a feature\n```\n\n"
            "## Setup\n\n- pip install ledger\n"
        ),
        "ledger/main.py": "def main():\n    pass\n",
    })
    about = summary["project_identity"]["about"]

    assert about["why_it_exists"].startswith("Ledger is a small tool")
    assert about["why_it_exists_source"] == "README.md"
    assert about["what_it_does_basis"] == "readme"
    assert [item["text"] for item in about["what_it_does"]] == ["CSV import — Parses bank exports", "Fuzzy matching of transactions by amount and date"]
    assert about["what_it_does"][0]["source"] == "README.md:9"


def test_about_falls_back_to_inferred_capabilities_without_readme(tmp_path):
    summary = _summarize(tmp_path, {
        "requirements.txt": "fastapi\n",
        "api/server.py": "from fastapi import FastAPI\napp = FastAPI()\n\n@app.get('/items')\ndef items():\n    return []\n",
        "tests/test_server.py": "def test_ok():\n    assert True\n",
    })
    about = summary["project_identity"]["about"]

    assert about["why_it_exists"] == ""
    assert about["why_it_exists_source"] is None
    assert about["what_it_does_basis"] == "inferred"
    texts = [item["text"] for item in about["what_it_does"]]
    assert any("HTTP API" in text and "api/server.py" in text for text in texts)
    assert any("test suite" in text for text in texts)
    assert all(item["source"] == "inferred from repository structure" for item in about["what_it_does"])
