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
