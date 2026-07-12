from __future__ import annotations

from app.analysis.repo_facts import RepoFactsCollector
from app.graph.code_graph import CodeGraphBuilder
from app.ingestion.scanner import FileScanner


def test_install_md_commands_are_official_and_scripts_are_helpers(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "README.md").write_text("# Demo\n\nSee INSTALL.md for installation.", encoding="utf-8")
    (repo / "INSTALL.md").write_text(
        "# Install\n\n"
        "## Fresh VM install\n\n"
        "```bash\n"
        "sudo apt-get update\n"
        "python3 -m pip install -r requirements.txt\n"
        "python3 app.py\n"
        "```\n",
        encoding="utf-8",
    )
    (repo / "requirements.txt").write_text("fastapi", encoding="utf-8")
    (repo / "benchmarks").mkdir()
    (repo / "benchmarks" / "smoke_test.sh").write_text("#!/usr/bin/env bash\necho smoke", encoding="utf-8")
    (repo / "devtools").mkdir()
    (repo / "devtools" / "setup-demo-vm.sh").write_text("#!/usr/bin/env bash\necho vm", encoding="utf-8")
    (repo / "app.py").write_text("print('run')", encoding="utf-8")

    records = FileScanner(max_file_size_bytes=10000).scan("demo", repo)
    graph = CodeGraphBuilder().build(records)
    facts = RepoFactsCollector().collect("https://github.com/example/demo", records, graph)

    commands = [item["command"] for item in facts["run_instructions"]]
    assert "sudo apt-get update" in commands
    assert "python3 app.py" in commands
    assert "python -m pip install -e ." not in commands
    assert not any("smoke_test.sh" in command for command in commands)
    assert not any("setup-demo-vm.sh" in command for command in commands)
    assert all(item["official"] for item in facts["run_instructions"])
    assert facts["install_doc_paths"] == ["INSTALL.md"]
    helper_paths = {item["path"] for item in facts["helper_scripts"]}
    assert "benchmarks/smoke_test.sh" in helper_paths
    assert "devtools/setup-demo-vm.sh" in helper_paths
