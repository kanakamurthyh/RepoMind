from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from app.agent.tools import MAX_OBSERVATION_CHARS, RepoTools
from app.analysis.drift import DocsDriftDetector
from app.analysis.quality import QualityScorer
from app.models import Chunk, FileRecord
from app.rag.evidence import assess_evidence
from app.rag.grounding import GroundedPromptBuilder, verify_citations

SOURCES = {"S1": {"file_path": "src/auth.py", "start_line": 10, "end_line": 20}}
TOKEN = "ghp_" + "Z7bQ9mN2vR5sT8wX3yH6jK4pL0dF1cA9"


@pytest.mark.parametrize("reference,valid", [
    ("src/auth.py:10-20", True), ("auth.py:15", True), ("src/auth.py:1-9999", False),
    ("src/auth.py:0-10", False), ("src/auth.py:20-10", False), ("src/auth.py:21", False),
])
def test_only_fully_observed_ordered_ranges_are_verified(reference, valid):
    report = verify_citations(f"The relevant implementation is `{reference}`.", SOURCES)
    assert bool(report.path_refs_verified) is valid
    if not valid:
        assert report.grounding_score == 0


def test_abbreviated_paths_must_be_unambiguous_and_location_is_not_source():
    ranges = [*SOURCES.values(), {"file_path": "tests/auth.py", "start_line": 10, "end_line": 20}]
    assert verify_citations("`auth.py:12`", {}, ranges).path_refs_unverified
    assert verify_citations("`src/auth.py:12`", {}, [{**ranges[0], "kind": "location"}]).path_refs_unverified
    assert verify_citations("`src/auth.py:12`", {}, [{"file_path": "src/auth.py", "start_line": None}]).path_refs_unverified


def test_adjacent_visible_spans_cover_a_range_but_gaps_do_not():
    spans = [{"file_path": "a.py", "start_line": 1, "end_line": 3}, {"file_path": "a.py", "start_line": 4, "end_line": 5}]
    assert verify_citations("`a.py:1-5`", {}, spans).path_refs_verified
    spans[1]["start_line"] = 5
    assert verify_citations("`a.py:1-5`", {}, spans).path_refs_unverified


def test_prompt_registers_only_complete_visible_lines_and_redacts_facts():
    text = "\n".join(f"# line {n}: " + "x" * 80 for n in range(1, 81))
    prompt = GroundedPromptBuilder().build("How to run?", {}, {
        "selected_chunks": [{"text": text, "file_path": "large.py", "start_line": 1, "end_line": 80}],
        "source_references": [{"id": "S1"}],
        "repo_facts": {"run_instructions": [{"command": f"curl -H 'Authorization: Bearer {TOKEN}'", "source": "README.md"}]},
    })
    assert TOKEN not in prompt.user
    assert "# line 80:" not in prompt.user
    assert verify_citations("`large.py:80`", prompt.source_ids).path_refs_unverified
    end = prompt.source_ids["S1"]["end_line"]
    assert f"# line {end}: " in prompt.user
    assert f"# line {end + 1}: " not in prompt.user


def test_oversize_first_line_grants_no_evidence():
    prompt = GroundedPromptBuilder().build("Explain", {}, {
        "selected_chunks": [{"text": "x" * 3000, "file_path": "min.js", "start_line": 1, "end_line": 1}],
        "source_references": [{"id": "S1"}],
    })
    assert not prompt.source_ids


@pytest.mark.parametrize("provenance", [{}, {"c1": {"mentioned": 1}}])
def test_symbol_or_filename_match_is_not_answer_support(provenance):
    chunk = Chunk("c1", "demo", "auth.py", "python", "code", 1, 2,
                  "def login(user): return user", symbol="login")
    evidence = assess_evidence("Does `login` enforce quantum-resistant encryption?", ["function_explanation"],
                               [chunk], provenance, identifiers={"login"}, absent_from_repo=["quantum", "encryption"])
    assert evidence.level in {"weak", "none"}
    assert evidence.to_dict()["answer_support"] == "unassessed"


def test_tool_previews_and_observation_limit_never_grant_unseen_lines():
    refs = [{"file_path": f"file{i}.py", "start_line": 10, "end_line": 90,
             "excerpt": "\n".join("a = 1" for _ in range(80))} for i in range(10)]
    tools = RepoTools(SimpleNamespace(), SimpleNamespace(retrieve=lambda *a, **kw: {"source_references": refs}), "demo")
    result = tools.run("search_code", {"query": "a", "k": 10})
    assert len(json.dumps(result.output, ensure_ascii=False)) <= MAX_OBSERVATION_CHARS
    assert len(result.evidence) == len(result.output)
    assert all(row["end_line"] == row["start_line"] + len(row["preview"].splitlines()) - 1 for row in result.output)
    assert verify_citations("`file0.py:90`", {}, result.evidence).path_refs_unverified


def test_detective_cannot_verify_unseen_preview_lines_or_echo_a_token():
    from app.agent.detective import RepoDetective
    from app.rag.llm import LLMResponse
    refs = [{"file_path": "source.py", "start_line": 1, "end_line": 80, "excerpt": "\n".join("x = 1" for _ in range(80))}]
    tools = RepoTools(SimpleNamespace(), SimpleNamespace(retrieve=lambda *a, **kw: {"source_references": refs}), "demo")
    replies = iter([{"action": "search_code", "args": {"query": "x"}},
                    {"final_answer": f"The code is `source.py:80`. Credential: {TOKEN}"}])
    llm = SimpleNamespace(name="fake", model="fake", complete=lambda *a, **kw: LLMResponse(json.dumps(next(replies)), "fake", "fake"))
    result = RepoDetective(SimpleNamespace(), SimpleNamespace(llm_factory=None))._llm_loop(llm, tools, "explain", 3)
    assert "source.py:80" in result["grounding"]["unverified_path_refs"]
    assert TOKEN not in result["answer"]


def test_security_scan_unknown_does_not_receive_full_marks():
    score = QualityScorer().score({}, [], {}, {"dependencies": {"scan_status": "offline"}})
    security = score["categories"]["security_hygiene"]
    assert security["score"] < 20
    assert security["status"] == "partial"
    assert score["assessed_max"] < 100 and score["possible_total"][1] > score["total"]


def test_parent_relative_and_encoded_doc_links_resolve():
    def rec(path, text):
        return FileRecord("demo", path, path, "markdown", "docs", len(text), 1, text)
    result = DocsDriftDetector().detect([
        rec("README.md", "# Demo"), rec("read me.md", "Hello"),
        rec("docs/guide.md", "[Overview](../README.md)\n[Spaces](../read%20me.md)\n[Missing](../../missing.md)"),
    ], {}, {})
    broken = [i for i in result["issues"] if i["kind"] == "broken_link"]
    assert [i["evidence"] for i in broken] == ["../../missing.md"]
