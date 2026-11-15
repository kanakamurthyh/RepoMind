"""Create a SHA-pinned, explicitly draft 160-question corpus from local checkouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from app.ingestion.git_client import git_command, git_env
from evals.corpus_seeds import REPOSITORIES


def span(root: Path, path: str, anchor: str | None = None, lines: int = 30) -> dict[str, Any]:
    source = root / path
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    rows = text.splitlines()
    start = 1
    if anchor is not None:
        # Code anchors in this corpus are literal, selected before retrieval is run.
        match = next((i for i, line in enumerate(rows, 1) if anchor in line), None)
        if match is None:
            raise ValueError(f"Anchor {anchor!r} not found in {source}")
        start = match
    return {"file_path": path, "start_line": start, "end_line": min(len(rows), start + lines - 1),
            "file_sha256": hashlib.sha256(raw).hexdigest()}


def prepare(repos_root: Path) -> dict[str, Any]:
    repositories, questions = [], []
    for seed in REPOSITORIES:
        root = repos_root / seed["name"].replace("/", "--")
        sha = subprocess.run(git_command("rev-parse", "HEAD"), cwd=root, env=git_env(), capture_output=True, text=True, check=True).stdout.strip()
        license_path = root / seed["license"]
        license_sha = hashlib.sha256(license_path.read_bytes()).hexdigest()
        repositories.append({key: seed[key] for key in ("name", "split", "language")}
                            | {"url": f"https://github.com/{seed['name']}", "commit_sha": sha,
                               "license_path": seed["license"], "license_sha256": license_sha})
        local = []
        for question, anchor, answer in seed["features"]:
            primary = span(root, seed["primary"], anchor)
            local.append(primary)
            questions.append({"category": "local_behavior", "question": question, "gold_answer": answer,
                              "answerable": True, "evidence_sets": [[primary]]})
        for feature, primary in zip(seed["features"], local, strict=True):
            questions.append({"category": "cross_file", "question": f"Which implementation and test files should be read together to verify this behavior: {feature[0]}",
                              "gold_answer": "Explain the implementation and independently check matching test cases; the draft test span must be narrowed by a reviewer.",
                              "answerable": True, "evidence_sets": [[primary, span(root, seed["tests"], lines=60)]]})
        for question, paths in [
            ("What public functionality does this package offer, and which source module implements its core behavior?", [seed["readme"], seed["primary"]]),
            ("How does the main implementation relate to the secondary module or type definitions?", [seed["primary"], seed["secondary"]]),
            ("How is the package organized for consumers versus its implementation source?", [seed["manifest"], seed["primary"]]),
        ]:
            questions.append({"category": "architecture", "question": question,
                              "gold_answer": "Identify the package boundary and explain only relationships established by these files. Reviewer must finalize the answer.",
                              "answerable": True, "evidence_sets": [[span(root, p, lines=50) for p in dict.fromkeys(paths)]]})
        for question, path in [
            ("What installation and basic usage instructions are provided for a new consumer?", seed["readme"]),
            ("Which dependency or runtime requirements must a contributor inspect before running this package?", seed["manifest"]),
            ("Where are the commands or tooling settings for running the test suite declared?", seed["manifest"]),
        ]:
            questions.append({"category": "run_docs", "question": question, "gold_answer": "Report declared instructions and requirements; do not claim the commands were executed. Reviewer must finalize the answer.",
                              "answerable": True, "evidence_sets": [[span(root, path, lines=80)]]})
        for question, paths in [
            ("Which test file supplies evidence for the core behavior, and which implementation does it exercise?", [seed["tests"], seed["primary"]]),
            ("What license terms should a downstream maintainer inspect before reuse?", [seed["license"]]),
        ]:
            questions.append({"category": "risk_testing", "question": question, "gold_answer": "Identify the relevant test or license evidence without claiming runtime correctness or legal interpretation.",
                              "answerable": True, "evidence_sets": [[span(root, p, lines=40) for p in dict.fromkeys(paths)]]})
        for question in [
            "Where is the bundled Kubernetes Helm chart that configures horizontal autoscaling?",
            "How does the built-in PostgreSQL migration system version customer audit tables?",
            "Which built-in SAML identity provider manages organization single sign-on?",
            "Where does the bundled Terraform module provision an AWS production network?",
        ]:
            questions.append({"category": "unsupported", "question": question, "gold_answer": "No supporting implementation was identified during corpus preparation. Abstain and distinguish indexed absence from proven impossibility; reviewer must confirm absence.",
                              "answerable": False, "evidence_sets": []})
        for index, question in enumerate(questions[-20:], 1):
            question.update(id=f"{seed['name']}:{index:02}", repo=seed["name"], commit_sha=sha,
                            review_status="draft", reviewers=[], limitations="Prepared by the implementation author; not independent human gold. Some spans are broad candidates.")
    return {"schema_version": 1, "status": "draft_not_a_release_gate", "repositories": repositories, "questions": questions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-root", type=Path, default=Path("data/evals/repos"))
    parser.add_argument("--output", type=Path, default=Path("data/evals/corpus.json"))
    args = parser.parse_args()
    if args.output.exists():
        prior = json.loads(args.output.read_text(encoding="utf-8"))
        if any(q.get("review_status") == "approved" for q in prior.get("questions", [])):
            raise ValueError("Refusing to overwrite reviewed annotations. Choose a new output path for a new corpus revision.")
    corpus = prepare(args.repos_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, indent=2), encoding="utf-8")
    print(f"Prepared {len(corpus['questions'])} draft questions on {len(corpus['repositories'])} SHA-pinned repositories: {args.output}")


if __name__ == "__main__":
    main()
