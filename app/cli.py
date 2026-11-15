"""Installable command-line interface for repository analysis and evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.config import Settings
from app.core.chat import ChatService
from app.core.repo_analyzer import RepoAnalyzer
from app.ingestion.git_client import GitClient
from app.utils.paths import repo_id_from_url
from app.utils.redaction import redact_data, scrub


class LocalRepository:
    source_kind = "local"

    def __init__(self, path: Path) -> None:
        self.path = path.resolve(strict=True)
        if not self.path.is_dir():
            raise ValueError("Local repository must be a directory")
        name = re.sub(r"[^a-z0-9_.-]", "-", self.path.name.lower()) or "repository"
        suffix = hashlib.sha256(str(self.path).encode()).hexdigest()[:10]
        self.url = f"https://github.com/repomind-local/{name}-{suffix}"

    def clone_or_get(self, repo_url: str, refresh: bool = False):
        return repo_id_from_url(self.url), self.path, False

    def head_sha(self, path: Path) -> str | None:
        return GitClient.head_sha(self, path)

    def path_for(self, repo_id: str) -> Path:
        return self.path

    def delete(self, repo_id: str) -> None:
        # A user's local directory is never an owned cache artifact.
        return None


def sarif(result: dict[str, Any]) -> dict[str, Any]:
    findings = result.get("weaknesses", [])
    rules = sorted({f.get("category", "maintainability") for f in findings})
    rows = []
    for finding in findings:
        row: dict[str, Any] = {
            "ruleId": finding.get("category", "maintainability"),
            "level": "error" if finding["severity"] in {"critical", "high"} else "warning" if finding["severity"] == "medium" else "note",
            "message": {"text": f"{finding['title']}: {finding.get('suggested_fix', '')}"},
        }
        if finding.get("file_path"):
            location: dict[str, Any] = {"artifactLocation": {"uri": quote(finding["file_path"], safe="/")}}
            if finding.get("line_number"):
                location["region"] = {"startLine": finding["line_number"]}
            row["locations"] = [{"physicalLocation": location}]
        rows.append(row)
    return {"version": "2.1.0", "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "runs": [{"tool": {"driver": {"name": "RepoMind", "rules": [{"id": r} for r in rules]}},
                      "results": rows, "properties": {"repo_id": result["repo_id"], "commit_sha": result.get("commit_sha")}}]}


def markdown(result: dict[str, Any]) -> str:
    if "answer" in result:
        return result["answer"]
    summary = result.get("summary", {})
    identity = summary.get("project_identity", {})
    lines = [f"# {identity.get('project_name', result.get('repo_id', 'RepoMind'))}", "",
             identity.get("what_this_repo_is_for", ""), "", f"Commit: {result.get('commit_sha') or 'unversioned local source'}",
             f"Run: {summary.get('ingestion', {}).get('run_id', 'legacy')}"]
    lines.extend(f"- **{w['severity'].upper()}** {w['title']}: {w['suggested_fix']}" for w in result.get("weaknesses", []))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repository intelligence with inspectable source evidence")
    parser.add_argument("--data-dir", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("analyze", "check"):
        item = sub.add_parser(command)
        item.add_argument("target", help="Public GitHub URL or local directory")
        item.add_argument("--force", action="store_true")
        item.add_argument("--format", choices=["json", "md", "sarif"] if command == "check" else ["json", "md"], default="json")
        if command == "check":
            item.add_argument("--fail-on", choices=["critical", "high", "medium", "low", "none"], default="high")
    for command in ("ask", "search"):
        item = sub.add_parser(command)
        item.add_argument("repo_id")
        item.add_argument("question")
        item.add_argument("--format", choices=["json", "md"], default="json")
    sub.add_parser("mcp")
    args = parser.parse_args(argv)
    config = Settings()
    if args.data_dir:
        from dataclasses import replace
        config = replace(config, data_dir=args.data_dir)
    analyzer = None
    try:
        if args.command == "mcp":
            from app.api.services import ServiceContainer
            from app.mcp_server import build_server
            build_server(ServiceContainer(config)).run()
            return 0
        local = LocalRepository(Path(args.target)) if hasattr(args, "target") and Path(args.target).is_dir() else None
        analyzer = RepoAnalyzer(config, git_client=local)
        if args.command in {"analyze", "check"}:
            result = analyzer.analyze(local.url if local else args.target, force=args.force).to_dict(include_chunks=False)
        elif args.command == "ask":
            result = ChatService(analyzer, config).answer(args.repo_id, args.question)
        else:
            context = ChatService(analyzer, config).retrieve(args.repo_id, args.question)
            result = {"results": context["source_references"], "evidence": context["evidence"], "snapshot": context["snapshot"]}
        result = redact_data(result)
        print(json.dumps(sarif(result) if args.format == "sarif" else result, indent=2) if args.format != "md" else markdown(result))
        if args.command == "check" and args.fail_on != "none":
            order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            return int(any(order.get(w["severity"], 99) <= order[args.fail_on] for w in result["weaknesses"]))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(scrub(str(exc), config.gemini_api_key, config.openai_api_key, config.github_token), file=sys.stderr)
        return 2
    finally:
        if analyzer is not None:
            analyzer.close()


if __name__ == "__main__":
    raise SystemExit(main())
