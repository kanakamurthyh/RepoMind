"""Git-history intelligence: churn hotspots, ownership concentration, bus factor."""

from __future__ import annotations

import logging
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.ingestion.git_client import git_command, git_env

logger = logging.getLogger(__name__)
COMMIT_MARK = "\x1ecommit\x1f"


class HistoryAnalyzer:
    def __init__(self, max_commits: int = 500, timeout: int = 60) -> None:
        self.max_commits = max_commits
        self.timeout = timeout

    def analyze(self, repo_path: Path, line_counts: dict[str, int] | None = None, complexity_by_file: dict[str, int] | None = None) -> dict[str, Any]:
        line_counts = line_counts or {}
        complexity_by_file = complexity_by_file or {}
        try:
            output = subprocess.run(
                git_command("log", f"-n{self.max_commits}", "--no-merges", "--numstat", f"--format={COMMIT_MARK}%H%x1f%an%x1f%aI"),
                cwd=repo_path, capture_output=True, text=True, timeout=self.timeout, check=True, env=git_env(),
            ).stdout
            shallow = subprocess.run(
                git_command("rev-parse", "--is-shallow-repository"), cwd=repo_path, capture_output=True, text=True, timeout=15, env=git_env(),
            ).stdout.strip() == "true"
        except (subprocess.SubprocessError, OSError) as exc:
            logger.info("git history unavailable for %s: %s", repo_path, exc)
            return {"status": "unavailable", "commits_analyzed": 0, "hotspots": [], "top_contributors": [], "bus_factor": None}

        commits = 0
        author_commits: Counter[str] = Counter()
        file_commits: Counter[str] = Counter()
        file_churn: Counter[str] = Counter()
        file_authors: dict[str, Counter[str]] = defaultdict(Counter)
        last_touched: dict[str, str] = {}
        author = ""
        date = ""
        for line in output.split("\n"):  # not splitlines(): it treats \x1e/\x1f-style separators as breaks
            if line.startswith(COMMIT_MARK):
                parts = line[len(COMMIT_MARK):].split("\x1f")
                author = parts[1] if len(parts) > 1 else "unknown"
                date = parts[2] if len(parts) > 2 else ""
                commits += 1
                author_commits[author] += 1
                continue
            fields = line.split("\t")
            if len(fields) != 3:
                continue
            added, removed, path = fields
            if "=>" in path:
                continue
            churn = (int(added) if added.isdigit() else 0) + (int(removed) if removed.isdigit() else 0)
            file_commits[path] += 1
            file_churn[path] += churn
            file_authors[path][author] += 1
            last_touched.setdefault(path, date)

        hotspots = []
        for path, count in file_commits.items():
            if path not in line_counts:
                continue  # deleted/ignored files
            size = line_counts.get(path, 1)
            complexity = complexity_by_file.get(path, 0)
            score = count * math.log1p(size) * (1 + min(complexity, 50) / 25)
            owners = file_authors[path]
            top_share = owners.most_common(1)[0][1] / sum(owners.values()) if owners else 0
            hotspots.append({
                "path": path, "commits": count, "churn": file_churn[path], "lines": size, "max_complexity": complexity,
                "authors": len(owners), "top_author_share": round(top_share, 2), "last_changed": last_touched.get(path), "score": round(score, 2),
            })
        hotspots.sort(key=lambda item: item["score"], reverse=True)

        total = sum(author_commits.values())
        bus_factor = 0
        covered = 0
        for _, count in author_commits.most_common():
            if total and covered / total >= 0.5:
                break
            covered += count
            bus_factor += 1
        return {
            "status": "ok",
            "commits_analyzed": commits,
            "is_shallow": shallow,
            "note": "History is shallow; churn reflects only the fetched commits. Set GIT_HISTORY_DEPTH higher for deeper analysis." if shallow else "",
            "hotspots": hotspots[:15],
            "top_contributors": [{"name": name, "commits": count} for name, count in author_commits.most_common(8)],
            "contributors": len(author_commits),
            "bus_factor": bus_factor if commits else None,
        }
