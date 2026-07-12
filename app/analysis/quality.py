from __future__ import annotations

from app.models import Weakness


class QualityScorer:
    def score(self, facts: dict, weaknesses: list[Weakness]) -> dict:
        security_issues = [w for w in weaknesses if w.category == "security"]
        scores = {
            "documentation": self._documentation(facts),
            "test_coverage_signals": self._tests(facts),
            "project_structure": self._structure(facts),
            "run_deployment_readiness": self._run_readiness(facts),
        }
        total = sum(item["score"] for item in scores.values())
        return {
            "total": total,
            "categories": scores,
            "security_review": self._security_review(security_issues),
            "top_improvements": self._top_improvements(facts, weaknesses)[:5],
        }

    def _documentation(self, facts: dict) -> dict:
        score = 0
        reasons = []
        if facts.get("has_readme"):
            score += 8
            reasons.append("README present")
        if facts.get("has_run_instructions"):
            score += 6
            reasons.append("run instructions detected")
        if facts.get("docs_dirs"):
            score += 3
            reasons.append("docs folder present")
        if facts.get("important_files"):
            score += 3
            reasons.append("important files identifiable")
        return {"score": min(25, round(score * 1.25)), "reason": ", ".join(reasons) if reasons else "README/run documentation is weak or missing."}

    def _tests(self, facts: dict) -> dict:
        score = 12 if facts.get("has_tests") else 0
        if facts.get("test_dirs"):
            score += 5
        if any("coverage" in path.lower() for path in facts.get("file_tree", [])):
            score += 3
        reason = "Test files or test directories detected." if score else "No test coverage signals detected."
        return {"score": min(25, round(score * 1.25)), "reason": reason}

    def _structure(self, facts: dict) -> dict:
        score = 6
        if facts.get("source_dirs"):
            score += 6
        if facts.get("package_files"):
            score += 4
        if facts.get("main_entrypoints") or facts.get("important_modules"):
            score += 4
        return {"score": min(25, round(score * 1.25)), "reason": "Structure signals include source dirs, package files, and identifiable entry modules."}

    def _run_readiness(self, facts: dict) -> dict:
        score = 0
        if facts.get("has_run_instructions"):
            score += 8
        if facts.get("dependency_files"):
            score += 5
        if facts.get("deployment_files"):
            score += 4
        if facts.get("scripts"):
            score += 3
        return {"score": min(25, round(score * 1.25)), "reason": "Run/deploy signals are derived from setup files, scripts, and container files."}

    def _security_review(self, security_issues: list[Weakness]) -> dict:
        severity_order = {"high": 3, "medium": 2, "low": 1}
        highest = max((item.severity for item in security_issues), key=lambda value: severity_order.get(value, 0), default="none")
        return {
            "label": "automated static review findings",
            "finding_count": len(security_issues),
            "highest_severity": highest,
            "confidence_summary": {
                "high": sum(1 for item in security_issues if item.confidence == "high"),
                "medium": sum(1 for item in security_issues if item.confidence == "medium"),
                "low": sum(1 for item in security_issues if item.confidence == "low"),
            },
            "note": "These findings are static signals, not an audit-level guarantee.",
        }

    def _top_improvements(self, facts: dict, weaknesses: list[Weakness]) -> list[str]:
        improvements = [f"{w.title}: {w.suggested_fix}" for w in weaknesses if w.severity in {"high", "medium"}]
        if not facts.get("has_run_instructions"):
            improvements.append("Document run commands in README with expected environment variables.")
        if not facts.get("has_tests"):
            improvements.append("Add smoke tests or unit tests for the main workflow.")
        return improvements
