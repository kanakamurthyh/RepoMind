from __future__ import annotations

from typing import Any

from app.models import Weakness


class QualityScorer:
    """0-100 engineering-health score from measured signals (5 categories x 20 points).

    Every point is traceable to a signal listed in `reasons`; it is a hygiene/health
    indicator, not a code-correctness or security guarantee.
    """

    def score(self, facts: dict, weaknesses: list[Weakness], graph: dict[str, Any] | None = None, insights: dict[str, Any] | None = None) -> dict:
        graph = graph or {}
        insights = insights or {}
        security_issues = [w for w in weaknesses if w.category in {"security", "dependency"}]
        categories = {
            "documentation": self._documentation(facts, insights),
            "testing": self._testing(facts, weaknesses),
            "structure": self._structure(facts, graph),
            "run_deployment_readiness": self._run_readiness(facts, weaknesses),
            "security_hygiene": self._security(weaknesses, insights),
        }
        total = sum(item["score"] for item in categories.values())
        maximum = sum(item.get("assessed_max", 20) for item in categories.values())
        return {
            "total": total,
            "assessed_max": maximum,
            "assessment_coverage": round(maximum / 100, 2),
            "possible_total": [total, total + 100 - maximum],
            "grade": "unrated" if maximum < 100 else "A" if total >= 85 else "B" if total >= 70 else "C" if total >= 55 else "D" if total >= 40 else "E",
            "categories": categories,
            "security_review": self._security_review(security_issues),
            "top_improvements": self._top_improvements(facts, weaknesses)[:5],
            "method": "5 categories x 20 points from measured repository signals; see each category's reasons.",
        }

    def _cap(self, score: float, reasons: list[str]) -> dict:
        return {"score": max(0, min(20, round(score))), "max": 20, "reason": "; ".join(reasons) if reasons else "No positive signals detected."}

    def _documentation(self, facts: dict, insights: dict[str, Any]) -> dict:
        score, reasons = 0.0, []
        if facts.get("has_readme"):
            score += 6
            reasons.append("README present (+6)")
        if facts.get("has_run_instructions"):
            score += 5
            reasons.append("documented run commands (+5)")
        if facts.get("docs_dirs"):
            score += 3
            reasons.append("docs folder (+3)")
        if any(path.lower().endswith(".env.example") for path in facts.get("config_files", [])):
            score += 2
            reasons.append(".env.example (+2)")
        accuracy = (insights.get("drift") or {}).get("docs_accuracy")
        if accuracy is not None:
            score += 4 * accuracy
            reasons.append(f"docs accuracy {accuracy:.0%} (+{4 * accuracy:.1f})")
        return self._cap(score, reasons)

    def _testing(self, facts: dict, weaknesses: list[Weakness]) -> dict:
        score, reasons = 0.0, []
        code, tests = facts.get("code_file_count", 0), facts.get("test_file_count", 0)
        if facts.get("has_tests"):
            score += 8
            reasons.append("tests present (+8)")
        if code:
            ratio = min(1.0, tests / code / 0.5)
            score += 6 * ratio
            reasons.append(f"test/code file ratio {tests}/{code} (+{6 * ratio:.1f})")
        if not any(w.title == "No CI workflow detected" for w in weaknesses):
            score += 6
            reasons.append("CI workflow (+6)")
        return self._cap(score, reasons)

    def _structure(self, facts: dict, graph: dict[str, Any]) -> dict:
        score, reasons = 4.0, ["baseline (+4)"]
        if facts.get("source_dirs"):
            score += 4
            reasons.append("source directories (+4)")
        if facts.get("main_entrypoints"):
            score += 4
            reasons.append("clear entrypoints (+4)")
        cycles = len(graph.get("dependency_cycles", []))
        score += 4 if cycles == 0 else max(0, 4 - cycles)
        reasons.append(f"{cycles} import cycle(s)")
        complex_count = sum(1 for item in graph.get("complex_functions", []) if (item.get("complexity") or 0) >= 15)
        score += 4 if complex_count == 0 else max(0, 4 - complex_count / 2)
        reasons.append(f"{complex_count} highly complex function(s)")
        return self._cap(score, reasons)

    def _run_readiness(self, facts: dict, weaknesses: list[Weakness]) -> dict:
        score, reasons = 0.0, []
        if facts.get("has_run_instructions"):
            score += 7
            reasons.append("run instructions (+7)")
        if facts.get("dependency_files"):
            score += 5
            reasons.append("dependency manifest (+5)")
        if facts.get("deployment_files"):
            score += 5
            reasons.append("deployment files (+5)")
        if not any(w.title == "No linting or formatting config detected" for w in weaknesses):
            score += 3
            reasons.append("lint/format config (+3)")
        return self._cap(score, reasons)

    def _security(self, weaknesses: list[Weakness], insights: dict[str, Any]) -> dict:
        score, reasons = 20.0, []
        high = sum(1 for w in weaknesses if w.category == "security" and w.severity == "high")
        medium = sum(1 for w in weaknesses if w.category == "security" and w.severity == "medium")
        if high or medium:
            score -= 4 * high + 1.5 * medium
            reasons.append(f"{high} high / {medium} medium static security findings")
        deps = insights.get("dependencies") or {}
        vulns = deps.get("vulnerabilities") or []
        if vulns:
            reachable = sum(1 for v in vulns if v.get("reachable"))
            score -= min(10, 2 * reachable + 0.5 * (len(vulns) - reachable))
            reasons.append(f"{len(vulns)} known vulnerable dependency versions ({reachable} imported)")
        elif deps.get("scan_status") != "ok":
            reasons.append("dependency CVE scan not available")
        if not reasons:
            reasons.append("no static security findings")
        assessed = deps.get("scan_status") == "ok"
        result = self._cap(score if assessed else max(0, score - 10), reasons)
        return {**result, "status": "checked" if assessed else "partial",
                "dependency_status": deps.get("scan_status", "unknown"), "assessed_max": 20 if assessed else 10}

    def _security_review(self, security_issues: list[Weakness]) -> dict:
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        highest = max((item.severity for item in security_issues), key=lambda value: severity_order.get(value, 0), default="none")
        return {
            "label": "automated static review findings",
            "finding_count": len(security_issues),
            "highest_severity": highest,
            "confidence_summary": {level: sum(1 for item in security_issues if item.confidence == level) for level in ("high", "medium", "low")},
            "note": "These findings are static signals, not an audit-level guarantee.",
        }

    def _top_improvements(self, facts: dict, weaknesses: list[Weakness]) -> list[str]:
        improvements = [f"{w.title}: {w.suggested_fix}" for w in weaknesses if w.severity in {"critical", "high", "medium"}]
        if not facts.get("has_run_instructions"):
            improvements.append("Document run commands in README with expected environment variables.")
        if not facts.get("has_tests"):
            improvements.append("Add smoke tests or unit tests for the main workflow.")
        return list(dict.fromkeys(improvements))
