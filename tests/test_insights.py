from __future__ import annotations

import subprocess

import pytest

from app.analysis.dependencies import DependencyScanner
from app.analysis.history import HistoryAnalyzer
from app.models import FileRecord


def rec(path, text, language="text", role="config"):
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def test_docs_drift_detects_missing_route_unused_env_and_broken_link(shop_env):
    analyzer, _, repo_id = shop_env
    drift = analyzer.load(repo_id).summary["insights"]["docs_drift"]
    kinds = {(issue["kind"], issue.get("evidence")) for issue in drift["top_issues"]}

    assert ("route_missing", "DELETE /orders/{order_id}") in kinds
    assert ("env_unused", "LEGACY_CACHE_TTL") in kinds
    assert ("broken_link", "docs/architecture.md") in kinds
    assert 0 < drift["docs_accuracy"] < 1


def test_docs_drift_does_not_flag_valid_commands(shop_env):
    analyzer, _, repo_id = shop_env
    drift = analyzer.load(repo_id).summary["insights"]["docs_drift"]
    assert not any(issue["kind"] == "command" for issue in drift["top_issues"])


class FakeResponse:
    def __init__(self, body, ok=True):
        self._body, self.ok = body, ok

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


def test_dependency_scan_reports_reachable_vulnerabilities_with_fixed_versions():
    records = [rec("requirements.txt", "requests==2.19.0\nflask>=2\n"), rec("package-lock.json", '{"packages": {"node_modules/lodash": {"version": "4.17.4"}}}')]
    posted = {}

    def fake_post(url, json, timeout):
        posted["queries"] = json["queries"]
        return FakeResponse({"results": [{"vulns": [{"id": "GHSA-x"}]} if q["package"]["name"] == "requests" else {} for q in json["queries"]]})

    def fake_get(url, timeout):
        return FakeResponse({"summary": "Leaks auth header", "aliases": ["CVE-2018-18074"], "database_specific": {"severity": "HIGH"},
                             "affected": [{"package": {"name": "requests"}, "ranges": [{"events": [{"introduced": "0"}, {"fixed": "2.20.0"}]}]}]})

    scanner = DependencyScanner(http_post=fake_post, http_get=fake_get)
    result = scanner.scan(records, {"requests": ["app/client.py"]})

    assert {q["package"]["ecosystem"] for q in posted["queries"]} == {"PyPI", "npm"}
    vuln = result["vulnerabilities"][0]
    assert (vuln["package"], vuln["severity"], vuln["fixed_versions"], vuln["reachable"]) == ("requests", "high", ["2.20.0"], True)
    assert result["unpinned"][0]["package"] == "flask"


def test_dependency_scan_degrades_gracefully_offline():
    import requests

    def boom(*args, **kwargs):
        raise requests.ConnectionError("offline")

    result = DependencyScanner(http_post=boom).scan([rec("requirements.txt", "requests==2.19.0\n")])
    assert result["scan_status"] == "offline" and result["vulnerabilities"] == []


def test_toml_inventory_is_structural_and_partial_inventory_is_unknown():
    scanner = DependencyScanner(http_post=lambda *a, **kw: FakeResponse({"results": [{}]}))
    valid = rec("uv.lock", 'version = 1\n[[package]]\nversion = "1.2.3"\nsource = { registry = "https://example.invalid" }\nname = "sample"\n')
    broken = rec("poetry.lock", "[[package]\n")
    result = scanner.scan([valid, broken])
    assert result["inventory_packages"] == 1 and result["packages_scanned"] == 1
    assert result["scan_status"] == "partial" and result["inventory_warnings"]
    packages, _ = scanner.inventory([valid])
    assert (packages[0].name, packages[0].version) == ("sample", "1.2.3")


def test_inventory_caps_and_missing_advisory_responses_are_visible():
    scanner = DependencyScanner(http_post=lambda *a, **kw: FakeResponse({"results": []}))
    result = scanner.scan([rec("requirements.txt", "\n".join(f"pkg{i}==1.0" for i in range(1501)))])
    assert result["inventory_packages"] == 1501
    assert result["inventory_truncated"] and result["packages_scanned"] == 0
    assert result["scan_status"] == "partial"


@pytest.mark.parametrize("path,text", [("package-lock.json", "[]"), ("uv.lock", 'package = "invalid"')])
def test_invalid_lockfile_structure_degrades_to_unknown(path, text):
    result = DependencyScanner(enabled=False).scan([rec(path, text)])
    assert result["scan_status"] == "unavailable"
    assert result["inventory_warnings"] and result["packages_scanned"] == 0


def test_history_hotspots_and_bus_factor(tmp_path):
    def git(*args, author="Ada"):
        subprocess.run(["git", "-c", f"user.name={author}", "-c", "user.email=a@example.com", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "core.py").write_text("a = 1\n")
    git("add", ".")
    git("commit", "-qm", "one")
    for index in range(3):
        (tmp_path / "core.py").write_text(f"a = {index}\nb = 2\n")
        git("commit", "-qam", f"edit {index}", author="Grace" if index else "Ada")
    result = HistoryAnalyzer().analyze(tmp_path, {"core.py": 2}, {"core.py": 12})

    assert result["status"] == "ok" and result["commits_analyzed"] == 4
    assert result["hotspots"][0]["path"] == "core.py" and result["hotspots"][0]["commits"] == 4
    assert result["bus_factor"] == 1
