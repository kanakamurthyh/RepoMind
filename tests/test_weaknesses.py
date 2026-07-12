from __future__ import annotations

from app.analysis.weaknesses import WeaknessDetector
from app.models import FileRecord


def record(path: str, text: str, language: str = "python", role: str = "code") -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def test_weakness_detector_finds_missing_project_hygiene_and_security_patterns():
    records = [
        record("app.py", "import subprocess\nsubprocess.run(user_input, shell=True)\nAPI_KEY='123456789abc'"),
        record("requirements.txt", "fastapi", "text", "config"),
    ]

    weaknesses = WeaknessDetector().detect(records)
    issues = {weakness.issue for weakness in weaknesses}

    assert "Missing README" in issues
    assert "No tests detected" in issues
    assert "subprocess shell=True detected" in issues
    assert "High-confidence hardcoded secret" in issues


def test_secret_detector_ignores_readme_prose_false_positive():
    records = [
        record("README.md", "Set your token or API key in the environment. Do not commit credentials.", "markdown", "docs"),
        record("requirements.txt", "fastapi", "text", "config"),
        record("tests/test_app.py", "def test_ok(): assert True", "python", "test"),
    ]

    weaknesses = WeaknessDetector().detect(records, {"has_tests": True, "dependency_files": ["requirements.txt"], "has_run_instructions": True})

    assert "High-confidence hardcoded secret" not in {weakness.issue for weakness in weaknesses}


def test_secret_detector_flags_assigned_real_secret_with_evidence():
    records = [record("settings.py", "OPENAI_API_KEY = 'sk-abcdefghijklmnopqrstuvwxyz123456'")]

    weaknesses = WeaknessDetector().detect(records)
    secret = next(weakness for weakness in weaknesses if weakness.issue == "High-confidence hardcoded secret")

    assert secret.line_number == 1
    assert secret.confidence == "high"
    assert secret.evidence and "<redacted>" in secret.evidence


def test_weakness_detector_deduplicates_findings():
    records = [record("app.py", "eval(user_input)\neval(user_input)\n")]

    weaknesses = WeaknessDetector().detect(records)
    ids = [weakness.id for weakness in weaknesses]

    assert len(ids) == len(set(ids))
