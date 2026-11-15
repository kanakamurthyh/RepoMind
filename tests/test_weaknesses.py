from __future__ import annotations

from app.analysis.weaknesses import WeaknessDetector
from app.models import FileRecord


def record(path: str, text: str, language: str = "python", role: str = "code") -> FileRecord:
    return FileRecord("demo", path, f"/tmp/{path}", language, role, len(text), text.count("\n") + 1, text)


def titles(weaknesses):
    return {weakness.title for weakness in weaknesses}


def test_detects_hygiene_and_ast_security_patterns():
    records = [
        record("app.py", "import subprocess\nsubprocess.run(user_input, shell=True)\nAPI_KEY='Zr8Qx2LmP0vTq9Wd'\neval(user_input)\ncursor.execute(f\"SELECT * FROM t WHERE id={uid}\")\n"),
        record("requirements.txt", "fastapi", "text", "config"),
    ]
    found = titles(WeaknessDetector().detect(records))

    assert {"Missing README", "No tests detected", "subprocess shell=True detected", "High-confidence hardcoded secret", "Dynamic code execution (eval)", "SQL built with string formatting"} <= found


def test_ml_code_is_not_flagged_as_dynamic_execution():
    records = [record("train.py", "model.eval()\nvalue = ast.literal_eval(s)\nctx = retrieval(query)\n")]
    assert not any("Dynamic code execution" in title for title in titles(WeaknessDetector().detect(records)))


def test_secret_detector_ignores_prose_regex_constants_and_fake_fixtures():
    records = [
        record("README.md", "Set your token or API key in the environment.", "markdown", "docs"),
        record("scan.py", 'GITHUB_TOKEN_RE = re.compile(r"gh_x")\n'),
        record("tests/test_x.py", "key = 'sk-abcdefghijklmnopqrstuvwxyz123456'", "python", "test"),
    ]
    found = titles(WeaknessDetector().detect(records, {"has_tests": True}))
    assert "High-confidence hardcoded secret" not in found
    assert "Possible secret in test fixture" not in found


def test_secret_evidence_is_redacted():
    weaknesses = WeaknessDetector().detect([record("settings.py", "OPENAI_API_KEY = 'sk-Q7vK2pX9mL4tR8wZ1nB5cY3dF6hJ0sA'")])
    secret = next(w for w in weaknesses if w.title == "High-confidence hardcoded secret")

    assert secret.line_number == 1 and secret.confidence == "high"
    assert "Q7vK2pX9" not in secret.evidence and "<redacted-secret>" in secret.evidence


def test_committed_env_file_and_js_xss_are_flagged():
    records = [record(".env", "", "unknown", "other"), record("web/app.js", "el.innerHTML = userInput;\n", "javascript")]
    found = titles(WeaknessDetector().detect(records))
    assert "Environment file committed to the repository" in found
    assert "innerHTML assignment" in found


def test_weakness_ids_are_repo_scoped_and_deduplicated():
    records = [record("app.py", "eval(user_input)\neval(user_input)\n")]
    a = WeaknessDetector().detect(records, repo_id="owner-a-0000000000")
    b = WeaknessDetector().detect(records, repo_id="owner-b-1111111111")

    assert len({w.id for w in a}) == len(a)
    assert not {w.id for w in a} & {w.id for w in b}
