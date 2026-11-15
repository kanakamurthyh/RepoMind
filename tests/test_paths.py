from __future__ import annotations

from pathlib import Path

import pytest

from app.utils.paths import is_valid_repo_id, legacy_repo_id_from_url, normalize_github_url, repo_id_from_url, safe_join


def test_github_url_normalization_is_case_stable() -> None:
    mixed = "https://github.com/VedantMoghe16/jiopc-home"
    lowered = "https://github.com/vedantmoghe16/jiopc-home/"

    assert normalize_github_url(mixed) == "https://github.com/vedantmoghe16/jiopc-home"
    assert repo_id_from_url(mixed) == repo_id_from_url(lowered)
    assert legacy_repo_id_from_url(mixed) != repo_id_from_url(mixed)


@pytest.mark.parametrize("url", ["https://gitlab.com/a/b", "file:///etc/passwd", "https://github.com/onlyowner", "https://github.com/../x", "https://github.com/a/b;rm"])
def test_invalid_urls_are_rejected(url):
    with pytest.raises(ValueError):
        normalize_github_url(url)


@pytest.mark.parametrize("repo_id,valid", [("owner-repo-0123456789", True), ("..", False), ("../etc-0123456789", False), ("a/b-0123456789", False), ("owner-repo", False)])
def test_repo_id_validation(repo_id, valid):
    assert is_valid_repo_id(repo_id) is valid


def test_safe_join_blocks_escape(tmp_path: Path):
    assert safe_join(tmp_path, "a/b.txt") == (tmp_path / "a/b.txt").resolve()
    with pytest.raises(ValueError):
        safe_join(tmp_path, "../outside")
