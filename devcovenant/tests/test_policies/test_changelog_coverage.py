"""
Tests for changelog-coverage policy.
"""

from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.changelog_coverage import (
    ChangelogCoverageCheck,
)


def _set_git_diff(monkeypatch: pytest.MonkeyPatch, output: str) -> None:
    """Monkeypatch subprocess.run to return the provided diff output."""

    def _fake_run(*_args, **_kwargs):
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr("subprocess.run", _fake_run)


def test_no_changes_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Empty diffs should yield no violations."""

    _set_git_diff(monkeypatch, "")
    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    assert checker.check(context) == []


def test_root_changelog_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Non-RNG files must be listed in the root changelog."""

    (tmp_path / "CHANGELOG.md").write_text("docs/readme.md", encoding="utf-8")
    _set_git_diff(monkeypatch, "docs/readme.md\nsrc/module.py\n")

    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    violations = checker.check(context)

    assert len(violations) == 1
    assert "CHANGELOG.md" in violations[0].message
    assert "src/module.py" in violations[0].message


def test_rng_changelog_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """RNG files must be documented in rng_minigames/CHANGELOG.md."""

    (tmp_path / "CHANGELOG.md").write_text("", encoding="utf-8")
    _set_git_diff(monkeypatch, "rng_minigames/emoji_meteors/game.py\n")

    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    violations = checker.check(context)

    assert len(violations) == 1
    assert "rng_minigames/CHANGELOG.md" in violations[0].message


def test_rng_changelog_entry_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """RNG files pass when mentioned in rng_minigames/CHANGELOG.md."""

    (tmp_path / "CHANGELOG.md").write_text("", encoding="utf-8")
    rng_changelog = tmp_path / "rng_minigames" / "CHANGELOG.md"
    rng_changelog.parent.mkdir(parents=True, exist_ok=True)
    rng_changelog.write_text(
        "rng_minigames/emoji_meteors/game.py", encoding="utf-8"
    )
    _set_git_diff(monkeypatch, "rng_minigames/emoji_meteors/game.py\n")

    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    violations = checker.check(context)

    assert violations == []


def test_rng_files_not_logged_in_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """RNG files should not appear in the root changelog."""

    root_changelog = tmp_path / "CHANGELOG.md"
    root_changelog.write_text(
        "rng_minigames/emoji_meteors/game.py", encoding="utf-8"
    )
    rng_changelog = tmp_path / "rng_minigames" / "CHANGELOG.md"
    rng_changelog.parent.mkdir(parents=True, exist_ok=True)
    rng_changelog.write_text(
        "rng_minigames/emoji_meteors/game.py", encoding="utf-8"
    )
    _set_git_diff(monkeypatch, "rng_minigames/emoji_meteors/game.py\n")

    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    violations = checker.check(context)

    assert len(violations) == 1
    assert "root changelog" in violations[0].message


def test_rng_entries_ignore_old_root_sections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Old root entries mentioning RNG files should not trigger violations."""

    root_changelog = tmp_path / "CHANGELOG.md"
    root_changelog.write_text(
        dedent(
            """
            ## Version 2.0.0
            - entry about docs/readme.md

            ## Version 1.0.0
            - rng_minigames/emoji_meteors/game.py
            """
        ).strip(),
        encoding="utf-8",
    )
    rng_changelog = tmp_path / "rng_minigames" / "CHANGELOG.md"
    rng_changelog.parent.mkdir(parents=True, exist_ok=True)
    rng_changelog.write_text(
        "## Version 2.0.0\n- rng_minigames/emoji_meteors/game.py",
        encoding="utf-8",
    )
    _set_git_diff(monkeypatch, "rng_minigames/emoji_meteors/game.py\n")

    checker = ChangelogCoverageCheck()
    context = CheckContext(repo_root=tmp_path, all_files=[])
    violations = checker.check(context)

    assert violations == []
