"""Metadata rule coverage for DriftGuard."""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from driftguard.rules import RuleContext
from driftguard.rules.metadata import (
    ChangelogRule,
    CitationYamlRule,
    LastUpdatedDocsRule,
    NoFutureDatesRule,
    VersionSyncRule,
)
from driftguard.spec import DriftConfig, DriftGuardSpec, SurfaceSpec


def _spec_with_surfaces() -> DriftGuardSpec:
    return DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={},
        surfaces={
            "docs": SurfaceSpec(
                name="docs",
                include=["README.md"],
                exclude=[],
                rules=["last-updated-header"],
            ),
            "interfaces": SurfaceSpec(
                name="interfaces",
                include=["start.sh"],
                exclude=[],
                rules=["last-updated-header"],
            ),
            "metadata": SurfaceSpec(
                name="metadata",
                include=[
                    "README.md",
                    "CITATION.cff",
                    "copernican_lib/VERSION",
                ],
                exclude=[],
                rules=[
                    "version-sync",
                    "citation-yaml",
                    "no-future-dates",
                    "changelog-entry",
                ],
            ),
        },
        drift=DriftConfig(),
    )


def _context(repo_root: Path) -> RuleContext:
    return RuleContext(
        repo_root=repo_root,
        spec=_spec_with_surfaces(),
        scope="repo",
        mode="full",
    )


def test_last_updated_rule_adds_missing_headers(tmp_path: Path) -> None:
    """Missing headers on doc and interface surfaces should be auto-fixed."""

    readme = tmp_path / "README.md"
    script = tmp_path / "start.sh"
    readme.write_text("Readme body.\n", encoding="utf-8")
    script.write_text("#!/bin/bash\necho hi\n", encoding="utf-8")

    rule = LastUpdatedDocsRule()
    context = _context(tmp_path)
    violations = rule.check(context)

    assert len(violations) == 2
    assert all(violation.fixable for violation in violations)

    rule.fix(context, safe_only=True)

    readme_lines = readme.read_text(encoding="utf-8").splitlines()
    script_lines = script.read_text(encoding="utf-8").splitlines()
    assert readme_lines[0].startswith("**Last Updated:**")
    assert script_lines[0].startswith("#!")
    assert script_lines[1].startswith("# Last Updated:")


def test_last_updated_rule_rejects_late_header(tmp_path: Path) -> None:
    """Headers buried beyond the third line should raise violations."""

    readme = tmp_path / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Overview",
                "",
                "Body line one.",
                "**Last Updated:** 2025-01-01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rule = LastUpdatedDocsRule()
    context = _context(tmp_path)

    violations = rule.check(context)

    assert violations
    assert "first three lines" in violations[0].message


def test_version_sync_and_citation_rules_detect_mismatches(
    tmp_path: Path,
) -> None:
    """Version and citation rules should flag inconsistent metadata."""

    (tmp_path / "copernican_lib").mkdir()
    (tmp_path / "copernican_lib" / "VERSION").write_text("1.0.0\n")
    (tmp_path / "README.md").write_text(
        "**Version:** 2.0.0\n**Last Updated:** 2025-11-25\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "\n".join(
            [
                "# Last Updated: 2025-11-25",
                "cff-version: 1.2.0",
                'title: "Test"',
                'version: "1.0.1"',
                'date-released: "2025-11-25"',
                "authors:",
                "  - name: Example",
                "preferred-citation:",
                "  type: article",
                '  title: "Example"',
                '  date-released: "2025-11-25"',
                '  version: "1.0.1"',
            ]
        ),
        encoding="utf-8",
    )

    context = _context(tmp_path)
    version_rule = VersionSyncRule()
    citation_rule = CitationYamlRule()

    version_violations = version_rule.check(context)
    citation_violations = citation_rule.check(context)

    assert any(
        violation.path and violation.path.name == "README.md"
        for violation in version_violations
    )
    assert any(
        violation.rule_name == citation_rule.name
        for violation in citation_violations
    )


def test_citation_rule_requires_authors_and_release_alignment(
    tmp_path: Path,
) -> None:
    """Citation metadata should supply authors and match its header date."""

    (tmp_path / "copernican_lib").mkdir()
    (tmp_path / "copernican_lib" / "VERSION").write_text("1.0.0\n")
    (tmp_path / "README.md").write_text(
        "**Version:** 1.0.0\n**Last Updated:** 2025-11-25\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        "\n".join(
            [
                "# Last Updated: 2025-11-26",
                "cff-version: 1.2.0",
                'title: "Test"',
                'version: "1.0.0"',
                'date-released: "2025-11-25"',
                "authors: []",
                "preferred-citation:",
                "  type: software",
                '  title: "Test"',
                '  version: "1.0.0"',
                '  date-released: "2025-11-25"',
                "  authors:",
                "    - name: Example",
            ]
        ),
        encoding="utf-8",
    )

    context = _context(tmp_path)
    rule = CitationYamlRule()

    violations = rule.check(context)

    assert len(violations) >= 2
    assert any("author" in v.message.lower() for v in violations)
    assert any("date-released" in v.message for v in violations)


def test_no_future_dates_flags_future_metadata(tmp_path: Path) -> None:
    """Future timestamps should trigger violations across tracked surfaces."""

    future_date = (
        datetime.date.today() + datetime.timedelta(days=1)
    ).isoformat()
    (tmp_path / "README.md").write_text(
        f"**Last Updated:** {future_date}\n",
        encoding="utf-8",
    )
    (tmp_path / "CITATION.cff").write_text(
        f'# Last Updated: {future_date}\nversion: "1.0.0"\n',
        encoding="utf-8",
    )
    (tmp_path / "copernican_lib").mkdir()
    (tmp_path / "copernican_lib" / "VERSION").write_text("1.0.0\n")

    context = _context(tmp_path)
    rule = NoFutureDatesRule()
    violations = rule.check(context)

    assert len(violations) >= 2
    assert all(future_date in violation.message for violation in violations)


def test_changelog_rule_requires_updates(tmp_path: Path) -> None:
    """Git status should require changelog edits alongside other changes."""

    changelog = tmp_path / "CHANGELOG.md"
    readme = tmp_path / "README.md"
    changelog.write_text("# Last Updated: 2025-11-25\n", encoding="utf-8")
    readme.write_text("**Last Updated:** 2025-11-25\n", encoding="utf-8")

    subprocess.run(
        ["git", "init"], cwd=tmp_path, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CI"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "add", "CHANGELOG.md", "README.md"], cwd=tmp_path, check=True
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    readme.write_text(
        "**Last Updated:** 2025-11-25\nUpdated.\n", encoding="utf-8"
    )

    context = _context(tmp_path)
    rule = ChangelogRule()
    violations = rule.check(context)
    assert violations

    changelog.write_text(
        "# Last Updated: 2025-11-25\nEntry.\n", encoding="utf-8"
    )
    subprocess.run(
        ["git", "add", "CHANGELOG.md", "README.md"], cwd=tmp_path, check=True
    )

    assert rule.check(context) == []
