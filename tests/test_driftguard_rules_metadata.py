"""Metadata rule coverage for DriftGuard."""

from __future__ import annotations

import datetime
import subprocess
from pathlib import Path

from driftguard.rules import RuleContext
from driftguard.rules.metadata import (
    ChangelogDiffCoverageRule,
    ChangelogRule,
    CitationYamlRule,
    LastUpdatedDocsRule,
    NoFutureDatesRule,
    PolicySyncRule,
    SemverBumpRequiredRule,
    StartLauncherParityRule,
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
                include=["README.md", "DRIFTGUARD.md"],
                exclude=[],
                rules=["last-updated-header"],
            ),
            "interfaces": SurfaceSpec(
                name="interfaces",
                include=["start.sh"],
                exclude=[],
                rules=["last-updated-header"],
            ),
            "python-lib": SurfaceSpec(
                name="python-lib",
                include=["copernican_lib/**/*.py"],
                exclude=[],
                rules=[],
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


def _init_git_repo(repo_root: Path) -> None:
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "ci@example.com"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "CI"], cwd=repo_root, check=True
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


def test_policy_sync_requires_joint_updates(tmp_path: Path) -> None:
    """Policy documents must change alongside the YAML and enforcement."""

    _init_git_repo(tmp_path)
    (tmp_path / "driftguard").mkdir()
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_yaml = tmp_path / "driftguard" / "repo_policy.yml"
    enforcement = tmp_path / "driftguard" / "core.py"

    policy_doc.write_text("Seed policy\n", encoding="utf-8")
    policy_yaml.write_text("version: 1\n", encoding="utf-8")
    enforcement.write_text("print('seed')\n", encoding="utf-8")

    subprocess.run(["git", "add", "DRIFTGUARD.md", "driftguard"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True)

    policy_doc.write_text("Seed policy\nUpdated guidance\n", encoding="utf-8")

    rule = PolicySyncRule()
    context = _context(tmp_path)

    violations = rule.check(context)

    assert violations
    assert "repo_policy.yml" in violations[0].message


def test_policy_sync_passes_when_all_surfaces_change(tmp_path: Path) -> None:
    """Aligned changes to doc, YAML and enforcement should pass."""

    _init_git_repo(tmp_path)
    (tmp_path / "driftguard").mkdir()
    policy_doc = tmp_path / "DRIFTGUARD.md"
    policy_yaml = tmp_path / "driftguard" / "repo_policy.yml"
    enforcement = tmp_path / "driftguard" / "core.py"

    policy_doc.write_text("Seed policy\n", encoding="utf-8")
    policy_yaml.write_text("version: 1\n", encoding="utf-8")
    enforcement.write_text("print('seed')\n", encoding="utf-8")

    subprocess.run(["git", "add", "DRIFTGUARD.md", "driftguard"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "seed"], cwd=tmp_path, check=True, capture_output=True)

    policy_doc.write_text("Seed policy\nUpdated guidance\n", encoding="utf-8")
    policy_yaml.write_text("version: 2\n", encoding="utf-8")
    enforcement.write_text("print('seed')\n# updated enforcement\n", encoding="utf-8")

    rule = PolicySyncRule()
    context = _context(tmp_path)

    assert rule.check(context) == []


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

def test_changelog_diff_coverage_flags_missing_files(tmp_path: Path) -> None:
    repo_root = tmp_path
    changelog = repo_root / "CHANGELOG.md"
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "**Last Updated:** 2025-11-27",
                "",
                "## Version 1.0.0",
                "- 2025-11-27: Initial entry without file references.",
            ]
        ),
        encoding="utf-8",
    )
    target = repo_root / "copernican_lib" / "example.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("print('hi')\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo_root, check=True)
    subprocess.run(["git", "add", "CHANGELOG.md", "copernican_lib/example.py"], cwd=repo_root, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True)

    target.write_text("print('bye')\n", encoding="utf-8")

    rule = ChangelogDiffCoverageRule()
    context = _context(repo_root)
    violations = rule.check(context)

    assert violations
    assert violations[0].path == changelog


def test_start_scripts_require_parity(tmp_path: Path) -> None:
    repo_root = tmp_path
    for name in ("start.sh", "start.bat", "start.command"):
        (repo_root / name).write_text(
            "# Last Updated: 2025-11-26\n", encoding="utf-8"
        )
    subprocess.run(["git", "init"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo_root, check=True)
    subprocess.run(
        ["git", "add", "start.sh", "start.bat", "start.command"],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True)

    start_sh = repo_root / "start.sh"
    start_sh.write_text("# Last Updated: 2025-11-27\n", encoding="utf-8")

    rule = StartLauncherParityRule()
    context = _context(repo_root)
    violations = rule.check(context)

    assert violations


def test_semver_bump_required_for_code_changes(tmp_path: Path) -> None:
    repo_root = tmp_path
    version_file = repo_root / "copernican_lib" / "VERSION"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text("1.0.0\n", encoding="utf-8")
    readme = repo_root / "README.md"
    readme.write_text(
        "**Version:** 1.0.0\n**Last Updated:** 2025-11-27\n",
        encoding="utf-8",
    )
    changelog = repo_root / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n## Version 1.0.0\n- 2025-11-27: Init\n",
        encoding="utf-8",
    )
    module = repo_root / "copernican_lib" / "module.py"
    module.write_text("print('change')\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.email", "ci@example.com"], cwd=repo_root, check=True)
    subprocess.run(["git", "config", "user.name", "CI"], cwd=repo_root, check=True)
    subprocess.run(
        [
            "git",
            "add",
            "README.md",
            "CHANGELOG.md",
            "copernican_lib/VERSION",
            "copernican_lib/module.py",
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(["git", "commit", "-m", "init"], cwd=repo_root, check=True)

    module.write_text("print('modified')\n", encoding="utf-8")

    rule = SemverBumpRequiredRule()
    context = _context(repo_root)
    violations = rule.check(context)

    assert violations
