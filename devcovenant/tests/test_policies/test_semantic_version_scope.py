"""Tests for the semantic-version-scope policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.semantic_version_scope import (
    SemanticVersionScopeCheck,
)


def _write_version_files(
    tmp_path: Path,
    current_version: str,
    previous_version: str,
    marker: str,
) -> tuple[Path, Path, Path]:
    """Create VERSION and CHANGELOG fixtures."""
    version_file = tmp_path / "copernican_lib" / "VERSION"
    version_file.parent.mkdir(parents=True, exist_ok=True)
    version_file.write_text(current_version, encoding="utf-8")

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "\n".join(
            [
                "# Changelog",
                "",
                f"## Version {current_version}",
                marker,
                "",
                f"## Version {previous_version}",
                "- 2025-12-24 [semver:patch]: previous release",
            ]
        ),
        encoding="utf-8",
    )

    other_file = tmp_path / "copernican_lib" / "module.py"
    other_file.write_text("# helper\n", encoding="utf-8")
    return version_file, changelog, other_file


def test_minor_marker_requires_minor_bump(tmp_path: Path) -> None:
    """A patch bump should fail when the changelog requests a minor bump."""
    version_file, changelog, other_file = _write_version_files(
        tmp_path,
        "1.2.4",
        "1.2.3",
        "- 2025-12-28 [semver:minor]: latest release",
    )

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[version_file, changelog, other_file],
    )
    check = SemanticVersionScopeCheck()
    violations = check.check(context)

    assert violations
    assert any("minor" in violation.message for violation in violations)


def test_minor_bump_passes_with_matching_marker(tmp_path: Path) -> None:
    """A minor bump should pass when the changelog tags it as minor."""
    version_file, changelog, other_file = _write_version_files(
        tmp_path,
        "1.3.0",
        "1.2.4",
        "- 2025-12-28 [semver:minor]: latest release",
    )

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[version_file, changelog, other_file],
    )
    check = SemanticVersionScopeCheck()

    assert check.check(context) == []


def test_override_file_sets_required_scope(tmp_path: Path) -> None:
    """Override files allow releases without explicit changelog tags."""
    version_file, changelog, other_file = _write_version_files(
        tmp_path,
        "2.0.1",
        "2.0.0",
        "- 2025-12-28 no marker",
    )
    waiver_dir = tmp_path / ".devcovenant" / "waivers"
    waiver_dir.mkdir(parents=True, exist_ok=True)
    (waiver_dir / "semantic-version-scope.txt").write_text(
        "override=patch\n", encoding="utf-8"
    )

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[version_file, changelog, other_file],
    )

    check = SemanticVersionScopeCheck()
    assert check.check(context) == []


def test_policy_skips_when_only_devconvenant_changes(tmp_path: Path) -> None:
    """Changes scoped to devcovenant should not trigger Copernican bumps."""
    version_file, _, _ = _write_version_files(
        tmp_path,
        "3.0.1",
        "3.0.0",
        "- 2025-12-28 [semver:patch]",
    )
    devcov_file = tmp_path / "devcovenant" / "engine.py"
    devcov_file.parent.mkdir(parents=True, exist_ok=True)
    devcov_file.write_text("# engine\n", encoding="utf-8")

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[version_file, devcov_file],
    )

    check = SemanticVersionScopeCheck()
    assert check.check(context) == []


def test_requires_version_bump_when_changelog_changes(tmp_path: Path) -> None:
    """Changelog releases must update the version file."""
    _, changelog, other_file = _write_version_files(
        tmp_path,
        "4.0.0",
        "3.5.0",
        "- 2025-12-28 [semver:patch]: docs refresh",
    )

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[changelog, other_file],
    )
    check = SemanticVersionScopeCheck()
    violations = check.check(context)

    assert violations
    assert any("VERSION was not updated" in v.message for v in violations)


def test_mixed_scope_markers_are_rejected(tmp_path: Path) -> None:
    """Multiple scope markers in one release cause a violation."""
    version_file, changelog, other_file = _write_version_files(
        tmp_path,
        "5.0.0",
        "4.9.0",
        "- 2025-12-28 [semver:major]: breaking change\n"
        "- 2025-12-28 [semver:patch]: docs tweak",
    )

    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[version_file, changelog, other_file],
    )
    check = SemanticVersionScopeCheck()
    violations = check.check(context)

    assert violations
    assert any("mixes multiple SemVer scopes" in v.message for v in violations)
