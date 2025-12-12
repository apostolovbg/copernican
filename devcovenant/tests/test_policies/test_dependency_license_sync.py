"""Tests for the dependency-license-sync policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.dependency_license_sync import (
    DependencyLicenseSyncCheck,
)


def _setup_repo(tmp_path: Path) -> Path:
    tmp_path.joinpath("licenses").mkdir(parents=True, exist_ok=True)
    (tmp_path / "requirements.in").write_text("numpy==1.0\n", encoding="utf-8")
    (tmp_path / "requirements.lock").write_text("numpy==1.0\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'\n", encoding="utf-8")
    (tmp_path / "THIRD_PARTY_LICENSES.md").write_text(
        "# Third-Party Licenses\n", encoding="utf-8"
    )
    (tmp_path / "licenses" / "BSD-3-Clause.txt").write_text(
        "BSD text\n", encoding="utf-8"
    )
    return tmp_path


def test_requires_license_table_update(tmp_path: Path):
    """Dependency changes without touching the license table fail."""
    repo = _setup_repo(tmp_path)
    checker = DependencyLicenseSyncCheck()
    context = CheckContext(
        repo_root=repo,
        changed_files=[repo / "requirements.in"],
    )
    violations = checker.check(context)

    assert any("license table" in v.message.lower() for v in violations)


def test_passes_when_report_and_license_refreshed(tmp_path: Path):
    """The policy passes when the report mentions the changed files."""
    repo = _setup_repo(tmp_path)
    report = repo / "THIRD_PARTY_LICENSES.md"
    report.write_text(
        "# Third-Party Licenses\n\n## License Report\n- requirements.lock updated\n",
        encoding="utf-8",
    )
    # Create a new license snapshot
    new_license = repo / "licenses" / "example.txt"
    new_license.write_text("MIT\n", encoding="utf-8")

    checker = DependencyLicenseSyncCheck()
    context = CheckContext(
        repo_root=repo,
        changed_files=[
            repo / "requirements.lock",
            report,
            new_license,
        ],
    )
    violations = checker.check(context)

    assert violations == []


def test_report_mentions_all_changed_files(tmp_path: Path):
    """Each dependency file needs a report line that cites it."""
    repo = _setup_repo(tmp_path)
    report = repo / "THIRD_PARTY_LICENSES.md"
    report.write_text(
        "# Third-Party Licenses\n\n## License Report\n- requirements.in added\n",
        encoding="utf-8",
    )

    checker = DependencyLicenseSyncCheck()
    context = CheckContext(
        repo_root=repo,
        changed_files=[repo / "requirements.lock", report, repo / "licenses" / "BSD-3-Clause.txt"],
    )
    violations = checker.check(context)

    assert any("requirements.lock" in v.message for v in violations)
