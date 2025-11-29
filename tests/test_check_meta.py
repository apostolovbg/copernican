"""Tests for :mod:`tools.check_meta`."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from tools import check_meta

# Capture the validator's notion of "today" so every scenario runs
# against the same UTC-normalised reference date that
# :func:`check_meta.validate_metadata` would use without an override.
_REFERENCE_DATE = check_meta._utc_today()


def test_validate_metadata_current_repo() -> None:
    """The repository metadata should already satisfy the checker."""

    errors = check_meta.validate_metadata(today=_REFERENCE_DATE)
    assert errors == []


def test_validate_metadata_reports_discrepancies(tmp_path: Path) -> None:
    """The validator should report drifted versions and future timestamps."""

    base = tmp_path
    (base / "copernican_lib").mkdir(parents=True)
    (base / "copernican_lib" / "VERSION").write_text(
        "1.0.0\n", encoding="utf-8"
    )

    (base / "README.md").write_text(
        "**Version:** 2.0.0\n**Last Updated:** 2099-01-01\n",
        encoding="utf-8",
    )
    (base / "CHANGELOG.md").write_text(
        "# Changelog\n**Last Updated:** 2099-01-01\n",
        encoding="utf-8",
    )

    docs_dir = base / "docs"
    docs_dir.mkdir()
    (docs_dir / "page.md").write_text(
        "# Doc Page\n**Last Updated:** 2099-01-01\n",
        encoding="utf-8",
    )

    (base / "CITATION.cff").write_text(
        '# Last Updated: 2099-01-01\nversion: "2.0.0"\n'
        'preferred-citation:\n  version: "2.0.0"\n',
        encoding="utf-8",
    )

    errors = check_meta.validate_metadata(
        base_path=base, today=_REFERENCE_DATE
    )

    assert any("README.md records version" in error for error in errors)
    assert any("CITATION.cff version fields" in error for error in errors)
    assert any(
        "README.md carries future timestamp" in error for error in errors
    )
    assert any(
        "docs/page.md carries future timestamp" in error for error in errors
    )


def test_validate_metadata_flags_late_last_updated(tmp_path: Path) -> None:
    """Markers appearing after the third line should fail validation."""

    base = tmp_path
    (base / "copernican_lib").mkdir(parents=True)
    (base / "copernican_lib" / "VERSION").write_text(
        "1.0.0\n", encoding="utf-8"
    )
    (base / "README.md").write_text(
        "**Version:** 1.0.0\n\n\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )
    (base / "CHANGELOG.md").write_text(
        "# Changelog\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )

    errors = check_meta.validate_metadata(
        base_path=base, today=_REFERENCE_DATE
    )

    assert any("first three lines" in error for error in errors)


def test_validate_metadata_accepts_third_line_marker(
    tmp_path: Path,
) -> None:
    """A marker on the third line should pass validation."""

    base = tmp_path
    (base / "copernican_lib").mkdir(parents=True)
    (base / "copernican_lib" / "VERSION").write_text(
        "1.0.0\n", encoding="utf-8"
    )
    (base / "README.md").write_text(
        "**Version:** 1.0.0\nHeading\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )
    (base / "CHANGELOG.md").write_text(
        "# Changelog\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )

    errors = check_meta.validate_metadata(
        base_path=base, today=_REFERENCE_DATE
    )

    assert errors == []


def test_validate_metadata_default_uses_utc(
    monkeypatch, tmp_path: Path
) -> None:
    """The default date source should rely on a UTC-normalised clock."""

    base = tmp_path
    (base / "copernican_lib").mkdir(parents=True)
    (base / "copernican_lib" / "VERSION").write_text(
        "1.0.0\n", encoding="utf-8"
    )
    (base / "README.md").write_text(
        "**Version:** 1.0.0\n**Last Updated:** 2099-01-01\n",
        encoding="utf-8",
    )
    (base / "CHANGELOG.md").write_text(
        "# Changelog\n**Last Updated:** 2099-01-01\n",
        encoding="utf-8",
    )
    (base / "CITATION.cff").write_text(
        '# Last Updated: 2099-01-01\nversion: "1.0.0"\n'
        'preferred-citation:\n  version: "1.0.0"\n',
        encoding="utf-8",
    )

    sentinel = _dt.date(2100, 1, 1)
    monkeypatch.setattr(check_meta, "_utc_today", lambda: sentinel)

    errors = check_meta.validate_metadata(base_path=base)

    assert all("future timestamp" not in error for error in errors)
