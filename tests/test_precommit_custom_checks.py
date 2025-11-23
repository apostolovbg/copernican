# Last Updated: 2025-11-23

"""Unit tests for the custom pre-commit policy checks."""

from __future__ import annotations

import datetime as dt
import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "precommit_custom_checks.py"
)
SPEC = importlib.util.spec_from_file_location(
    "copernican_precommit", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader  # for mypy/static analyzers
SPEC.loader.exec_module(MODULE)


def test_detect_future_dates_flags_future_timestamp(tmp_path) -> None:
    """Dates beyond the supplied ``today`` reference should trigger errors."""

    target = tmp_path / "README.md"
    target.write_text("**Last Updated:** 2099-01-01\n", encoding="utf-8")
    errors = MODULE._detect_future_dates(
        [target], dt.date(2025, 1, 1), root=tmp_path
    )
    assert errors
    assert "2099-01-01" in errors[0]


def test_check_last_updated_headers_flags_late_marker(tmp_path) -> None:
    """Markers beyond the third line should be rejected."""

    root = tmp_path
    target = root / "README.md"
    target.write_text(
        "**Version:** 1.0.0\n\n\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )
    errors = MODULE._check_last_updated_headers(root, [target])
    assert errors
    assert "first three lines" in errors[0]


def test_check_last_updated_headers_accepts_header(tmp_path) -> None:
    """Markers within the first three lines should pass."""

    root = tmp_path
    target = root / "README.md"
    target.write_text(
        "**Version:** 1.0.0\n**Last Updated:** 2025-01-01\n",
        encoding="utf-8",
    )
    errors = MODULE._check_last_updated_headers(root, [target])
    assert not errors


def test_check_version_sync_detects_mismatched_versions(tmp_path) -> None:
    """A mismatch between metadata files should surface an explicit error."""

    root = tmp_path
    (root / "copernican_lib").mkdir()
    (root / "copernican_lib" / "VERSION").write_text(
        "1.2.3\n", encoding="utf-8"
    )
    (root / "README.md").write_text("**Version:** 1.2.4\n", encoding="utf-8")
    citation = (
        "# Last Updated: 2025-10-30\n"
        "cff-version: 1.2.0\n"
        'version: "1.2.3"\n'
        "preferred-citation:\n"
        '  version: "1.2.3"\n'
    )
    (root / "CITATION.cff").write_text(citation, encoding="utf-8")
    errors = MODULE._check_version_sync(root)
    assert errors
    assert "README.md version" in errors[0]


def test_check_print_usage_blocks_direct_prints(tmp_path) -> None:
    """Direct ``print`` calls within library files must be rejected."""

    root = tmp_path
    (root / "copernican_lib").mkdir()
    offender = root / "copernican_lib" / "bad.py"
    offender.write_text("print('nope')\n", encoding="utf-8")
    errors = MODULE._check_print_usage(root, [offender])
    assert errors
    assert "copernican_lib/bad.py" in errors[0]


def test_check_print_usage_allows_console_module(tmp_path) -> None:
    """The console helper module is permitted to use ``print`` internally."""

    root = tmp_path
    (root / "copernican_lib").mkdir()
    console = root / "copernican_lib" / "console_output.py"
    console.write_text("print('allowed')\n", encoding="utf-8")
    errors = MODULE._check_print_usage(root, [console])
    assert not errors


def test_utc_today_requests_utc_clock(monkeypatch) -> None:
    """The helper should request the UTC timezone from datetime.now."""

    real_datetime = dt.datetime

    class _DummyDateTime:
        called = False

        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            assert tz is dt.timezone.utc
            cls.called = True
            return real_datetime(2025, 1, 1, tzinfo=dt.timezone.utc)

    monkeypatch.setattr(MODULE._dt, "datetime", _DummyDateTime)
    result = MODULE._utc_today()
    assert result == dt.date(2025, 1, 1)
    assert _DummyDateTime.called


def test_enforce_last_updated_freshness_requires_today(tmp_path) -> None:
    """Files changed today must refresh their ``Last Updated`` headers."""

    target = tmp_path / "README.md"
    target.write_text("**Last Updated:** 2025-01-01\n", encoding="utf-8")
    today = dt.date(2025, 1, 2)

    errors = MODULE._enforce_last_updated_freshness(tmp_path, [target], today)

    assert errors
    assert "2025-01-02" in errors[0]


def test_ensure_changelog_updated_demands_entry(tmp_path) -> None:
    """Any change outside the changelog should force a new entry."""

    tracked = tmp_path / "README.md"
    tracked.write_text("**Last Updated:** 2025-01-01\n", encoding="utf-8")
    (tmp_path / "CHANGELOG.md").write_text("# Changelog\n", encoding="utf-8")

    errors = MODULE._ensure_changelog_updated(tmp_path, [tracked])

    assert errors
    assert "CHANGELOG.md" in errors[0]


def test_new_modules_require_tests(tmp_path) -> None:
    """Adding modules should be paired with fresh or updated tests."""

    module_root = tmp_path / "copernican_lib"
    module_root.mkdir()
    module = module_root / "fresh.py"
    module.write_text("# Last Updated: 2025-01-02\n", encoding="utf-8")

    errors = MODULE._check_new_modules_have_tests(tmp_path, [module], [module])

    assert errors
    assert "tests/" in errors[0]

    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_fresh.py"
    test_file.write_text("# Last Updated: 2025-01-02\n", encoding="utf-8")

    errors = MODULE._check_new_modules_have_tests(
        tmp_path, [module], [module, test_file]
    )

    assert not errors
