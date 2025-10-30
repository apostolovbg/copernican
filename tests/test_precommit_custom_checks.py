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
