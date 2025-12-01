"""Tests for UTC-normalised timestamp helpers.

"""

from __future__ import annotations

from datetime import datetime, timezone

from copernican_lib import utils


def test_get_utc_now_returns_timezone_aware() -> None:
    """The helper should provide an aware datetime in UTC."""

    current = utils.get_utc_now()
    assert current.tzinfo is timezone.utc


def test_get_timestamp_uses_utc_reference(monkeypatch) -> None:
    """Timestamps should rely on the UTC clock regardless of locale."""

    fixed = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    monkeypatch.setattr(utils, "get_utc_now", lambda: fixed)
    assert utils.get_timestamp() == "20250102_030405"


def test_get_timestamp_accepts_naive_utc_reference() -> None:
    """Naive inputs are treated as already-normalised UTC values."""

    naive = datetime(2030, 6, 1, 12, 0, 0)
    assert utils.get_timestamp(now=naive) == "20300601_120000"
