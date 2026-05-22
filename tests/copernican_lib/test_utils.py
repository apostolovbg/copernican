"""Tests for UTC-normalised timestamp helpers."""

import unittest
from datetime import datetime, timezone

from copernican_lib import utils


class TestUtils(unittest.TestCase):
    """Exercise UTC timestamp helpers."""

    def test_get_utc_now_returns_timezone_aware(self) -> None:
        current = utils.get_utc_now()
        self.assertIs(current.tzinfo, timezone.utc)

    def test_get_timestamp_uses_utc_reference(self) -> None:
        fixed = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
        original = utils.get_utc_now
        try:
            utils.get_utc_now = lambda: fixed  # type: ignore[assignment]
            self.assertEqual(utils.get_timestamp(), "20250102_030405")
        finally:
            utils.get_utc_now = original  # type: ignore[assignment]

    def test_get_timestamp_accepts_naive_utc_reference(self) -> None:
        naive = datetime(2030, 6, 1, 12, 0, 0)
        self.assertEqual(utils.get_timestamp(now=naive), "20300601_120000")


if __name__ == "__main__":
    unittest.main()
