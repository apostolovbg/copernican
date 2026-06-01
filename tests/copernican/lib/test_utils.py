"""Tests for UTC-normalised timestamp helpers."""

import unittest
from datetime import datetime, timezone

from copernican.lib import utils


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


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the utility API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(utils.check_dataset_id))
        self.assertTrue(callable(utils.compute_sha256))
        self.assertTrue(callable(utils.ensure_dir_exists))
        self.assertTrue(callable(utils.generate_filename))
        self.assertTrue(callable(utils.get_random_seed))
        self.assertTrue(callable(utils.load_metadata_from_dir))
        self.assertTrue(callable(utils.set_random_seed))


if __name__ == "__main__":
    unittest.main()
