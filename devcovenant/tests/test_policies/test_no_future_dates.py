"""Tests for no_future_dates policy."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from devcovenant.policy_scripts.no_future_dates import NoFutureDatesPolicy


class TestNoFutureDatesPolicy(unittest.TestCase):
    """Test suite for NoFutureDatesPolicy."""

    def test_detects_future_dates(self):
        """Policy should detect future dates in Last Updated headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            future_date = (dt.date.today() + dt.timedelta(days=1)).isoformat()

            test_file = repo_root / "test.md"
            test_file.write_text(f"# Last Updated: {future_date}\n")

            policy = NoFutureDatesPolicy(repo_root)
            violations = policy.check([test_file])

            self.assertEqual(len(violations), 1)
            self.assertIn("future date", violations[0].message.lower())

    def test_allows_current_dates(self):
        """Policy should allow current dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            today = dt.date.today().isoformat()

            test_file = repo_root / "test.md"
            test_file.write_text(f"# Last Updated: {today}\n")

            policy = NoFutureDatesPolicy(repo_root)
            violations = policy.check([test_file])

            self.assertEqual(len(violations), 0)
