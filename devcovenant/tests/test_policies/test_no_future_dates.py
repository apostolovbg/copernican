"""Tests for no_future_dates policy."""

import datetime as dt
import tempfile
import unittest
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.fixers.no_future_dates import NoFutureDatesFixer
from devcovenant.policy_scripts.no_future_dates import NoFutureDatesCheck


class TestNoFutureDatesPolicy(unittest.TestCase):
    """Test suite for NoFutureDatesCheck."""

    def test_detects_future_dates(self):
        """Policy should detect future dates in Last Updated headers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # future = (dt.date.today() + dt.timedelta(days=1)).isoformat()

            test_file = repo_root / "test.md"
            future_date = (
                dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=1)
            ).isoformat()
            test_file.write_text(f"**Last Updated:** {future_date}\n")

            context = CheckContext(repo_root=repo_root, all_files=[test_file])
            policy = NoFutureDatesCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 1)
            self.assertIn("future date", violations[0].message.lower())

    def test_allows_current_dates(self):
        """Policy should allow current dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            # today = dt.date.today().isoformat()

            test_file = repo_root / "test.md"
            today_date = dt.datetime.now(dt.timezone.utc).date().isoformat()
            test_file.write_text(f"**Last Updated:** {today_date}\n")

            context = CheckContext(repo_root=repo_root, all_files=[test_file])
            policy = NoFutureDatesCheck()
            violations = policy.check(context)

            self.assertEqual(len(violations), 0)

    def test_auto_fix_replaces_future_date(self):
        """Auto-fix brings a future timestamp back to today."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            test_file = repo_root / "CHANGELOG.md"
            future_date = (
                dt.datetime.now(dt.timezone.utc).date() + dt.timedelta(days=7)
            ).isoformat()
            test_file.write_text(f"Last Updated: {future_date}\n")

            context = CheckContext(repo_root=repo_root, all_files=[test_file])
            violations = NoFutureDatesCheck().check(context)
            self.assertEqual(len(violations), 1)
            fixer = NoFutureDatesFixer()
            result = fixer.fix(violations[0])
            self.assertTrue(result.success)
            updated = test_file.read_text()
            self.assertNotIn(future_date, updated)
