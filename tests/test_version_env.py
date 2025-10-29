"""Tests for version overrides via environment variables."""

from __future__ import annotations

import os
import unittest

from copernican_lib.version import get_version


class VersionEnvTest(unittest.TestCase):
    """Ensure COPERNICAN_VERSION takes precedence over SCM lookups."""

    def test_env_override(self) -> None:
        """get_version returns the environment value when set."""

        test_version = "1.2.3-custom"
        old = os.environ.get("COPERNICAN_VERSION")
        os.environ["COPERNICAN_VERSION"] = test_version
        try:
            self.assertEqual(get_version(), test_version)
        finally:
            if old is None:
                os.environ.pop("COPERNICAN_VERSION", None)
            else:
                os.environ["COPERNICAN_VERSION"] = old


if __name__ == "__main__":
    unittest.main()
