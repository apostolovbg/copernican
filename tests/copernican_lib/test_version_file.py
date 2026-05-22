import os
from pathlib import Path
from unittest import TestCase

from copernican_lib import version as version_module


class VersionFileTests(TestCase):
    """Verify the runtime version is sourced from the tracked VERSION file."""

    def test_version_file_used_when_env_missing(self) -> None:
        version_path = Path(version_module.__file__).with_name("VERSION")
        expected = version_path.read_text(encoding="utf-8").strip()
        self.assertTrue(expected)
        original_env = os.environ.pop("COPERNICAN_VERSION", None)
        try:
            observed = version_module.get_version()
        finally:
            if original_env is not None:
                os.environ["COPERNICAN_VERSION"] = original_env
        self.assertEqual(observed, expected)

    def test_env_variable_overrides_tracked_version(self) -> None:
        os.environ["COPERNICAN_VERSION"] = "99.99.99"
        try:
            self.assertEqual(version_module.get_version(), "99.99.99")
        finally:
            os.environ.pop("COPERNICAN_VERSION", None)
