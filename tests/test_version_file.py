import os
from pathlib import Path
from unittest import TestCase, mock

from copernican_lib import version as version_module

class VersionFileTests(TestCase):
    """Verify the runtime version is sourced from the tracked VERSION file."""

    def test_get_version_prefers_version_file_before_scm(self) -> None:
        version_path = Path(version_module.__file__).with_name("VERSION")
        expected = version_path.read_text(encoding="utf-8").strip()
        self.assertTrue(expected)
        original_env = os.environ.pop("COPERNICAN_VERSION", None)
        try:
            missing = version_module.PackageNotFoundError("copernican-suite")
            with mock.patch.object(
                version_module, "version", side_effect=missing
            ):
                with mock.patch.object(
                    version_module,
                    "scm_get_version",
                    side_effect=AssertionError(
                        "setuptools_scm should not run"
                    ),
                ):
                    observed = version_module.get_version()
        finally:
            if original_env is not None:
                os.environ["COPERNICAN_VERSION"] = original_env
        self.assertEqual(observed, expected)
