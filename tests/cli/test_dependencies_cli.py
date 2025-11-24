# Last Updated: 2025-11-24
"""Tests for CLI dependency handling helpers."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican_lib.cli import dependencies


class DependencyCacheTestCase(unittest.TestCase):
    """Guard dependency caching logic in the CLI helpers."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name) / "cache"
        self.search_root = Path(self.tmp.name) / "src"
        self.search_root.mkdir(parents=True, exist_ok=True)
        (self.search_root / "demo.py").write_text(
            "import numpy\nimport pandas\n",
            encoding="utf-8",
        )
        patcher = mock.patch.dict(
            os.environ, {dependencies.DEPENDENCY_CACHE_ENV_VAR: str(self.cache_dir)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cached_scan_skips_reparse(self) -> None:
        """A second scan loads import names from the cache file."""

        search_dirs = [str(self.search_root)]
        first = dependencies._gather_required_packages(search_dirs=search_dirs)
        self.assertIn("numpy", first)
        self.assertIn("pandas", first)
        with mock.patch(
            "copernican_lib.cli.dependencies.ast.parse",
            side_effect=AssertionError("AST should not be invoked on cache hit"),
        ):
            cached = dependencies._gather_required_packages(search_dirs=search_dirs)
        self.assertEqual(first, cached)


class CheckDependenciesPromptTestCase(unittest.TestCase):
    """Validate the interactive dependency installer flow."""

    @mock.patch("copernican_lib.cli.dependencies.Path")
    def test_auto_confirm_skips_prompt(self, path_mock) -> None:
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican_lib.cli.dependencies._gather_required_packages",
                return_value={"demo"},
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch("copernican_lib.cli.dependencies.console.ask") as ask_mock,
            mock.patch("subprocess.run") as run_mock,
            mock.patch("importlib.import_module"),
        ):
            dependencies.check_dependencies(auto_confirm=True)
            ask_mock.assert_not_called()
            run_mock.assert_called_once_with(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    unittest.main()
