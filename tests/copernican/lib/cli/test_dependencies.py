"""Tests for CLI dependency handling helpers."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican.lib.cli import dependencies


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
            os.environ,
            {dependencies.DEPENDENCY_CACHE_ENV_VAR: str(self.cache_dir)},
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
            "copernican.lib.cli.dependencies.ast.parse",
            side_effect=AssertionError(
                "AST should not be invoked on cache hit"
            ),
        ):
            cached = dependencies._gather_required_packages(
                search_dirs=search_dirs
            )
        self.assertEqual(first, cached)

    def test_relative_imports_are_ignored(self) -> None:
        """Relative imports should not be flagged as missing dependencies."""

        package_root = Path(self.tmp.name) / "pkg"
        package_root.mkdir(parents=True, exist_ok=True)
        (package_root / "__init__.py").write_text("", encoding="utf-8")
        (package_root / "shared.py").write_text("pass\n", encoding="utf-8")
        (package_root / "module.py").write_text(
            "from . import shared\n", encoding="utf-8"
        )

        required = dependencies._gather_required_packages(
            search_dirs=[str(package_root)]
        )

        self.assertNotIn("shared", required)


class CheckDependenciesPromptTestCase(unittest.TestCase):
    """Validate the dependency guard after auto-install was removed."""

    @mock.patch("copernican.lib.cli.dependencies.Path")
    def test_missing_dependencies_exit_with_instruction(
        self, path_mock
    ) -> None:
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican.lib.cli.dependencies._gather_required_packages",
                return_value={"demo"},
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch(
                "copernican.lib.cli.dependencies.console.write"
            ) as write,
            mock.patch("subprocess.run") as run_mock,
        ):
            with self.assertRaises(SystemExit):
                dependencies.check_dependencies()
            write.assert_called()
            self.assertIn("python -m copernican", write.call_args[0][0])
            run_mock.assert_not_called()


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the CLI dependency API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(dependencies, "RuntimeOptions"))
        self.assertTrue(callable(dependencies.get_runtime_options))
        self.assertTrue(callable(dependencies.run_startup_tests))
        self.assertTrue(callable(dependencies.load_third_party_modules))


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    unittest.main()
