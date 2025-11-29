# See LICENSE.md in the repository root for details.

"""Tests for the dependency scan cache used by ``copernican.py``."""

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


class DependencyCacheTestCase(unittest.TestCase):
    """Guard dependency caching when optional wheels are unavailable."""

    skip_reason = (
        "Dependency cache tests require Python 3.12+ for optional wheels."
    )

    @classmethod
    def setUpClass(cls) -> None:  # pragma: no cover - environment guard
        if sys.version_info < (3, 12):
            raise unittest.SkipTest(cls.skip_reason)

        if "copernican" in sys.modules:
            cls.copernican = sys.modules["copernican"]
            return

        with mock.patch("sys.version_info", (3, 12, 0)):
            with mock.patch.dict(
                os.environ,
                {
                    "VIRTUAL_ENV": str(
                        Path(__file__).resolve().parents[1] / ".venv"
                    )
                },
                clear=False,
            ):
                cls.copernican = importlib.import_module("copernican")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cache_dir = Path(self.tmp.name) / "cache"
        patcher = mock.patch.dict(
            os.environ, {"COPERNICAN_DEP_CACHE_DIR": str(self.cache_dir)}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.source_dir = Path(self.tmp.name) / "src"
        self.source_dir.mkdir(parents=True, exist_ok=True)
        (self.source_dir / "demo.py").write_text(
            "import numpy\nimport pandas\n",
            encoding="utf-8",
        )
        self.search_dirs = [str(self.source_dir)]

    def test_cached_scan_skips_ast_parse(self) -> None:
        """The second scan should load imports from the cache."""

        scan = (
            self.copernican._gather_required_packages
        )  # pylint: disable=W0212
        first = scan(search_dirs=self.search_dirs)
        self.assertIn("numpy", first)
        self.assertIn("pandas", first)
        with mock.patch(
            "copernican.ast.parse",
            side_effect=AssertionError("AST parsing should be skipped"),
        ):
            second = scan(search_dirs=self.search_dirs)
        self.assertEqual(first, second)


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    unittest.main()
