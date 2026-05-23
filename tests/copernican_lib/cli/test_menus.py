"""Smoke tests for copernican_lib.cli.menus."""

import inspect
import unittest

from copernican_lib.cli import menus as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.cli.menus")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Assert the module keeps its exported CLI menu surface."""

    def test_public_symbols_remain_available(self) -> None:
        self.assertTrue(hasattr(module, "show_splash_screen"))
        self.assertTrue(hasattr(module, "select_seed"))
        self.assertTrue(hasattr(module, "select_from_list"))
        self.assertTrue(hasattr(module, "normalise_failure_reasons"))
        self.assertTrue(hasattr(module, "prompt_stage1_retry"))

    def test_source_mentions_public_symbols(self) -> None:
        source = inspect.getsource(module)
        self.assertIn("show_splash_screen", source)
        self.assertIn("select_seed", source)
        self.assertIn("select_from_list", source)
        self.assertIn("normalise_failure_reasons", source)
        self.assertIn("prompt_stage1_retry", source)


if __name__ == "__main__":
    unittest.main()
