"""Smoke tests for copernican.validation.runner."""

import unittest

from copernican.validation import runner as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.validation.runner")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "discover_manifests"))
        self.assertTrue(hasattr(module, "run_validation_suite"))


if __name__ == "__main__":
    unittest.main()
