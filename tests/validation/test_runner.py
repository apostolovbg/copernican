"""Smoke tests for validation.runner."""

import unittest

from validation import runner as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "validation.runner")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "discover_manifests"))
        self.assertTrue(hasattr(module, "run_validation_suite"))


if __name__ == "__main__":
    unittest.main()
