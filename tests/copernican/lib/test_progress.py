"""Smoke tests for copernican.lib.progress."""

import unittest

from copernican.lib import progress as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.progress")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "BatchProgressBar"))
        self.assertTrue(hasattr(module.BatchProgressBar, "start_batch"))
        self.assertTrue(hasattr(module.BatchProgressBar, "start_step"))
        self.assertTrue(hasattr(module.BatchProgressBar, "update"))
        self.assertTrue(hasattr(module.BatchProgressBar, "finish_batch"))
        self.assertTrue(hasattr(module.BatchProgressBar, "suspend_display"))


if __name__ == "__main__":
    unittest.main()
