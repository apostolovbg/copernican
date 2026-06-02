"""Smoke tests for copernican.lib.run_lifecycle."""

import unittest

from copernican.lib import run_lifecycle as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.run_lifecycle")

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "ManifestWorkspace"))
        self.assertTrue(hasattr(module, "create_manifest_workspace"))
        self.assertTrue(hasattr(module, "delete_manifest_workspace"))
        self.assertTrue(hasattr(module, "import_manifest_to_workspace"))
        self.assertTrue(hasattr(module, "finalize_run_workspace"))


if __name__ == "__main__":
    unittest.main()
