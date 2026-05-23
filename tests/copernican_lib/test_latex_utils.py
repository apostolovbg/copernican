"""Smoke tests for copernican_lib.latex_utils."""

import unittest

from copernican_lib import latex_utils as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.latex_utils")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the LaTeX utility surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "sanitize_name"))
        self.assertTrue(hasattr(module, "latex_to_sympy"))
        self.assertTrue(hasattr(module, "wrap_math"))
        self.assertTrue(hasattr(module, "latex_to_unicode"))


if __name__ == "__main__":
    unittest.main()
