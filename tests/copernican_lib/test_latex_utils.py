"""Smoke tests for copernican_lib.latex_utils."""

import unittest

from copernican_lib import latex_utils as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.latex_utils")


if __name__ == "__main__":
    unittest.main()
