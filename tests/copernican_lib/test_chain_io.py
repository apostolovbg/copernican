"""Smoke tests for copernican_lib.chain_io."""

import unittest

from copernican_lib import chain_io as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.chain_io")


if __name__ == "__main__":
    unittest.main()
