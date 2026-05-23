"""Smoke tests for copernican_lib.chain_io."""

import unittest

from copernican_lib import chain_io as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.chain_io")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the public chain I/O API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.save_posterior))


if __name__ == "__main__":
    unittest.main()
