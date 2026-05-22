"""Smoke tests for copernican_lib.likelihoods._protocol."""

import unittest

from copernican_lib.likelihoods import _protocol as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican_lib.likelihoods._protocol"
        )


if __name__ == "__main__":
    unittest.main()
