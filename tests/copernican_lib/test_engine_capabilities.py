"""Smoke tests for copernican_lib.engine_capabilities."""

import unittest

from copernican_lib import engine_capabilities as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.engine_capabilities")


if __name__ == "__main__":
    unittest.main()
