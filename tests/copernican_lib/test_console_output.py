"""Smoke tests for copernican_lib.console_output."""

import unittest

from copernican_lib import console_output as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.console_output")


if __name__ == "__main__":
    unittest.main()
