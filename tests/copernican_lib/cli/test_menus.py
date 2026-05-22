"""Smoke tests for copernican_lib.cli.menus."""

import unittest

from copernican_lib.cli import menus as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.cli.menus")


if __name__ == "__main__":
    unittest.main()
