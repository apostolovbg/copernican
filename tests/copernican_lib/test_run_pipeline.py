"""Smoke tests for copernican_lib.run_pipeline."""

import unittest

from copernican_lib import run_pipeline as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.run_pipeline")


if __name__ == "__main__":
    unittest.main()
