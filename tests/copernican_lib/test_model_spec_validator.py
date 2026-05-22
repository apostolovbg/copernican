"""Smoke tests for copernican_lib.model_spec_validator."""

import unittest

from copernican_lib import model_spec_validator as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican_lib.model_spec_validator"
        )


if __name__ == "__main__":
    unittest.main()
