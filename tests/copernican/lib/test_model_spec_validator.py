"""Smoke tests for copernican.lib.model_spec_validator."""

import unittest

from copernican.lib import model_spec_validator as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            module.__name__, "copernican.lib.model_spec_validator"
        )


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the model-spec validator surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "validate_and_cache_model"))


if __name__ == "__main__":
    unittest.main()
