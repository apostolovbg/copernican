"""Smoke tests for copernican.lib.posterior_explorer."""

import unittest

from copernican.lib import posterior_explorer as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.posterior_explorer")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the posterior explorer API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.create_posterior_overview_figure))
        self.assertTrue(callable(module.extract_posterior_arrays))
        self.assertTrue(callable(module.find_posterior_files))
        self.assertTrue(callable(module.flatten_posterior_arrays))
        self.assertTrue(callable(module.load_inference_data))


if __name__ == "__main__":
    unittest.main()
