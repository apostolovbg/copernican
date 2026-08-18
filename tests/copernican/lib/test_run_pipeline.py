"""Smoke tests for copernican.lib.run_pipeline."""

import unittest

from copernican.lib import run_pipeline as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.run_pipeline")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the run pipeline API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.execute_run_pipeline))
        self.assertTrue(callable(module.extract_model_param_vector))
        self.assertTrue(callable(module.resolve_sampler_function))


if __name__ == "__main__":
    unittest.main()
