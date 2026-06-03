"""Smoke tests for the package CLI entrypoint module."""

import unittest
from unittest import mock

import copernican.cli as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.cli")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the CLI surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "main"))

    @mock.patch.object(module, "workflow_main")
    def test_main_delegates_to_workflow_main(self, workflow_main):
        workflow_main.return_value = 0
        self.assertEqual(module.main([]), 0)
        workflow_main.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
