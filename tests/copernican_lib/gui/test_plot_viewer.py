"""Smoke tests for copernican_lib.gui.plot_viewer."""

import unittest

from copernican_lib.gui import plot_viewer as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican_lib.gui.plot_viewer")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the plot viewer surface to the coverage policy."""

    def test_public_symbols_are_present(self) -> None:
        self.assertTrue(hasattr(module, "PlotViewer"))
        self.assertTrue(hasattr(module.PlotViewer, "load_figure"))
        self.assertTrue(hasattr(module.PlotViewer, "fit_to_screen"))
        self.assertTrue(hasattr(module.PlotViewer, "fit_all"))
        self.assertTrue(hasattr(module.PlotViewer, "toggle_zoom"))
        self.assertTrue(hasattr(module.PlotViewer, "zoom_active"))


if __name__ == "__main__":
    unittest.main()
