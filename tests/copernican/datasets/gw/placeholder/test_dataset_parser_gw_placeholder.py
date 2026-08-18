"""Parser test for the gravitational-wave placeholder."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


class TestGWPlaceholderParser(unittest.TestCase):
    """Exercise the bundled placeholder parser."""

    def test_loader_reports_placeholder(self) -> None:
        parser_module = importlib.import_module(
            "copernican.datasets.gw.placeholder."
            "dataset_parser_gw_placeholder"
        )
        self.assertTrue(callable(parser_module.parse_gw_placeholder))
        self.assertEqual(
            parser_module.parse_gw_placeholder.__name__,
            "parse_gw_placeholder",
        )
        with self.assertLogs(level="INFO") as log_capture:
            data_dir = (
                Path(__file__).resolve().parents[5]
                / "copernican"
                / "datasets"
                / "gw"
                / "placeholder"
            )
            placeholder_result = parser_module.parse_gw_placeholder(
                str(data_dir)
            )
        self.assertIsNone(placeholder_result)
        self.assertIn(
            "Feature not implemented yet",
            "".join(log_capture.output),
        )


if __name__ == "__main__":
    unittest.main()
