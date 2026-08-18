"""Parser test for the compound BAO compilation."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


class TestCompoundParser(unittest.TestCase):
    """Exercise the bundled compound BAO parser."""

    def test_loader_returns_bao_compilation(self) -> None:
        parser_module = importlib.import_module(
            "copernican.datasets.bao.compound.dataset_parser_compound"
        )
        data_dir = (
            Path(__file__).resolve().parents[5]
            / "copernican"
            / "datasets"
            / "bao"
            / "compound"
        )
        self.assertTrue(callable(parser_module.parse_bao))
        self.assertEqual(parser_module.parse_bao.__name__, "parse_bao")
        bao_dataframe = parser_module.parse_bao(str(data_dir))
        self.assertIsNotNone(bao_dataframe)
        self.assertEqual(len(bao_dataframe), 25)
        self.assertIn("redshift", bao_dataframe.columns)
        self.assertIn("observable_type", bao_dataframe.columns)
        self.assertIn("value", bao_dataframe.columns)
        self.assertIn("error", bao_dataframe.columns)
        self.assertNotIn("covariance_matrix_inv", bao_dataframe.attrs)
        self.assertEqual(bao_dataframe.attrs["covariance_model"], "diagonal")
        self.assertGreater(len(bao_dataframe["observable_type"].unique()), 1)


if __name__ == "__main__":
    unittest.main()
