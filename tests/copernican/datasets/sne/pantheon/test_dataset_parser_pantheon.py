"""Parser test for the Pantheon+SH0ES supernova sample."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path

import numpy as numpy_module


class TestPantheonParser(unittest.TestCase):
    """Exercise the bundled Pantheon+SH0ES parser."""

    def test_loader_returns_sorted_covariance_sample(self) -> None:
        parser_module = importlib.import_module(
            "copernican.datasets.sne.pantheon.dataset_parser_pantheon"
        )
        data_dir = (
            Path(__file__).resolve().parents[5]
            / "copernican"
            / "datasets"
            / "sne"
            / "pantheon"
        )
        self.assertTrue(callable(parser_module.parse_pantheon_plus))
        self.assertEqual(
            parser_module.parse_pantheon_plus.__name__,
            "parse_pantheon_plus",
        )
        sne_dataframe = parser_module.parse_pantheon_plus(str(data_dir))
        self.assertIsNotNone(sne_dataframe)
        self.assertEqual(len(sne_dataframe), 1701)
        self.assertIn("zcmb", sne_dataframe.columns)
        self.assertIn("mu_obs", sne_dataframe.columns)
        self.assertIn("e_mu_obs", sne_dataframe.columns)
        cov_inv = sne_dataframe.attrs.get("covariance_matrix_inv")
        self.assertIsNotNone(cov_inv)
        self.assertEqual(cov_inv.shape, (1701, 1701))
        self.assertIn("diag_errors_for_plot", sne_dataframe.attrs)
        self.assertEqual(
            len(sne_dataframe.attrs["diag_errors_for_plot"]),
            1701,
        )
        self.assertTrue(
            numpy_module.all(
                numpy_module.diff(sne_dataframe["zcmb"].to_numpy(dtype=float))
                >= 0
            )
        )


if __name__ == "__main__":
    unittest.main()
