"""Parser test for the JLA 2014 supernova sample."""

from __future__ import annotations

import importlib
import unittest

import numpy as numpy_module

from copernican_lib import dataset_registry


class TestJLA2014Parser(unittest.TestCase):
    """Exercise the bundled JLA 2014 parser."""

    def test_loader_returns_projected_covariance_sample(self) -> None:
        parser_module = importlib.import_module(
            "data.sne.jla2014.cosmo_parser_jla2014"
        )
        self.assertTrue(callable(parser_module.parse_jla2014))
        self.assertEqual(
            parser_module.parse_jla2014.__name__,
            "parse_jla2014",
        )
        sne_dataframe = dataset_registry.load_sne_data("jla_2014")
        self.assertIsNotNone(sne_dataframe)
        self.assertEqual(len(sne_dataframe), 740)
        self.assertIn("zcmb", sne_dataframe.columns)
        self.assertIn("mu_obs", sne_dataframe.columns)
        self.assertIn("e_mu_obs", sne_dataframe.columns)
        cov_inv = sne_dataframe.attrs.get("covariance_matrix_inv")
        self.assertIsNotNone(cov_inv)
        self.assertEqual(cov_inv.shape, (740, 740))
        self.assertIn("diag_errors_for_plot", sne_dataframe.attrs)
        self.assertEqual(len(sne_dataframe.attrs["diag_errors_for_plot"]), 740)
        self.assertTrue(
            numpy_module.all(
                numpy_module.diff(sne_dataframe["zcmb"].to_numpy(dtype=float))
                >= 0
            )
        )


if __name__ == "__main__":
    unittest.main()
