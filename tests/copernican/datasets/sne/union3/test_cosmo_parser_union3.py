"""Basic smoke test for the Union3 parser."""

from __future__ import annotations

import importlib
import tempfile
import unittest
from pathlib import Path

import numpy as numpy_module
from astropy.io import fits

from copernican_lib import dataset_registry


class TestUnion3Parser(unittest.TestCase):
    """Exercise the Union3 dataset loader."""

    def test_union3_loader_returns_compressed_sample(self) -> None:
        importlib.import_module(
            "copernican.datasets.sne.union3.cosmo_parser_union3"
        )
        union3_dataframe = dataset_registry.load_sne_data("union3_2025")
        self.assertFalse(union3_dataframe.empty)
        self.assertEqual(union3_dataframe.shape[0], 22)
        self.assertIn("zcmb", union3_dataframe.columns)
        self.assertIn("mu_obs", union3_dataframe.columns)
        self.assertIn("covariance_matrix_inv", union3_dataframe.attrs)
        self.assertTrue(
            union3_dataframe.attrs.get(
                "requires_sne_intercept_marginalization"
            )
        )
        self.assertEqual(
            union3_dataframe.attrs.get("sne_intercept_name"),
            "Delta_mu",
        )
        self.assertIn("sne_intercept_reason", union3_dataframe.attrs)
        inv_covariance = union3_dataframe.attrs["covariance_matrix_inv"]
        self.assertEqual(inv_covariance.shape, (22, 22))
        diag_errors = union3_dataframe.attrs.get("diag_errors_for_plot")
        self.assertIsNotNone(diag_errors)
        self.assertTrue(
            numpy_module.allclose(
                union3_dataframe["e_mu_obs"].values,
                diag_errors,
            )
        )
        parser_module = importlib.import_module(
            "copernican.datasets.sne.union3.cosmo_parser_union3"
        )
        self.assertTrue(callable(parser_module.looks_like_mu_mat))
        self.assertTrue(
            parser_module.looks_like_mu_mat("mu_mat_union3_cosmo=2_mu.fits")
        )
        self.assertFalse(parser_module.looks_like_mu_mat("union3.txt"))

    def test_parse_union3_preserves_documented_layout(self) -> None:
        redshifts = numpy_module.array([0.11, 0.22, 0.33], dtype=float)
        mu_values = numpy_module.array([34.1, 35.2, 36.3], dtype=float)
        inv_covariance = numpy_module.array(
            [
                [2.0, 0.1, 0.0],
                [0.1, 2.5, 0.2],
                [0.0, 0.2, 3.0],
            ],
            dtype=float,
        )
        matrix = numpy_module.zeros((4, 4), dtype=float)
        matrix[0, 1:] = redshifts
        matrix[1:, 0] = mu_values
        matrix[1:, 1:] = inv_covariance

        with tempfile.TemporaryDirectory() as temporary_dir:
            fits_path = Path(temporary_dir) / "mu_mat_union3_test.fits"
            fits.PrimaryHDU(matrix).writeto(fits_path)
            parser_module = importlib.import_module(
                "copernican.datasets.sne.union3.cosmo_parser_union3"
            )
            self.assertEqual(
                parser_module.looks_like_mu_mat.__name__,
                "looks_like_mu_mat",
            )
            union3_dataframe = parser_module.parse_union3(temporary_dir)

        self.assertIsNotNone(union3_dataframe)
        self.assertTrue(
            union3_dataframe.attrs["requires_sne_intercept_marginalization"]
        )
        self.assertEqual(
            union3_dataframe.attrs["sne_intercept_name"],
            "Delta_mu",
        )
        self.assertIn("sne_intercept_reason", union3_dataframe.attrs)
        self.assertIn("covariance_matrix_inv", union3_dataframe.attrs)
        self.assertIn("redshift_nodes", union3_dataframe.attrs)
        self.assertIn("mu_matrix_path", union3_dataframe.attrs)
        self.assertEqual(
            len(union3_dataframe),
            len(union3_dataframe.attrs["redshift_nodes"]),
        )
        self.assertTrue(
            numpy_module.allclose(
                union3_dataframe["zcmb"].to_numpy(dtype=float),
                redshifts,
            )
        )
        self.assertTrue(
            numpy_module.allclose(
                union3_dataframe.attrs["redshift_nodes"],
                redshifts,
            )
        )
        self.assertEqual(
            union3_dataframe.attrs["covariance_matrix_inv"].shape,
            (len(union3_dataframe), len(union3_dataframe)),
        )
        self.assertEqual(
            Path(union3_dataframe.attrs["mu_matrix_path"]),
            fits_path,
        )
        self.assertTrue(
            numpy_module.allclose(
                union3_dataframe["mu_obs"].to_numpy(dtype=float),
                mu_values,
            )
        )


if __name__ == "__main__":
    unittest.main()
