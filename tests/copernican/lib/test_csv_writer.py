"""Smoke tests for copernican.lib.csv_writer."""

import tempfile
import unittest
from pathlib import Path

import numpy
import pandas

from copernican.lib import csv_writer as module


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(module.__name__, "copernican.lib.csv_writer")


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the CSV writer API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(module.save_bao_results_csv))
        self.assertTrue(callable(module.save_cmb_results_csv))
        self.assertTrue(callable(module.save_sne_results_detailed_csv))


class SNeCsvWriterRegressionTestCase(unittest.TestCase):
    """Exercise the Union-style SNe residual correction path."""

    def test_save_sne_results_detailed_csv_marginalizes_intercept(
        self,
    ) -> None:
        class DummyPlugin:
            """Minimal plugin exposing the API the CSV writer expects."""

            MODEL_NAME = "Dummy Model"

            @staticmethod
            def distance_modulus_model(z_values, *params):
                baseline = numpy.array([10.0, 11.0, 12.0], dtype=float)
                return baseline[: z_values.shape[0]].copy()

        observations = pandas.DataFrame(
            {
                "Name": ["SN1", "SN2", "SN3"],
                "zcmb": [0.1, 0.2, 0.3],
                "mu_obs": [10.5, 11.5, 12.5],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        observations.attrs["dataset_id"] = "union3_2025"
        observations.attrs["requires_sne_intercept_marginalization"] = True
        observations.attrs["covariance_matrix_inv"] = numpy.eye(3)

        fit_results = {
            "success": True,
            "fitted_cosmological_params": {"H0": 70.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_sne_results_detailed_csv(
                observations,
                fit_results,
                fit_results,
                DummyPlugin,
                DummyPlugin,
                csv_dir=tmpdir,
                timestamp="20260528_000000",
            )

            csv_files = list(Path(tmpdir).glob("*.csv"))
            self.assertEqual(len(csv_files), 1)

            output_df = pandas.read_csv(csv_files[0])
            self.assertTrue(
                numpy.allclose(output_df["residual_lcdm"].to_numpy(), 0.0)
            )
            alt_residual_columns = [
                col for col in output_df.columns if col.startswith("residual_")
            ]
            self.assertGreaterEqual(len(alt_residual_columns), 2)
            alt_column = next(
                col for col in alt_residual_columns if col != "residual_lcdm"
            )
            self.assertTrue(
                numpy.allclose(output_df[alt_column].to_numpy(), 0.0)
            )


if __name__ == "__main__":
    unittest.main()
