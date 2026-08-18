"""Smoke tests for copernican.lib.csv_writer."""

import tempfile
import unittest
from pathlib import Path

import numpy
import pandas

from copernican.lib import csv_writer as module
from copernican.lib.model_selection import build_comparison_request


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
        class ReferencePlugin:
            """Minimal plugin exposing the API the CSV writer expects."""

            MODEL_NAME = "Reference Model"

            @staticmethod
            def distance_modulus_model(z_values, *params):
                baseline = numpy.array([10.0, 11.0, 12.0], dtype=float)
                return baseline[: z_values.shape[0]].copy()

        class CandidatePlugin(ReferencePlugin):
            """Second model role with the same deterministic prediction."""

            MODEL_NAME = "Candidate Model"

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
            "fitted_model_params": {"H0": 70.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_sne_results_detailed_csv(
                observations,
                fit_results,
                fit_results,
                ReferencePlugin,
                CandidatePlugin,
                csv_dir=tmpdir,
                timestamp="20260528_000000",
                comparison=build_comparison_request(
                    "Reference Model",
                    "Candidate Model",
                ),
            )

            csv_files = list(Path(tmpdir).glob("*.csv"))
            self.assertEqual(len(csv_files), 1)

            output_df = pandas.read_csv(csv_files[0])
            self.assertTrue(
                numpy.allclose(
                    output_df["residual_Reference_Model"].to_numpy(),
                    0.0,
                )
            )
            residual_columns = [
                col for col in output_df.columns if col.startswith("residual_")
            ]
            self.assertEqual(
                residual_columns,
                [
                    "residual_Reference_Model",
                    "residual_Candidate_Model",
                ],
            )
            self.assertTrue(
                numpy.allclose(
                    output_df["residual_Candidate_Model"].to_numpy(),
                    0.0,
                )
            )


class CMBCsvWriterRegressionTestCase(unittest.TestCase):
    """Exercise exact long-form CMB output serialization."""

    def test_long_form_cmb_csv_preserves_surface_rows(self) -> None:
        """CSV theory columns must align repeated interleaved spectra."""

        observations = pandas.DataFrame(
            {
                "ell": [30, 20, 40, 30],
                "spectrum": ["scalar_TT", "PP", "scalar_TT", "PP"],
                "Dl_obs": [10.0, 0.1, 12.0, 0.2],
            }
        )
        observations.attrs["dataset_id"] = "cmb_long_form"
        theory = {
            "scalar_TT": numpy.array([9.0, 0.0, 11.0, 0.0]),
            "PP": numpy.array([0.0, 0.08, 0.0, 0.18]),
        }
        results = {"theory_spectrum": theory}

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_cmb_results_csv(
                observations,
                results,
                results,
                csv_dir=tmpdir,
                timestamp="20260812_000000",
                comparison=build_comparison_request(
                    "Reference Model",
                    "Candidate Model",
                ),
            )

            csv_files = list(Path(tmpdir).glob("*.csv"))
            self.assertEqual(len(csv_files), 1)
            output_df = pandas.read_csv(csv_files[0])

        self.assertEqual(
            output_df["spectrum"].tolist(),
            ["scalar_TT", "PP", "scalar_TT", "PP"],
        )
        numpy.testing.assert_allclose(
            output_df["Dl_Reference_Model"],
            [9.0, 0.08, 11.0, 0.18],
        )
        numpy.testing.assert_allclose(
            output_df["residual_Reference_Model"],
            [1.0, 0.02, 1.0, 0.02],
        )


if __name__ == "__main__":
    unittest.main()
