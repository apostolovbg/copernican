"""Regression tests for logging diagnostics helpers."""

import unittest

import numpy as np
import pandas as pd

from copernican_lib.diagnostics import (
    bao_residual_diagnostics,
    cmb_residual_diagnostics,
)


class DiagnosticsTestCase(unittest.TestCase):
    """Ensure residual diagnostics emit the expected log snippets."""

    def test_bao_diagnostics_groups_observables(self):
        df = pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0],
                "model_prediction": [1.1, 1.9, 2.8],
                "observable_type": [
                    "DM_over_rs",
                    "DM_over_rs",
                    "DV_over_rs",
                ],
            }
        )

        lines = bao_residual_diagnostics(df, model_name="TestModel")
        self.assertIn("TestModel BAO residual RMS", lines[0])
        self.assertTrue(any("DM_over_rs" in line for line in lines))
        self.assertTrue(any("DV_over_rs" in line for line in lines))

    def test_cmb_diagnostics_reports_components(self):
        df = pd.DataFrame(
            {
                "ell": [2, 3, 4],
                "Dl_obs": [10.0, 11.0, 12.0],
                "Dl_te_obs": [0.1, np.nan, 0.2],
                "Dl_ee_obs": [0.3, 0.4, np.nan],
            }
        )
        theory = {
            "TT": np.array([9.5, 10.5, 11.5]),
            "TE": np.array([0.05, 0.05, 0.05]),
        }

        lines = cmb_residual_diagnostics(df, theory, model_name="ModelX")
        self.assertTrue(any("ModelX CMB TT" in line for line in lines))
        self.assertTrue(any("ModelX CMB TE" in line for line in lines))
        self.assertFalse(any("mismatched" in line for line in lines))


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    unittest.main()
