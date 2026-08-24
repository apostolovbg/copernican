"""Regression tests for logging diagnostics helpers."""

import unittest

import numpy
import pandas

from copernican.lib import diagnostics as module
from copernican.lib.diagnostics import (
    bao_residual_diagnostics,
    cmb_residual_diagnostics,
)
from copernican.lib.likelihoods import cmb as cmb_package
from copernican.lib.likelihoods.cmb import diagnostics as cmb_module
from copernican.lib.likelihoods.cmb.contracts_audit import (
    CMBContractAudit,
    audit_bundled_cmb_contracts,
)
from copernican.lib.likelihoods.cmb.diagnostics import (
    CMBModelDiagnostic,
    run_bundled_cmb_diagnostics,
    run_cmb_model_diagnostic,
)


class DiagnosticsTestCase(unittest.TestCase):
    """Ensure residual diagnostics emit the expected log snippets."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(hasattr(module, "bao_residual_diagnostics"))
        self.assertTrue(hasattr(module, "cmb_residual_diagnostics"))

    def test_ccmbs_diagnostic_symbols_are_exposed(self) -> None:
        """The fixed-point CCMBS diagnostics surface remains importable."""

        self.assertIs(cmb_module.CMBModelDiagnostic, CMBModelDiagnostic)
        self.assertIs(
            cmb_module.run_bundled_cmb_diagnostics,
            run_bundled_cmb_diagnostics,
        )
        self.assertIs(
            cmb_module.run_cmb_model_diagnostic,
            run_cmb_model_diagnostic,
        )
        self.assertIs(cmb_package.CMBContractAudit, CMBContractAudit)
        self.assertIs(
            cmb_package.audit_bundled_cmb_contracts,
            audit_bundled_cmb_contracts,
        )

    def test_bao_diagnostics_groups_observables(self):
        data_frame = pandas.DataFrame(
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

        lines = bao_residual_diagnostics(data_frame, model_name="TestModel")
        self.assertIn("TestModel BAO residual RMS", lines[0])
        self.assertTrue(any("DM_over_rs" in line for line in lines))
        self.assertTrue(any("DV_over_rs" in line for line in lines))

    def test_cmb_diagnostics_reports_components(self):
        data_frame = pandas.DataFrame(
            {
                "ell": [2, 3, 4],
                "Dl_obs": [10.0, 11.0, 12.0],
                "Dl_te_obs": [0.1, numpy.nan, 0.2],
                "Dl_ee_obs": [0.3, 0.4, numpy.nan],
            }
        )
        theory = {
            "TT": numpy.array([9.5, 10.5, 11.5]),
            "TE": numpy.array([0.05, 0.05, 0.05]),
        }

        lines = cmb_residual_diagnostics(
            data_frame, theory, model_name="ModelX"
        )
        self.assertTrue(any("ModelX CMB TT" in line for line in lines))
        self.assertTrue(any("ModelX CMB TE" in line for line in lines))
        self.assertFalse(any("mismatched" in line for line in lines))

    def test_cmb_diagnostics_preserves_long_form_surface_names(self):
        """Long-form diagnostics must not merge physical output surfaces."""

        data_frame = pandas.DataFrame(
            {
                "ell": [30, 20, 40, 30],
                "spectrum": ["scalar_TT", "PP", "scalar_TT", "PP"],
                "Dl_obs": [10.0, 0.1, 12.0, 0.2],
            }
        )
        theory = {
            "scalar_TT": numpy.array([9.0, 0.0, 11.0, 0.0]),
            "PP": numpy.array([0.0, 0.08, 0.0, 0.18]),
        }

        lines = cmb_residual_diagnostics(
            data_frame,
            theory,
            model_name="ModelX",
        )

        self.assertTrue(any("CMB scalar_TT" in line for line in lines))
        self.assertTrue(any("CMB PP" in line for line in lines))


if __name__ == "__main__":  # pragma: no cover - convenience for local runs
    unittest.main()
