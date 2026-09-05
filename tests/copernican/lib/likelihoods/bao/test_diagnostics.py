"""Tests for the independent BAO certification boundary."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.bao.diagnostics import (
    assess_bao_cmb_isolation,
    assess_bao_sound_horizon_epochs,
)


class BAODiagnosticSymbolCoverageTestCase(unittest.TestCase):
    """Expose and exercise the BAO isolation helper."""

    def test_public_helper_preserves_nested_values(self) -> None:
        """Nested values and typed failures remain part of the comparison."""

        baseline = {
            "chi2": 4.0,
            "covariance": {"mode": "full", "inverse": numpy.eye(2)},
            "typed_failure": {"category": "none"},
        }
        evidence = assess_bao_cmb_isolation(baseline, dict(baseline))
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["converged"])
        self.assertTrue(evidence["covariance_preserved"])
        self.assertTrue(evidence["typed_failures_preserved"])

    def test_sound_horizon_report_separates_epochs(self) -> None:
        """BAO diagnostics retain both physical sound-horizon endpoints."""

        class Plugin:
            """Minimal generated-plugin shape for epoch evidence."""

            def get_sound_horizon_rs_rec_Mpc(self, *_params):
                return 144.0

            def get_sound_horizon_rs_drag_Mpc(self, *_params):
                return 147.0

            def get_bao_drag_redshift(self, *_params):
                return 1020.0

        report = assess_bao_sound_horizon_epochs(Plugin(), ())
        self.assertTrue(report["available"])
        self.assertTrue(report["finite"])
        self.assertTrue(report["distinct"])
        self.assertEqual(report["sound_horizon_epoch"], "drag")


if __name__ == "__main__":
    unittest.main()
