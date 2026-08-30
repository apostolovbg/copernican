"""Tests for the independent BAO certification boundary."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.bao.diagnostics import assess_bao_cmb_isolation


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


if __name__ == "__main__":
    unittest.main()
