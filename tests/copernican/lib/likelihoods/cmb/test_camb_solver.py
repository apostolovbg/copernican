"""Focused tests for the standard CMB solver module."""

import unittest

from copernican.lib.likelihoods.cmb import camb_solver


class CambSolverModuleTestCase(unittest.TestCase):
    """Exercise the public helper surface exposed by camb_solver."""

    def test_describe_camb_configuration_reports_expected_defaults(self):
        """The standard helper should report the default CAMB settings."""

        configuration = camb_solver.describe_camb_configuration()

        self.assertEqual(
            configuration["reionization_model"], "optical_depth_tau"
        )
        self.assertIn("lmax_padding", configuration)
        self.assertIn("lens_potential_accuracy", configuration)
        self.assertIn("accuracy", configuration)

    def test_public_solver_symbols_are_exposed(self):
        """The module should keep standard solver entrypoints importable."""

        self.assertIn(
            "compute_cmb_spectrum_from_camb_contract", camb_solver.__all__
        )
        self.assertIn(
            "compute_camb_background_observables", camb_solver.__all__
        )
        self.assertIn(
            "compute_cmb_spectrum_from_legacy_params_for_tests",
            camb_solver.__all__,
        )
        self.assertIn("describe_camb_configuration", camb_solver.__all__)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
