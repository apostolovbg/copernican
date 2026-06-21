"""Focused tests for the native CMB solver module."""

import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import copcmb_solver


class CopcmbSolverModuleTestCase(unittest.TestCase):
    """Exercise the native solver helpers directly."""

    def test_public_solver_symbols_are_exposed(self):
        """The native module should keep its public surface importable."""

        self.assertTrue(callable(copcmb_solver.compute_cmb_spectrum_from_dict))
        self.assertTrue(callable(copcmb_solver.compute_cmb_spectrum_cached))
        self.assertTrue(callable(copcmb_solver.compute_cmb_spectrum))
        self.assertTrue(
            callable(copcmb_solver.compute_camb_background_observables)
        )
        self.assertTrue(callable(copcmb_solver.describe_camb_configuration))
        self.assertTrue(
            callable(
                copcmb_solver.compute_cmb_spectrum_from_legacy_params_for_tests
            )
        )
        self.assertTrue(hasattr(copcmb_solver, "CMBLike"))
        self.assertTrue(hasattr(copcmb_solver.CMBLike, "loglike"))
        self.assertTrue(hasattr(copcmb_solver.CMBLike, "state"))
        self.assertTrue(
            hasattr(copcmb_solver._CustomCMBBackgroundData, "sample")
        )
        self.assertTrue(hasattr(copcmb_solver, "CustomCMBSpectrumData"))

    def test_precompiled_perturbation_payload_is_reused(self):
        """Existing compiled perturbation data should bypass recompilation."""

        payload = object()
        contract = {"perturbation_data": payload}

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            side_effect=AssertionError("recompilation should not run"),
        ):
            compiled = copcmb_solver._compile_declared_perturbation_contract(
                contract
            )

        self.assertIs(compiled, payload)

    def test_custom_spectrum_data_accessors_return_named_payloads(self):
        """Transfer and spectrum accessors should expose stable arrays."""

        spectrum_data = copcmb_solver.CustomCMBSpectrumData(
            ell_grid=numpy.array([20.0, 30.0]),
            k_grid=numpy.array([0.1, 0.2]),
            transfer_components={
                "temperature": numpy.array([1.0, 2.0]),
                "polarization_e": numpy.array([3.0, 4.0]),
            },
            spectra={
                "TT": numpy.array([5.0, 6.0]),
                "TE": numpy.array([7.0, 8.0]),
                "EE": numpy.array([9.0, 10.0]),
            },
        )

        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_T, numpy.array([1.0, 2.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.Delta_l_E, numpy.array([3.0, 4.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TT, numpy.array([5.0, 6.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_TE, numpy.array([7.0, 8.0]))
        )
        self.assertTrue(
            numpy.array_equal(spectrum_data.C_l_EE, numpy.array([9.0, 10.0]))
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
