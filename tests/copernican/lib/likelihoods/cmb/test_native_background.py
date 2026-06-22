"""Focused tests for the native CMB background module."""

import unittest
from pathlib import Path

import numpy

from copernican.lib.likelihoods.cmb import native_background


class NativeBackgroundModuleTestCase(unittest.TestCase):
    """Exercise native background helpers directly."""

    def test_manifest_summary_tracks_declared_background_metadata(self):
        """Manifest summaries should report declared background details."""

        contract = {
            "background": {
                "derived": {
                    "H": "1.0",
                    "rho_tot": "2.0",
                    "p_tot": "0.5",
                    "w_tot": "0.25",
                }
            },
            "param_map": {},
        }

        summary = (
            native_background._summarize_declared_background_manifest_summary(
                contract
            )
        )

        self.assertIn("H", summary["background_derived_names"])
        self.assertEqual(
            summary["recombination_runtime"]["hydrogen_model"],
            "peebles_case_b_ode",
        )
        self.assertTrue(
            hasattr(native_background._CustomCMBBackgroundData, "sample")
        )

    def test_custom_spectrum_accessors_return_named_payloads(self):
        """Spectrum payload accessors should expose stable arrays."""

        spectrum_data = native_background.CustomCMBSpectrumData(
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

        self.assertIs(
            type(spectrum_data),
            native_background.CustomCMBSpectrumData,
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

    def test_native_background_source_does_not_import_camb(self):
        """The native background module should remain CAMB-free."""

        source_text = Path(native_background.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import camb", source_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
