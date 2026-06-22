"""Focused tests for the native CMB projection module."""

import unittest
from pathlib import Path

import numpy

from copernican.lib.likelihoods.cmb import native_projection


class NativeProjectionModuleTestCase(unittest.TestCase):
    """Exercise native projection helpers directly."""

    def test_custom_spectrum_data_accessors_return_named_payloads(self):
        """Transfer and spectrum accessors should expose stable arrays."""

        spectrum_data = native_projection.CustomCMBSpectrumData(
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

    def test_native_projection_source_does_not_import_camb(self):
        """The native projection module should remain CAMB-free."""

        source_text = Path(native_projection.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import camb", source_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
