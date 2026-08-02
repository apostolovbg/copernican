"""Focused tests for the native CMB solver orchestration module."""

import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb import (
    copernican_cmb_solver as native_cmb_solver,
)
from copernican.lib.likelihoods.cmb import (
    native_background,
    native_evolution,
    native_projection,
)


class CopernicanCmbSolverModuleTestCase(unittest.TestCase):
    """Exercise the native solver orchestration helpers directly."""

    def test_native_module_keeps_only_internal_helpers(self):
        """The native module should not expose a second public CMB facade."""

        self.assertFalse(hasattr(native_cmb_solver, "compute_cmb_spectrum"))
        self.assertFalse(
            hasattr(native_cmb_solver, "compute_cmb_spectrum_cached")
        )
        self.assertFalse(
            hasattr(native_cmb_solver, "compute_cmb_spectrum_from_contract")
        )
        self.assertFalse(
            hasattr(native_cmb_solver, "compute_cmb_spectrum_from_dict")
        )
        self.assertFalse(hasattr(native_cmb_solver, "CMBLike"))
        self.assertFalse(
            hasattr(native_cmb_solver, "_CustomCMBBackgroundData")
        )
        self.assertFalse(hasattr(native_cmb_solver, "CustomCMBSpectrumData"))
        self.assertFalse(
            hasattr(
                native_cmb_solver,
                "_compile_declared_graph_execution_plan",
            )
        )
        self.assertTrue(
            hasattr(native_background._CustomCMBBackgroundData, "sample")
        )
        self.assertTrue(hasattr(native_projection, "CustomCMBSpectrumData"))
        self.assertTrue(
            callable(native_evolution._compile_declared_graph_execution_plan)
        )
        self.assertTrue(
            callable(native_cmb_solver._compute_declared_perturbation_spectrum)
        )

    def test_lensed_assembly_uses_native_unlensed_and_pp_surfaces(self):
        """Lensed output must be assembled from declared native surfaces."""

        ell_grid = numpy.arange(16, dtype=int)
        temperature_cls = numpy.linspace(1.0, 2.0, ell_grid.size)
        electric_cls = numpy.linspace(0.5, 1.0, ell_grid.size)
        magnetic_cls = numpy.linspace(0.1, 0.2, ell_grid.size)
        temperature_electric_cls = numpy.linspace(-0.2, 0.3, ell_grid.size)
        lensing_potential_cls = numpy.linspace(0.01, 0.02, ell_grid.size)

        with mock.patch.object(
            native_cmb_solver,
            "_lensed_cls",
            return_value=numpy.zeros(
                (ell_grid.size, 4), dtype=numpy.longdouble
            ),
        ) as remapper:
            result = native_cmb_solver._assemble_exact_lensed_spectra(
                {
                    "TT": temperature_cls,
                    "EE": electric_cls,
                    "BB": magnetic_cls,
                    "TE": temperature_electric_cls,
                    "PP": lensing_potential_cls,
                },
                ell_grid,
            )

        remapper.assert_called_once()
        base_cls, clpp = remapper.call_args.args[:2]
        numpy.testing.assert_array_equal(base_cls[:, 0], temperature_cls)
        numpy.testing.assert_array_equal(base_cls[:, 1], electric_cls)
        numpy.testing.assert_array_equal(base_cls[:, 2], magnetic_cls)
        numpy.testing.assert_array_equal(
            base_cls[:, 3], temperature_electric_cls
        )
        numpy.testing.assert_array_equal(clpp[2:], lensing_potential_cls[2:])
        self.assertEqual(
            set(result),
            {"lensed_TT", "lensed_TE", "lensed_EE", "lensed_BB"},
        )

    def test_lensed_assembly_rejects_sparse_analysis_grid(self):
        """Sparse requests must be expanded before exact remapping."""

        spectra = {
            name: numpy.ones(8, dtype=float)
            for name in ("TT", "TE", "EE", "PP")
        }
        with self.assertRaisesRegex(ValueError, "contiguous ell grid"):
            native_cmb_solver._assemble_exact_lensed_spectra(
                spectra,
                numpy.asarray((0, 2, 4, 6), dtype=int),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
