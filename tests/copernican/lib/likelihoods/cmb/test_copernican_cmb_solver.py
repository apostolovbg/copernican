"""Focused tests for the native CMB solver orchestration module."""

import unittest

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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
