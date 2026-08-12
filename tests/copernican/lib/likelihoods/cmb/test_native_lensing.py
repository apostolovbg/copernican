"""Focused tests for the native curved-sky lensing remapper."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.cmb import native_convergence
from copernican.lib.likelihoods.cmb.native_lensing import (
    legendre_funcs,
    legendrep,
)
from copernican.lib.likelihoods.cmb.native_lensing import (
    lensed_cls as remap_lensed_cls,
)
from copernican.lib.likelihoods.cmb.native_lensing import lensed_correlations


class NativeLensingTestCase(unittest.TestCase):
    """Validate the exact native CMB lensing helper."""

    def test_legendrep_returns_expected_polynomials(self) -> None:
        """Legendre helper output should include the low-order basis."""

        polynomials, derivatives = legendrep(4, 0.25)

        self.assertEqual(len(polynomials), 5)
        self.assertEqual(len(derivatives), 5)
        self.assertAlmostEqual(float(polynomials[0]), 1.0)
        self.assertAlmostEqual(float(polynomials[1]), 0.25)
        self.assertAlmostEqual(float(derivatives[0]), 0.0)

    def test_legendre_funcs_returns_helper_arrays(self) -> None:
        """Spin-weighted Legendre helpers should expose all requested modes."""

        helper_arrays = legendre_funcs(4, 0.25, modes=(0, 1, 2))

        self.assertEqual(len(helper_arrays), 3)
        self.assertEqual(len(helper_arrays[0][0]), 5)
        self.assertEqual(len(helper_arrays[1][0]), 4)
        self.assertEqual(len(helper_arrays[2][0]), 3)
        for helper_pair in helper_arrays:
            for helper_array in helper_pair:
                self.assertTrue(numpy.all(numpy.isfinite(helper_array)))

    def test_lensed_correlations(self) -> None:
        """The direct correlation helper should return finite arrays."""

        self.assertEqual(lensed_correlations.__name__, "lensed_correlations")

        ell_count = 8
        ell_grid = numpy.arange(ell_count, dtype=float)
        unlensed_cls = numpy.zeros((ell_count, 4), dtype=float)
        unlensed_cls[:, 0] = 1.0 / (ell_grid + 1.0)
        unlensed_cls[:, 1] = 0.5 / (ell_grid + 1.0)
        unlensed_cls[:, 2] = 0.25 / (ell_grid + 1.0)
        unlensed_cls[:, 3] = 0.75 / (ell_grid + 1.0)
        lensing_potential_cls = numpy.zeros(ell_count, dtype=float)
        lensing_potential_cls[2:] = 1.0 / (ell_grid[2:] + 1.0)
        xvals = numpy.linspace(-0.5, 0.5, 3, dtype=float)
        weights = numpy.full(xvals.shape, 1.0 / xvals.size, dtype=float)

        correlations, lensed_cls = lensed_correlations(
            unlensed_cls,
            lensing_potential_cls,
            xvals,
            weights,
            lmax=ell_count - 1,
            delta=True,
        )

        self.assertEqual(correlations.shape, (xvals.size, 4))
        self.assertEqual(lensed_cls.shape, unlensed_cls.shape)
        self.assertTrue(numpy.all(numpy.isfinite(correlations)))
        self.assertTrue(numpy.all(numpy.isfinite(lensed_cls)))
        self.assertGreater(float(numpy.max(numpy.abs(correlations))), 0.0)
        self.assertGreater(float(numpy.max(numpy.abs(lensed_cls))), 0.0)

    def test_lensed_cls_remaps_finite_spectra(self) -> None:
        """Exact remapping should stay finite and change the inputs."""

        ell_count = 12
        ell_grid = numpy.arange(ell_count, dtype=float)
        unlensed_cls = numpy.zeros((ell_count, 4), dtype=float)
        unlensed_cls[:, 0] = 1.0e-6 / (ell_grid + 1.0)
        unlensed_cls[:, 1] = 2.0e-6 / (ell_grid + 1.0)
        unlensed_cls[:, 2] = 5.0e-7 / (ell_grid + 1.0)
        unlensed_cls[:, 3] = 8.0e-7 / (ell_grid + 1.0)
        lensing_potential_cls = numpy.zeros(ell_count, dtype=float)

        lensed_cls = remap_lensed_cls(
            unlensed_cls,
            lensing_potential_cls,
            lmax=ell_count - 1,
            lmax_lensed=ell_count - 1,
        )

        self.assertEqual(lensed_cls.shape, unlensed_cls.shape)
        self.assertTrue(numpy.all(numpy.isfinite(lensed_cls)))
        numpy.testing.assert_allclose(lensed_cls, unlensed_cls)

    def test_remapping_sampling_refinement_is_stable(self) -> None:
        """Increasing curved-sky quadrature remains interpolation-stable."""

        ell_count = 33
        ell_grid = numpy.arange(ell_count, dtype=float)
        unlensed_cls = numpy.zeros((ell_count, 4), dtype=numpy.longdouble)
        unlensed_cls[:, 0] = 1.0e-5 / numpy.sqrt(ell_grid + 1.0)
        unlensed_cls[:, 1] = 2.0e-6 / (ell_grid + 1.0) ** 0.3
        unlensed_cls[:, 2] = 1.0e-7 / (ell_grid + 1.0) ** 0.2
        unlensed_cls[:, 3] = 3.0e-6 / (ell_grid + 1.0) ** 0.4
        lensing_potential_cls = numpy.zeros(ell_count, dtype=numpy.longdouble)
        lensing_potential_cls[2:] = 1.0e-7 / (
            ell_grid[2:] * (ell_grid[2:] + 1.0)
        )

        coarse = remap_lensed_cls(
            unlensed_cls,
            lensing_potential_cls,
            lmax=ell_count - 1,
            lmax_lensed=ell_count - 1,
            sampling_factor=1.4,
        )
        refined = remap_lensed_cls(
            unlensed_cls,
            lensing_potential_cls,
            lmax=ell_count - 1,
            lmax_lensed=ell_count - 1,
            sampling_factor=2.2,
        )

        self.assertTrue(numpy.all(numpy.isfinite(refined)))
        numpy.testing.assert_allclose(
            coarse,
            refined,
            rtol=1.0e-7,
            atol=1.0e-14,
        )
        bb_metric = native_convergence.evaluate_control_refinement(
            coarse[:, 2],
            refined[:, 2],
            name="lensed BB quadrature",
            tolerance=(
                native_convergence.FINAL_SPECTRUM_RELATIVE_TOLERANCES[
                    "lensed_BB"
                ]
            ),
        )
        native_convergence.require_native_convergence(bb_metric)

    def test_remapping_rejects_incompatible_surfaces(self) -> None:
        """Invalid remapping surfaces fail before numerical work begins."""

        cls = numpy.zeros((8, 4), dtype=float)
        clpp = numpy.zeros(8, dtype=float)
        with self.assertRaisesRegex(ValueError, r"shape \(ell, 4\)"):
            remap_lensed_cls(
                cls[:, :3],
                clpp,
                lmax=7,
                lmax_lensed=7,
            )
        with self.assertRaisesRegex(ValueError, "cover every ell"):
            remap_lensed_cls(
                cls,
                clpp[:-1],
                lmax=7,
                lmax_lensed=7,
            )
        with self.assertRaisesRegex(ValueError, "sampling_factor"):
            remap_lensed_cls(
                cls,
                clpp,
                lmax=7,
                lmax_lensed=7,
                sampling_factor=0.5,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
