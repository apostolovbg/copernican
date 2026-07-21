"""Focused tests for native CMB adaptive refinement controls."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.cmb.native_adaptive import (
    NativeAdaptiveControls,
    NativeConvergenceEstimate,
    estimate_convergence,
    phase_aware_eta_grid,
    phase_aware_k_grid,
    require_convergence,
    resolve_native_adaptive_controls,
)


class NativeAdaptiveControlsTestCase(unittest.TestCase):
    """Validate physical grid refinement and convergence failure behavior."""

    def test_public_symbols_are_exposed(self) -> None:
        """The adaptive module keeps its control and diagnostic API stable."""

        self.assertTrue(callable(estimate_convergence))
        self.assertTrue(callable(phase_aware_eta_grid))
        self.assertTrue(callable(phase_aware_k_grid))
        self.assertTrue(callable(require_convergence))
        self.assertTrue(callable(resolve_native_adaptive_controls))
        self.assertEqual(
            NativeAdaptiveControls.__name__,
            "NativeAdaptiveControls",
        )
        self.assertEqual(
            NativeConvergenceEstimate.__name__,
            "NativeConvergenceEstimate",
        )

    def test_controls_resolve_the_three_refinement_surfaces(self) -> None:
        """Transfer, source, and projection sections resolve independently."""

        controls = resolve_native_adaptive_controls(
            {
                "adaptive_transfer": {
                    "minimum_nodes": 8,
                    "maximum_nodes": 24,
                    "relative_tolerance": 0.1,
                },
                "adaptive_source": {
                    "minimum_nodes": 12,
                    "maximum_nodes": 48,
                },
                "adaptive_projection": {
                    "relative_tolerance": 0.2,
                },
                "phase_points_per_cycle": 6,
            },
            base_k_nodes=8,
            base_eta_nodes=12,
        )

        self.assertTrue(controls.transfer_enabled)
        self.assertTrue(controls.source_enabled)
        self.assertTrue(controls.projection_enabled)
        self.assertEqual(controls.transfer_maximum_nodes, 24)
        self.assertEqual(controls.source_maximum_nodes, 48)
        self.assertEqual(controls.phase_points_per_cycle, 6.0)

    def test_phase_aware_k_grid_tracks_acoustic_and_radial_phase(self) -> None:
        """The transfer grid adds physical phase nodes within its bounds."""

        grid = phase_aware_k_grid(
            0.01,
            0.25,
            minimum_nodes=8,
            maximum_nodes=40,
            phase_points_per_cycle=8.0,
            eta_distance=14000.0,
            sound_horizon=140.0,
            anchors=(0.05, 0.1),
        )

        self.assertGreaterEqual(grid.size, 8)
        self.assertLessEqual(grid.size, 40)
        self.assertTrue(numpy.all(numpy.diff(grid) > 0.0))
        self.assertAlmostEqual(float(grid[0]), 0.01)
        self.assertAlmostEqual(float(grid[-1]), 0.25)
        self.assertTrue(numpy.any(numpy.isclose(grid, 0.05)))
        self.assertTrue(numpy.any(numpy.isclose(grid, 0.1)))

    def test_phase_aware_eta_grid_refines_visibility_and_oscillations(
        self,
    ) -> None:
        """Visibility peaks and rapid Fourier phase receive extra nodes."""

        eta = numpy.linspace(0.0, 10.0, 9)
        visibility = numpy.exp(-0.5 * numpy.square((eta - 5.0) / 0.5))
        refined = phase_aware_eta_grid(
            eta,
            visibility=visibility,
            k_max=3.0,
            minimum_nodes=9,
            maximum_nodes=48,
            phase_points_per_cycle=8.0,
        )

        self.assertGreater(refined.size, eta.size)
        self.assertLessEqual(refined.size, 48)
        self.assertTrue(numpy.all(numpy.diff(refined) > 0.0))
        self.assertTrue(numpy.any(numpy.isclose(refined, 5.0)))
        self.assertLess(
            float(numpy.max(numpy.diff(refined))),
            float(numpy.max(numpy.diff(eta))),
        )

    def test_convergence_estimate_rejects_underresolved_result(self) -> None:
        """A strict tolerance raises a named under-resolution error."""

        estimate = estimate_convergence(
            numpy.asarray((1.0, 2.0)),
            numpy.asarray((1.0, 2.5)),
            relative_tolerance=0.01,
            absolute_tolerance=1.0e-12,
        )
        self.assertFalse(estimate.converged)
        self.assertGreater(estimate.relative_error, 0.01)
        with self.assertRaisesRegex(ValueError, "transfer refinement"):
            require_convergence(
                estimate,
                label="transfer",
                fail_on_nonconvergence=True,
            )

    def test_convergence_estimate_accepts_absolute_floor(self) -> None:
        """Tiny physical signals may pass through the absolute tolerance."""

        estimate = estimate_convergence(
            numpy.asarray((0.0, 1.0e-14)),
            numpy.asarray((0.0, 2.0e-14)),
            relative_tolerance=1.0e-6,
            absolute_tolerance=1.0e-12,
        )
        self.assertTrue(estimate.converged)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
