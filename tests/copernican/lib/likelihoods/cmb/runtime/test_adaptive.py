"""Focused tests for declared CMB adaptive refinement controls."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.cmb.runtime.adaptive import (
    AdaptiveControls,
    ConvergenceEstimate,
    HistoryConvergence,
    LOSQuadratureControls,
    estimate_convergence,
    estimate_history_convergence,
    phase_aware_eta_grid,
    phase_aware_k_grid,
    phase_aware_k_grid_requirements,
    phase_aware_k_grid_status,
    require_convergence,
    resolve_adaptive_controls,
    resolve_los_quadrature_controls,
)


class AdaptiveControlsTestCase(unittest.TestCase):
    """Validate physical grid refinement and convergence failure behavior."""

    def test_public_symbols_are_exposed(self) -> None:
        """The adaptive module keeps its control and diagnostic API stable."""

        self.assertTrue(callable(estimate_convergence))
        self.assertTrue(callable(phase_aware_eta_grid))
        self.assertTrue(callable(phase_aware_k_grid))
        self.assertTrue(callable(phase_aware_k_grid_requirements))
        self.assertTrue(callable(phase_aware_k_grid_status))
        self.assertTrue(callable(require_convergence))
        self.assertTrue(callable(resolve_adaptive_controls))
        self.assertEqual(
            AdaptiveControls.__name__,
            "AdaptiveControls",
        )
        self.assertEqual(
            ConvergenceEstimate.__name__,
            "ConvergenceEstimate",
        )
        self.assertEqual(
            HistoryConvergence.__name__,
            "HistoryConvergence",
        )
        self.assertTrue(callable(estimate_history_convergence))

    def test_controls_resolve_the_three_refinement_surfaces(self) -> None:
        """Transfer, source, and projection sections resolve independently."""

        controls = resolve_adaptive_controls(
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

    def test_controls_resolve_scalar_evolution_bounds(self) -> None:
        """Scalar evolution refinement keeps explicit node bounds."""

        controls = resolve_adaptive_controls(
            {
                "adaptive_evolution": {
                    "minimum_nodes": 64,
                    "maximum_nodes": 256,
                    "relative_tolerance": 1.0e-2,
                }
            },
            base_k_nodes=8,
            base_eta_nodes=128,
            base_evolution_nodes=128,
        )
        self.assertTrue(controls.evolution_enabled)
        self.assertEqual(controls.evolution_minimum_nodes, 64)
        self.assertEqual(controls.evolution_maximum_nodes, 256)
        self.assertAlmostEqual(controls.evolution_relative_tolerance, 1.0e-2)

    def test_los_phase_controls_resolve_explicit_bounded_grid(self) -> None:
        """LOS phase controls preserve explicit minimum and maximum nodes."""

        controls = resolve_los_quadrature_controls(
            {
                "los_phase_quadrature": {
                    "minimum_nodes": 512,
                    "maximum_nodes": 2048,
                    "phase_points_per_cycle": 4,
                }
            },
            base_eta_nodes=192,
        )

        self.assertIsInstance(controls, LOSQuadratureControls)
        self.assertTrue(controls.enabled)
        self.assertEqual(controls.minimum_nodes, 512)
        self.assertEqual(controls.maximum_nodes, 2048)
        self.assertEqual(controls.phase_points_per_cycle, 4.0)

    def test_los_phase_controls_are_disabled_without_declaration(self) -> None:
        """Low-resolution contracts do not inherit a hidden LOS multiplier."""

        controls = resolve_los_quadrature_controls({}, base_eta_nodes=192)

        self.assertFalse(controls.enabled)
        self.assertEqual(controls.minimum_nodes, 0)
        self.assertEqual(controls.maximum_nodes, 0)

    def test_los_phase_cap_promotes_existing_history_length(self) -> None:
        """A phase cap cannot discard an already sampled history grid."""

        controls = resolve_los_quadrature_controls(
            {
                "los_phase_quadrature": {
                    "minimum_nodes": 512,
                    "maximum_nodes": 2048,
                    "phase_points_per_cycle": 4,
                }
            },
            base_eta_nodes=3000,
        )

        self.assertEqual(controls.configured_maximum_nodes, 2048)
        self.assertEqual(controls.maximum_nodes, 3000)

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

    def test_phase_requirements_report_uncapped_physical_resolution(
        self,
    ) -> None:
        """Runtime evidence exposes the phase ladder's physical node need."""

        requirements = phase_aware_k_grid_requirements(
            0.01,
            0.25,
            phase_points_per_cycle=8.0,
            eta_distance=14000.0,
            sound_horizon=140.0,
        )

        self.assertGreater(
            requirements["radial_required_nodes"],
            requirements["acoustic_required_nodes"],
        )
        self.assertEqual(
            requirements["required_nodes"],
            requirements["radial_required_nodes"],
        )
        self.assertGreater(requirements["phase_step"], 0.0)

    def test_phase_status_exposes_capped_grid_as_under_resolved(
        self,
    ) -> None:
        """A bounded ladder reports its physical phase-resolution status."""

        status = phase_aware_k_grid_status(
            numpy.geomspace(0.01, 0.25, 8),
            phase_points_per_cycle=8.0,
            eta_distance=14000.0,
            sound_horizon=140.0,
        )

        self.assertFalse(bool(status["resolved"]))
        self.assertGreater(
            int(status["required_nodes"]),
            int(status["actual_nodes"]),
        )

    def test_phase_grid_can_reject_an_under_resolved_budget(self) -> None:
        """Production callers may reject a capped phase ladder explicitly."""

        with self.assertRaisesRegex(ValueError, "under-resolved"):
            phase_aware_k_grid(
                0.01,
                0.25,
                minimum_nodes=8,
                maximum_nodes=16,
                phase_points_per_cycle=8.0,
                eta_distance=14000.0,
                sound_horizon=140.0,
                require_phase_resolution=True,
            )

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

    def test_history_convergence_checks_physical_anchor_regions(self) -> None:
        """State histories compare independently at all declared anchors."""

        coarse_eta = numpy.asarray((0.0, 0.5, 1.0))
        fine_eta = numpy.linspace(0.0, 1.0, 9)
        coarse = {"theta": numpy.square(coarse_eta)}
        fine = {"theta": numpy.square(fine_eta) + 2.0e-2}
        estimate = estimate_history_convergence(
            coarse_eta,
            coarse,
            fine_eta,
            fine,
            relative_tolerance=1.0e-2,
            absolute_tolerance=1.0e-12,
        )
        self.assertEqual(
            set(estimate.anchor_relative_errors),
            {"early", "recombination", "late"},
        )
        self.assertFalse(estimate.converged)
        with self.assertRaisesRegex(ValueError, "history refinement"):
            require_convergence(
                ConvergenceEstimate(
                    absolute_error=estimate.absolute_error,
                    relative_error=estimate.relative_error,
                    converged=estimate.converged,
                ),
                label="history refinement",
                fail_on_nonconvergence=True,
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
