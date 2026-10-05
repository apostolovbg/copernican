"""Focused tests for declared CMB adaptive refinement controls."""

from __future__ import annotations

import unittest

import numpy

from copernican.lib.likelihoods.cmb.errors import classify_exception
from copernican.lib.likelihoods.cmb.runtime.adaptive import (
    AdaptiveControls,
    AdaptiveNonConvergenceError,
    ConvergenceEstimate,
    HistoryConvergence,
    LOSQuadratureControls,
    estimate_convergence,
    estimate_history_convergence,
    informative_k_indices,
    nested_coarse_indices,
    nested_phase_aware_k_grid,
    phase_aware_eta_grid,
    phase_aware_k_grid,
    phase_aware_k_grid_requirements,
    phase_aware_k_grid_status,
    physical_history_anchors,
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
        self.assertTrue(callable(nested_phase_aware_k_grid))
        self.assertTrue(callable(informative_k_indices))
        self.assertTrue(callable(physical_history_anchors))
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
        self.assertEqual(controls.evolution_validation_mode_count, 3)
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

    def test_los_phase_cap_preserves_configured_history_bound(self) -> None:
        """A configured phase cap remains authoritative for dense histories."""

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
        self.assertEqual(controls.maximum_nodes, 2048)

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

    def test_phase_aware_k_grid_fills_duplicate_optional_nodes(self) -> None:
        """Duplicate optional nodes cannot violate the minimum budget."""

        grid = phase_aware_k_grid(
            1.0e-5,
            0.30,
            minimum_nodes=1024,
            maximum_nodes=1024,
            phase_points_per_cycle=8.0,
            eta_distance=14000.0,
            sound_horizon=140.0,
        )

        self.assertEqual(grid.size, 1024)
        self.assertTrue(numpy.all(numpy.isfinite(grid)))
        self.assertTrue(numpy.all(numpy.diff(grid) > 0.0))
        self.assertAlmostEqual(float(grid[0]), 1.0e-5)
        self.assertAlmostEqual(float(grid[-1]), 0.30)

    def test_nested_phase_grid_preserves_base_nodes(self) -> None:
        """Projection refinement must reuse every evolved base mode."""

        base = numpy.asarray((0.01, 0.05, 0.10, 0.25), dtype=float)
        refined = nested_phase_aware_k_grid(
            base,
            maximum_nodes=12,
            phase_points_per_cycle=8.0,
            eta_distance=5.0,
            sound_horizon=2.5,
            require_phase_resolution=True,
        )

        self.assertGreater(refined.size, base.size)
        self.assertLessEqual(refined.size, 12)
        self.assertTrue(numpy.all(numpy.diff(refined) > 0.0))
        for value in base:
            self.assertTrue(numpy.any(numpy.isclose(refined, value)))

    def test_nested_phase_grid_refines_measured_spacing_after_count_floor(
        self,
    ) -> None:
        """A clustered base ladder must refine beyond its count estimate."""

        base = numpy.concatenate(
            (numpy.linspace(0.1, 0.2, 255), numpy.asarray((1.0,)))
        )
        refined = nested_phase_aware_k_grid(
            base,
            maximum_nodes=1024,
            phase_points_per_cycle=8.0,
            eta_distance=100.0,
            sound_horizon=1.0,
            require_phase_resolution=True,
        )

        self.assertGreater(refined.size, base.size)
        self.assertLessEqual(refined.size, 1024)
        self.assertTrue(numpy.all(numpy.isin(base, refined)))
        status = phase_aware_k_grid_status(
            refined,
            phase_points_per_cycle=8.0,
            eta_distance=100.0,
            sound_horizon=1.0,
        )
        self.assertTrue(status["spacing_resolved"])
        self.assertTrue(status["resolved"])

    def test_nested_phase_grid_rejects_an_impossible_cap(self) -> None:
        """A physical phase requirement must not be silently truncated."""

        base = numpy.asarray((0.01, 0.10, 0.25), dtype=float)
        with self.assertRaisesRegex(ValueError, "node cap"):
            nested_phase_aware_k_grid(
                base,
                maximum_nodes=5,
                phase_points_per_cycle=8.0,
                eta_distance=10.0,
                sound_horizon=5.0,
                require_phase_resolution=True,
            )

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
        self.assertFalse(bool(status["spacing_resolved"]))
        self.assertGreater(float(status["maximum_radial_phase_step"]), 0.0)
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

    def test_dense_background_leaves_budget_for_phase_refinement(self) -> None:
        """A dense background cannot consume the LOS refinement budget."""

        eta = numpy.linspace(0.0, 20.0, 1001)
        visibility = numpy.exp(-0.5 * numpy.square((eta - 7.0) / 0.05))
        refined = phase_aware_eta_grid(
            eta,
            visibility=visibility,
            k_max=2.0,
            minimum_nodes=32,
            maximum_nodes=128,
            phase_points_per_cycle=8.0,
        )

        self.assertEqual(refined.size, 128)
        self.assertEqual(float(refined[0]), float(eta[0]))
        self.assertEqual(float(refined[-1]), float(eta[-1]))
        self.assertTrue(numpy.any(numpy.isclose(refined, 7.0)))
        visibility_region = refined[(refined >= 6.8) & (refined <= 7.2)]
        self.assertGreater(visibility_region.size, 3)
        self.assertLess(
            float(numpy.max(numpy.diff(visibility_region))),
            20.0 / 127.0,
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
        self.assertGreater(estimate.sample_count, fine_eta.size)
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

    def test_history_convergence_rejects_spikes_between_named_anchors(
        self,
    ) -> None:
        """A localized fine-grid source spike cannot hide between anchors."""

        coarse_eta = numpy.asarray((0.0, 0.5, 1.0))
        fine_eta = numpy.linspace(0.0, 1.0, 9)
        coarse = {"source": numpy.zeros(coarse_eta.size)}
        fine_values = numpy.zeros(fine_eta.size)
        fine_values[2] = 1.0
        fine = {"source": fine_values}

        estimate = estimate_history_convergence(
            coarse_eta,
            coarse,
            fine_eta,
            fine,
            relative_tolerance=1.0e-2,
            absolute_tolerance=1.0e-12,
            feature_eta={"start": 0.0, "peak": 0.5, "end": 1.0},
        )

        self.assertFalse(estimate.converged)
        self.assertGreater(estimate.relative_error, 0.9)

    def test_history_convergence_rejects_resolved_sign_reversal(self) -> None:
        """A physical sign reversal near a zero receives a finite error."""

        eta = numpy.linspace(0.0, 1.0, 17)
        coarse = {"source": eta - 0.5}
        fine = {"source": 0.5 - eta}

        estimate = estimate_history_convergence(
            eta,
            coarse,
            eta,
            fine,
            relative_tolerance=1.0e-2,
            absolute_tolerance=1.0e-12,
        )

        self.assertFalse(estimate.converged)
        self.assertTrue(numpy.isfinite(estimate.relative_error))

    def test_history_convergence_accepts_a_resolved_dense_control(
        self,
    ) -> None:
        """Independent dense histories pass when their full surfaces agree."""

        coarse_eta = numpy.linspace(0.0, 1.0, 33)
        fine_eta = numpy.linspace(0.0, 1.0, 65)
        coarse = {"source": numpy.sin(2.0 * numpy.pi * coarse_eta)}
        fine = {"source": numpy.sin(2.0 * numpy.pi * fine_eta)}

        estimate = estimate_history_convergence(
            coarse_eta,
            coarse,
            fine_eta,
            fine,
            relative_tolerance=1.0e-2,
            absolute_tolerance=1.0e-12,
        )

        self.assertTrue(estimate.converged)
        self.assertLess(estimate.relative_error, 1.0e-2)

    def test_physical_features_and_nested_coarse_grid_are_distinct(
        self,
    ) -> None:
        """Feature sampling preserves endpoints without an identical grid."""

        eta = numpy.linspace(0.0, 10.0, 33)
        visibility = numpy.exp(-numpy.square((eta - 4.0) / 0.5))
        interaction = numpy.exp(8.0 - eta)
        anchors = physical_history_anchors(
            eta,
            visibility=visibility,
            interaction_rate=interaction,
        )
        indices = nested_coarse_indices(eta, feature_eta=anchors)

        self.assertEqual(anchors["integration_start"], 0.0)
        self.assertEqual(anchors["integration_end"], 10.0)
        self.assertLess(indices.size, eta.size)
        self.assertEqual(int(indices[0]), 0)
        self.assertEqual(int(indices[-1]), eta.size - 1)

    def test_informative_k_modes_cover_endpoints_and_features(self) -> None:
        """Validation modes include the physical k surface, not fractions."""

        k_values = numpy.geomspace(1.0e-4, 0.4, 100)
        indices = informative_k_indices(
            k_values,
            count=7,
            feature_k=(0.01, 0.1),
        )

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 99)
        self.assertEqual(len(indices), 7)

    def test_nonconvergence_error_retains_failed_products(self) -> None:
        """Adaptive failures expose typed machine-readable product context."""

        estimate = ConvergenceEstimate(1.0, 1.0, False)
        with self.assertRaises(AdaptiveNonConvergenceError) as raised:
            require_convergence(
                estimate,
                label="source-history",
                fail_on_nonconvergence=True,
            )

        self.assertEqual(raised.exception.label, "source-history")
        self.assertEqual(
            raised.exception.failed_products,
            ("source-history",),
        )
        diagnostic = classify_exception(raised.exception).diagnostic()
        self.assertEqual(diagnostic["category"], "convergence_failure")
        self.assertEqual(
            diagnostic["context"]["failed_products"],
            ("source-history",),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
