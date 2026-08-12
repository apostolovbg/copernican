"""Focused tests for native CMB cross-sector numerical convergence."""

from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace

import numpy

from copernican.lib.likelihoods.cmb.native_convergence import (
    FINAL_HIERARCHY_RELATIVE_TOLERANCE,
    FINAL_Q_GRID_RELATIVE_TOLERANCE,
    FINAL_SPECTRUM_RELATIVE_TOLERANCES,
    NativeConvergenceReport,
    NativeNumericalEnvelope,
    NativeRefinementMetric,
    evaluate_control_refinement,
    evaluate_spectrum_refinement,
    require_native_convergence,
    resolve_native_numerical_envelope,
)


def _final_tier_contract() -> dict[str, object]:
    """Return a compact all-sector contract at the final numerical floor."""

    numerics = {
        "ell_min": 2,
        "ell_max": 2000,
        "k_min": 1.0e-4,
        "k_max": 0.3,
        "k_sample_count": 18,
        "eta_sample_count": 192,
        "evolution_eta_sample_count": 128,
        "evolution_phase_step": 2.0,
        "ode_rtol": 1.0e-5,
        "ode_atol": 1.0e-8,
        "tight_coupling_ratio": 1600.0,
        "tight_coupling_exit_ratio": 0.1,
        "a_min": 1.0e-6,
        "source_grid_multiplier": 2,
        "initial_redshift": 2.0e4,
        "lensing_sampling_factor": 1.4,
        "photon_hierarchy_l_max": 12,
        "photon_polarization_hierarchy_l_max": 12,
        "neutrino_hierarchy_l_max": 9,
        "massive_neutrino_hierarchy_l_max": 7,
        "momentum_grids": {
            "massive_neutrino_default": {
                "count": 16,
                "q_min": 0.05,
                "q_max": 15.0,
                "quadrature_order": 2,
            }
        },
    }
    hierarchy_families = {
        "photon_temperature": SimpleNamespace(
            species=("photon",), default_l_max=8
        ),
        "photon_polarization_e": SimpleNamespace(
            species=("photon",), default_l_max=8
        ),
        "massless_neutrino": SimpleNamespace(
            species=("massless_neutrino",), default_l_max=5
        ),
        "massive_neutrino": SimpleNamespace(
            species=("massive_neutrino",), default_l_max=5
        ),
    }
    perturbation_data = SimpleNamespace(
        accuracy_controls={
            "accuracy_tier": "final",
            "runtime_envelope": "bounded",
        },
        hierarchy_families=hierarchy_families,
        numerics=numerics,
        sectors={"scalar": {}, "tensor": {}, "vector": {}},
    )
    return {
        "numerical": copy.deepcopy(numerics),
        "perturbation_data": perturbation_data,
    }


class NativeConvergenceTestCase(unittest.TestCase):
    """Validate final accuracy tiers and physical refinement reports."""

    def test_public_symbols_expose_final_thresholds(self) -> None:
        """The convergence module exposes the roadmap acceptance surface."""

        self.assertEqual(FINAL_SPECTRUM_RELATIVE_TOLERANCES["TT"], 0.01)
        self.assertEqual(FINAL_SPECTRUM_RELATIVE_TOLERANCES["EE"], 0.01)
        self.assertEqual(FINAL_SPECTRUM_RELATIVE_TOLERANCES["TE"], 0.02)
        self.assertEqual(FINAL_SPECTRUM_RELATIVE_TOLERANCES["PP"], 0.03)
        self.assertEqual(FINAL_SPECTRUM_RELATIVE_TOLERANCES["lensed_BB"], 0.05)
        self.assertEqual(FINAL_Q_GRID_RELATIVE_TOLERANCE, 0.02)
        self.assertEqual(FINAL_HIERARCHY_RELATIVE_TOLERANCE, 0.01)
        self.assertEqual(
            NativeNumericalEnvelope.__name__, "NativeNumericalEnvelope"
        )
        self.assertEqual(
            NativeConvergenceReport.__name__, "NativeConvergenceReport"
        )
        self.assertEqual(
            NativeRefinementMetric.__name__, "NativeRefinementMetric"
        )
        self.assertEqual(
            evaluate_control_refinement.__name__,
            "evaluate_control_refinement",
        )

    def test_final_tier_records_every_physical_control_family(self) -> None:
        """The final envelope records grids, sectors, hierarchies, and q."""

        envelope = resolve_native_numerical_envelope(_final_tier_contract())
        payload = envelope.to_dict()

        self.assertEqual(payload["accuracy_tier"], "final")
        self.assertTrue(payload["bounded"])
        self.assertEqual(payload["sectors"], ["scalar", "tensor", "vector"])
        self.assertEqual(
            set(payload["hierarchy_controls"]),
            {
                "massive_neutrino",
                "massless_neutrino",
                "photon_polarization",
                "photon_temperature",
            },
        )
        self.assertEqual(
            payload["momentum_grid_controls"]["massive_neutrino_default"][
                "count"
            ],
            16,
        )
        self.assertEqual(
            payload["numerical_controls"]["lensing_sampling_factor"],
            1.4,
        )
        self.assertEqual(len(payload["runtime_limits"]), 4)

    def test_envelope_infers_sector_from_compiled_graph_metadata(self) -> None:
        """Explicit graphs without sector blocks retain their graph sector."""

        contract = {
            "perturbation_data": SimpleNamespace(
                accuracy_controls={},
                hierarchy_families={},
                numerics={},
                sectors={},
                variables={
                    "signal": SimpleNamespace(
                        sector=None,
                        tensor_character="scalar_like",
                    )
                },
                observables={
                    "TT": SimpleNamespace(
                        sector="scalar",
                        tensor_character="scalar_like",
                    )
                },
            )
        }

        envelope = resolve_native_numerical_envelope(contract)

        self.assertEqual(envelope.sectors, ("scalar",))

    def test_final_tier_rejects_each_underresolved_control_family(
        self,
    ) -> None:
        """Final requests fail before work when any control family is short."""

        cases = {
            "background": ("eta_sample_count", 191),
            "evolution": ("evolution_eta_sample_count", 127),
            "transfer": ("k_sample_count", 17),
            "source": ("source_grid_multiplier", 1),
            "lensing": ("lensing_sampling_factor", 1.3),
            "photon hierarchy": ("photon_hierarchy_l_max", 11),
            "neutrino hierarchy": ("neutrino_hierarchy_l_max", 8),
        }
        for label, (control_name, value) in cases.items():
            with self.subTest(label=label):
                contract = _final_tier_contract()
                contract["numerical"][control_name] = value
                contract["perturbation_data"].numerics[control_name] = value
                with self.assertRaisesRegex(ValueError, "under-resolved"):
                    resolve_native_numerical_envelope(contract)

        contract = _final_tier_contract()
        contract["perturbation_data"].numerics["momentum_grids"][
            "massive_neutrino_default"
        ]["count"] = 15
        with self.assertRaisesRegex(ValueError, "momentum_grids"):
            resolve_native_numerical_envelope(contract)

    def test_final_tier_requires_a_bounded_runtime_envelope(self) -> None:
        """A named final tier cannot silently omit bounded work limits."""

        contract = _final_tier_contract()
        contract["perturbation_data"].accuracy_controls = {
            "accuracy_tier": "final"
        }
        with self.assertRaisesRegex(ValueError, "runtime_envelope"):
            resolve_native_numerical_envelope(contract)

    def test_unknown_accuracy_tier_fails_loudly(self) -> None:
        """Unknown tiers cannot downgrade the final physical envelope."""

        contract = _final_tier_contract()
        accuracy_controls = contract["perturbation_data"].accuracy_controls
        accuracy_controls["accuracy_tier"] = "fast"
        with self.assertRaisesRegex(ValueError, "must be 'final'"):
            resolve_native_numerical_envelope(contract)

    def test_spectrum_report_uses_normalized_te_and_final_thresholds(
        self,
    ) -> None:
        """Final spectrum metrics apply the required per-surface bounds."""

        fine = {
            "TT": numpy.asarray((10.0, 20.0)),
            "EE": numpy.asarray((2.0, 4.0)),
            "TE": numpy.asarray((1.0, 2.0)),
            "PP": numpy.asarray((0.4, 0.8)),
            "lensed_BB": numpy.asarray((0.2, 0.3)),
        }
        coarse = {
            name: values * factor
            for name, values, factor in (
                ("TT", fine["TT"], 1.005),
                ("EE", fine["EE"], 1.005),
                ("TE", fine["TE"], 1.01),
                ("PP", fine["PP"], 1.02),
                ("lensed_BB", fine["lensed_BB"], 1.04),
            )
        }

        report = evaluate_spectrum_refinement(coarse, fine)

        self.assertTrue(report.converged)
        self.assertEqual(set(report.metrics), set(fine))
        self.assertLess(report.metrics["TE"].relative_error, 0.02)
        require_native_convergence(report)

    def test_control_report_rejects_q_and_hierarchy_drift(self) -> None:
        """Q-grid and hierarchy refinements retain their named thresholds."""

        q_metric = evaluate_control_refinement(
            numpy.asarray((1.0, 0.8)),
            numpy.asarray((1.0, 1.0)),
            name="massive-neutrino q grid",
            tolerance=FINAL_Q_GRID_RELATIVE_TOLERANCE,
        )
        hierarchy_metric = evaluate_control_refinement(
            numpy.asarray((1.0, 0.98)),
            numpy.asarray((1.0, 1.0)),
            name="tensor hierarchy",
            tolerance=FINAL_HIERARCHY_RELATIVE_TOLERANCE,
        )

        self.assertFalse(q_metric.converged)
        self.assertFalse(hierarchy_metric.converged)
        with self.assertRaisesRegex(ValueError, "q grid"):
            require_native_convergence(q_metric)
        with self.assertRaisesRegex(ValueError, "tensor hierarchy"):
            require_native_convergence(hierarchy_metric)

    def test_refinement_metric_remains_finite_at_spectrum_zeroes(self) -> None:
        """Relative L-infinity refinement must tolerate physical zeroes."""

        metric = evaluate_control_refinement(
            numpy.asarray((-1.0, 0.0, 1.0)),
            numpy.asarray((-1.0, 0.0, 1.01)),
            name="cross-zero surface",
            tolerance=0.02,
        )

        self.assertTrue(metric.converged)
        self.assertTrue(numpy.isfinite(metric.relative_error))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
