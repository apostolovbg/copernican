"""Direct tests for declared numerical grid and work controls."""

import unittest
from types import SimpleNamespace

from copernican.lib.likelihoods.cmb.runtime import numerical_controls


class NumericalControlsTestCase(unittest.TestCase):
    """Exercise numerical-control decisions without spectrum orchestration."""

    def test_background_density_is_only_an_initial_resolution_candidate(self):
        contract = {
            "_engine_numerical_plan": {
                "eta_sample_count": 192,
                "evolution_eta_sample_count": 128,
            },
            "_engine_accuracy_controls": {
                "adaptive_evolution": {
                    "enabled": True,
                    "maximum_nodes": 512,
                    "relative_tolerance": 0.01,
                },
                "adaptive_source": {
                    "enabled": True,
                    "minimum_nodes": 32,
                    "maximum_nodes": 512,
                    "relative_tolerance": 0.02,
                },
                "adaptive_projection": False,
            },
        }
        self.assertIsInstance(
            numerical_controls.seed_resolution_from_background(contract, 3796),
            numerical_controls.InitialResolutionCandidate,
        )
        candidate = numerical_controls.seed_resolution_from_background(
            contract, 3796
        )
        controls = candidate.contract["_engine_accuracy_controls"]
        self.assertEqual(
            candidate.contract["_engine_numerical_plan"][
                "evolution_eta_sample_count"
            ],
            3796,
        )
        self.assertEqual(controls["adaptive_source"]["minimum_nodes"], 7592)
        self.assertEqual(controls["adaptive_source"]["maximum_nodes"], 15184)
        self.assertEqual(controls["adaptive_projection"], False)
        self.assertEqual(
            controls["adaptive_source"]["relative_tolerance"], 0.02
        )
        self.assertEqual(
            candidate.contract["_engine_numerical_plan"]["eta_sample_count"],
            192,
        )
        self.assertNotIn("background_minimum_nodes", controls)
        self.assertEqual(
            contract["_engine_numerical_plan"]["evolution_eta_sample_count"],
            128,
        )
        self.assertFalse(candidate.evidence["acceptance_reused"])
        self.assertIsNone(
            numerical_controls.seed_resolution_from_background(
                candidate.contract, 3796
            )
        )
        changed, _ = numerical_controls.refine_numerical_axis(
            candidate.contract, "background"
        )
        self.assertEqual(
            changed["_engine_numerical_plan"],
            candidate.contract["_engine_numerical_plan"],
        )
        self.assertEqual(
            changed["_engine_accuracy_controls"]["adaptive_source"],
            controls["adaptive_source"],
        )
        with self.assertRaises(ValueError):
            numerical_controls.seed_resolution_from_background(contract, 1)

    def test_unrecognized_failure_does_not_promote_a_grid(self):
        self.assertIsNone(
            numerical_controls.promote_failed_resolution(
                {}, SimpleNamespace(label="unrelated", evidence={})
            )
        )

    def test_axis_promotion_preserves_accuracy_and_materializes_new_work(self):
        contract = {
            "_engine_numerical_plan": {
                "eta_sample_count": 192,
                "evolution_eta_sample_count": 128,
                "k_sample_count": 64,
                "photon_hierarchy_l_max": 28,
                "neutrino_hierarchy_l_max": 28,
                "momentum_grids": {
                    "q": {"count": 32, "q_min": 0.02, "q_max": 24}
                },
            },
            "_engine_accuracy_controls": {
                "adaptive_evolution": {"relative_tolerance": 0.01},
                "adaptive_source": {"relative_tolerance": 0.02},
            },
        }
        for axis in (
            "background",
            "hierarchy_depth",
            "momentum_q",
            "evolution",
            "source",
            "projection",
            "k",
        ):
            changed, record = numerical_controls.refine_numerical_axis(
                contract, axis
            )
            self.assertEqual(record["axis"], axis)
            self.assertEqual(
                changed["_engine_accuracy_controls"]["adaptive_evolution"][
                    "relative_tolerance"
                ],
                0.01,
            )
            if axis in {"source", "projection", "background"}:
                self.assertEqual(
                    changed["_engine_numerical_plan"],
                    contract["_engine_numerical_plan"],
                )
                self.assertNotEqual(
                    changed["_engine_accuracy_controls"],
                    contract["_engine_accuracy_controls"],
                )
            else:
                self.assertNotEqual(
                    changed["_engine_numerical_plan"],
                    contract["_engine_numerical_plan"],
                )
        changed, record = numerical_controls.promote_failed_resolution(
            contract,
            SimpleNamespace(
                label="observable-error-budget",
                evidence={
                    "products": {
                        "TT": {
                            "relative_tolerance": 0.01,
                            "relative_error_bound": 0.031,
                            "axis_relative_bounds": {
                                "source": 0.03,
                                "k": 0.001,
                            },
                        }
                    }
                },
            ),
        )
        self.assertEqual(record["axis"], "source")
        # refine_numerical_axis preserves the source axis's physical request.
        self.assertEqual(
            changed["_engine_accuracy_controls"]["adaptive_source"][
                "minimum_nodes"
            ],
            384,
        )

    def test_budget_search_uses_measured_axis_improvement(self):
        """A flat source bound cannot monopolize joint-budget refinement."""

        contract = {
            "_engine_numerical_plan": {
                "eta_sample_count": 192,
                "evolution_eta_sample_count": 128,
            },
            "_engine_accuracy_controls": {
                "adaptive_evolution": {"relative_tolerance": 0.01},
                "adaptive_source": {"relative_tolerance": 0.01},
            },
        }
        products = {
            "TT": {
                "relative_error_bound": 0.011,
                "relative_tolerance": 0.01,
                "axis_relative_bounds": {
                    "source": 0.0045,
                    "evolution": 0.004,
                    "background": 0.0025,
                },
            }
        }
        error = SimpleNamespace(
            label="observable-error-budget", evidence={"products": products}
        )
        changed, first = numerical_controls.promote_failed_resolution(
            contract, error
        )
        self.assertEqual(first["axis"], "source")
        products["TT"]["axis_relative_bounds"]["source"] = 0.0046
        products["TT"]["relative_error_bound"] = 0.0111
        changed, next_trial = numerical_controls.promote_failed_resolution(
            changed, error, attempts=(first,)
        )
        self.assertEqual(next_trial["axis"], "evolution")
        self.assertEqual(next_trial["unproductive_budget_axes"], ["source"])
        self.assertAlmostEqual(next_trial["relative_error"], 1.11)
        self.assertEqual(
            changed["_numerical_overrides"]["evolution_eta_sample_count"], 256
        )
        self.assertEqual(
            changed["_engine_accuracy_controls"]["adaptive_source"][
                "relative_tolerance"
            ],
            0.01,
        )
        # Cross normalization can magnify the total violation without
        # changing which measured axis is a material alternative.
        products["TE"] = {
            "relative_error_bound": 0.0222,
            "normalized_cross_bound": 0.0666,
            "relative_tolerance": 0.02,
            "axis_relative_bounds": {
                "source": 0.0092,
                "evolution": 0.008,
                "background": 0.005,
            },
        }
        products["EE"] = {
            "relative_error_bound": 0.004,
            "relative_tolerance": 0.01,
            "axis_relative_bounds": {
                "source": 0.002,
                "evolution": 0.0015,
                "background": 0.0005,
            },
        }
        _, correlated = numerical_controls.promote_failed_resolution(
            contract, error, attempts=(first,)
        )
        self.assertEqual(correlated["axis"], "evolution")
        self.assertAlmostEqual(correlated["relative_error"], 3.33)
        products.pop("TE")
        products.pop("EE")
        # A productive source trial remains eligible; history is a search
        # hint, never permission to accept an unchanged failed budget.
        products["TT"]["axis_relative_bounds"]["source"] = 0.004
        products["TT"]["axis_relative_bounds"]["evolution"] = 0.0039
        products["TT"]["relative_error_bound"] = 0.0104
        _, productive = numerical_controls.promote_failed_resolution(
            contract, error, attempts=(first,)
        )
        self.assertEqual(productive["axis"], "source")
        self.assertEqual(productive["unproductive_budget_axes"], [])
        # A tiny remaining contribution cannot fix a material cross-budget
        # deficit. Retry a material axis instead of refining the tiny one.
        error.evidence["products"] = {
            "TE": {
                "relative_error_bound": 0.01720002,
                "normalized_cross_bound": 0.0222,
                "relative_tolerance": 0.02,
                "axis_relative_bounds": {
                    "source": 0.0092,
                    "evolution": 0.008,
                    "background": 0.00000002,
                },
            }
        }
        _, material = numerical_controls.promote_failed_resolution(
            contract, error, attempts=(first, next_trial)
        )
        self.assertEqual(material["axis"], "source")
        self.assertEqual(
            material["unproductive_budget_axes"], ["evolution", "source"]
        )

    def test_background_refinement_starts_from_accepted_seed_only(self):
        contract = {"_engine_numerical_plan": {"eta_sample_count": 192}}
        changed, record = numerical_controls.refine_numerical_axis(
            contract,
            "background",
            evidence={"background_resolution_seed": 3072},
        )
        source = numerical_controls._resolve_custom_cmb_numerics(changed)
        self.assertEqual(source.eta_sample_count, 192)
        self.assertEqual(
            numerical_controls.resolve_background_numerics(
                changed, source
            ).eta_sample_count,
            6144,
        )
        self.assertEqual(record["next_background_minimum_nodes"], 6144)
        self.assertEqual(
            changed["_engine_numerical_plan"],
            contract["_engine_numerical_plan"],
        )

    def test_projection_promotion_preserves_disabled_source_control(self):
        contract = {
            "_engine_numerical_plan": {"eta_sample_count": 128},
            "_engine_accuracy_controls": {
                "adaptive_source": False,
                "adaptive_projection": {
                    "minimum_nodes": 64,
                    "maximum_nodes": 256,
                    "relative_tolerance": 0.01,
                },
            },
        }
        changed, _ = numerical_controls.refine_numerical_axis(
            contract, "projection"
        )
        self.assertIs(
            changed["_engine_accuracy_controls"]["adaptive_source"], False
        )
        self.assertEqual(
            changed["_engine_accuracy_controls"]["adaptive_projection"][
                "minimum_nodes"
            ],
            256,
        )
        self.assertEqual(
            changed["_engine_numerical_plan"]["eta_sample_count"], 128
        )

    def test_runtime_work_estimate_is_deterministic_and_accounted(self):
        """Large bounded requests are accounted for, not rejected."""

        contract = {
            "perturbations": {
                "accuracy_controls": {"runtime_envelope": "bounded"}
            }
        }
        arguments = {
            "ell_count": 2500,
            "k_count": 2048,
            "eta_count": 2048,
            "state_slot_count": 64,
            "transfer_component_count": 3,
            "momentum_point_count": 256,
            "evolution_multiplier": 3,
        }
        first = numerical_controls._enforce_runtime_envelope(
            contract, **arguments
        )
        second = numerical_controls._enforce_runtime_envelope(
            contract, **arguments
        )
        self.assertEqual(first, second)
        self.assertEqual(first["work_accounting_mode"], "accounted")
        self.assertEqual(first["work_limits"], {})
        self.assertFalse(first["work_limits_enforced"])
        self.assertGreater(first["total_work_units"], 100_000_000)

    def test_explicit_work_limit_is_metadata_not_a_machine_ceiling(self):
        """A valid request is not rejected by an operator work hint."""

        contract = {
            "perturbations": {
                "accuracy_controls": {
                    "runtime_envelope": {
                        "maximum_total_work_units": 1,
                    }
                }
            }
        }
        envelope = numerical_controls._enforce_runtime_envelope(
            contract,
            ell_count=100,
            k_count=100,
            eta_count=100,
            state_slot_count=8,
            transfer_component_count=2,
            momentum_point_count=0,
        )
        self.assertEqual(
            envelope["work_limits"], {"maximum_total_work_units": 1}
        )
        self.assertFalse(envelope["work_limits_enforced"])

    def test_evolution_chunk_size_is_deterministic(self):
        """Evolution chunking is derived from declared array dimensions."""

        first = numerical_controls._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        second = numerical_controls._resolve_evolution_chunk_size(
            k_count=2048,
            eta_count=2048,
            state_slot_count=64,
        )
        self.assertEqual(first, second)
        self.assertGreaterEqual(first, 1)
        self.assertLessEqual(first * 2048 * 64, 16_000_000)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
