"""Direct tests for declared hierarchy-evolution ownership."""

import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.runtime import mode_evolution


class ModeEvolutionTestCase(unittest.TestCase):
    """Exercise hierarchy scheduling apart from public orchestration."""

    def test_generated_stage_hubble_rate_matches_declared_expansion(self):
        """RK stages must obey Hconf=aH/c after evaluating declared H(a)."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        metrics = {"maximum_relative_error": 0.0, "stage_count": 0}
        resolve = mode_evolution._resolve_declared_graph_context_ordered

        def observe(context, *args, **kwargs):
            if {"a", "H", "Hconf"}.issubset(context):
                expected = (
                    numpy.asarray(context["a"])
                    * numpy.asarray(context["H"])
                    / 299792.458
                )
                relative = numpy.abs(
                    numpy.asarray(context["Hconf"]) - expected
                ) / numpy.maximum(numpy.abs(expected), 1e-30)
                metrics["maximum_relative_error"] = max(
                    metrics["maximum_relative_error"],
                    float(numpy.max(relative)),
                )
                metrics["stage_count"] += 1
            return resolve(context, *args, **kwargs)

        contract = test_cmb._prepare_declared_contract(
            test_cmb._speedup_contract(
                test_cmb._declared_scalar_hierarchy_contract()
            )
        )
        with mock.patch.object(
            mode_evolution,
            "_resolve_declared_graph_context_ordered",
            side_effect=observe,
        ):
            test_cmb.cmb_execution._compute_custom_cmb_spectrum_data(
                contract,
                numpy.asarray((20, 21)),
                requested_spectra=("TT",),
                workload="fixed_parameter_diagnostic",
            )
        self.assertGreater(metrics["stage_count"], 10)
        self.assertLess(metrics["maximum_relative_error"], 1e-12)

    def test_batch_capability_requires_a_shared_declared_schedule(self):
        """Batching remains available only for compatible scalar modes."""

        common = {
            "generated_scalar_hierarchy": True,
            "shared_mode_grids_enabled": True,
            "mode_count": 2,
            "has_momentum_runtimes": False,
            "has_end_boundaries": False,
            "adaptive_evolution_enabled": False,
            "adaptive_source_enabled": False,
            "adaptive_transfer_enabled": False,
            "adaptive_projection_enabled": False,
            "adaptive_k_enabled": False,
            "continuous_collision_solver": False,
            "has_declared_collision_operators": False,
            "state_slots": (SimpleNamespace(wrt="eta"),),
            "collision_runtimes": (
                SimpleNamespace(activation_strategy="always"),
            ),
        }
        self.assertTrue(mode_evolution._can_batch_declared_evolution(**common))
        self.assertFalse(
            mode_evolution._can_batch_declared_evolution(
                **{**common, "shared_mode_grids_enabled": False}
            )
        )

    def test_evolution_runtime_exposes_explicit_input_and_output_types(self):
        """The hierarchy owner publishes typed composition boundaries."""

        self.assertTrue(callable(mode_evolution.build_declared_mode_evolution))
        input_type = mode_evolution.DeclaredModeEvolutionInputs
        self.assertIn(
            "execution_plan",
            {field.name for field in fields(input_type)},
        )
        runtime_names = {
            field.name
            for field in fields(mode_evolution.DeclaredModeEvolutionRuntime)
        }
        self.assertIn("evolve_declared_mode", runtime_names)
        self.assertIn("snapshot", runtime_names)

    def test_evolution_owner_has_no_diagnostics_or_evidence_import(self):
        """Physics execution cannot depend on its diagnostic consumers."""

        source = Path(mode_evolution.__file__).read_text(encoding="utf-8")
        self.assertNotIn(".diagnostics import", source)
        self.assertNotIn(".evidence import", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
