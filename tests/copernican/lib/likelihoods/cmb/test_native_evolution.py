"""Focused tests for the native CMB evolution module."""

import unittest
from pathlib import Path
from unittest import mock

from copernican.lib.likelihoods.cmb import native_background, native_evolution
from copernican.lib.perturbation_contract import compile_perturbation_contract


def _compiled_graph_fixture():
    """Return one small declared graph with compiled expression entries."""

    contract = {
        "contract_version": 2,
        "standard": False,
        "gauge": "conformal_newtonian",
        "variables": {
            "delta_x": {"kind": "density_contrast"},
            "theta_x": {"kind": "velocity_divergence"},
            "phi_aux": {"kind": "metric_potential_phi"},
            "psi_aux": {"kind": "metric_potential_psi"},
        },
        "derived": {
            "density_drive": {"expression": "delta_x + phi_aux"},
        },
        "equations": {
            "continuity_x": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_x",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_x + phi_aux",
                "role": "continuity",
            },
            "euler_x": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "theta_x",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-Hconf * theta_x + k * psi_aux",
                "role": "euler",
            },
        },
        "constraints": {
            "poisson_phi": {
                "target": "phi_aux",
                "expression": "0.25 * delta_x",
                "role": "constraint",
            }
        },
        "closures": {
            "psi_equals_phi": {
                "target": "psi_aux",
                "expression": "phi_aux",
                "role": "closure",
            }
        },
        "sources": {
            "monopole_source": {
                "expression": "visibility * density_drive",
                "role": "monopole",
            }
        },
        "observables": {
            "temperature": {
                "kind": "transfer_component",
                "projection": "line_of_sight_temperature",
                "source_terms": {"monopole": "monopole_source"},
            },
            "TT": {
                "kind": "angular_power_spectrum",
                "primary": "temperature",
                "secondary": "temperature",
            },
        },
        "initial_conditions": {
            "delta_seed": {
                "target": {
                    "variable": "delta_x",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "seed",
            },
            "theta_seed": {
                "target": {
                    "variable": "theta_x",
                    "wrt": "tau",
                    "order": 0,
                },
                "expression": "0.1 * seed",
            },
        },
        "boundary_conditions": {},
        "numerics": {},
        "validity": {"regimes": ["linear"]},
        "backend_mapping": {
            "camb": {
                "native_solver_required": True,
                "implemented": True,
            }
        },
    }
    return compile_perturbation_contract(
        contract,
        model_name="TemplateModel",
        backend="camb",
        parameter_names=("seed",),
        latex_names=("seed",),
        background_reference_names=("Hconf", "k", "seed", "visibility"),
    )


class NativeEvolutionModuleTestCase(unittest.TestCase):
    """Exercise native evolution helpers directly."""

    def test_precompiled_perturbation_payload_is_reused(self):
        """Existing compiled perturbation data should bypass recompilation."""

        payload = object()
        contract = {"perturbation_data": payload}

        with mock.patch(
            "copernican.lib.perturbation_contract."
            "compile_perturbation_contract",
            side_effect=AssertionError("recompilation should not run"),
        ):
            compiled = (
                native_evolution._compile_declared_perturbation_contract(
                    contract
                )
            )

        self.assertIs(compiled, payload)

    def test_declared_graph_context_uses_compiled_expression_plans(self):
        """Declared graph resolution should avoid AST re-interpretation."""

        perturbation_data = _compiled_graph_fixture()
        execution_plan = (
            native_evolution._compile_declared_graph_execution_plan(
                perturbation_data
            )
        )
        context = {"delta_x": 2.0}

        with mock.patch.object(
            native_background,
            "_evaluate_safe_expression",
            side_effect=AssertionError("compiled graph plan should run"),
        ):
            resolved = native_evolution._resolve_declared_graph_context(
                context,
                perturbation_data,
                allow_partial=False,
                eta_grid=None,
                execution_plan=execution_plan,
            )

        self.assertAlmostEqual(resolved["phi_aux"], 0.5)
        self.assertAlmostEqual(resolved["psi_aux"], 0.5)
        self.assertAlmostEqual(resolved["density_drive"], 2.5)

    def test_native_evolution_source_does_not_import_camb(self):
        """The native evolution module should remain CAMB-free."""

        source_text = Path(native_evolution.__file__).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("import camb", source_text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
