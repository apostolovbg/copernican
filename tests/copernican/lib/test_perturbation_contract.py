"""Tests for the declared perturbation-graph compiler."""

from __future__ import annotations

import unittest

import copernican.lib.perturbation_contract as perturbation_contract_module
from copernican.lib.perturbation_contract import (
    PerturbationBackendMappingData,
    PerturbationClosureData,
    PerturbationCompiledExpressionData,
    PerturbationConditionData,
    PerturbationConditionTargetData,
    PerturbationConstraintData,
    PerturbationContractData,
    PerturbationDependencyGraphSummaryData,
    PerturbationDerivativeLhsData,
    PerturbationDerivedData,
    PerturbationEquationData,
    PerturbationObservableData,
    PerturbationSourceData,
    PerturbationValidityData,
    PerturbationVariableData,
    compile_perturbation_contract,
    evaluate_compiled_expression,
)


def _base_nonstandard_contract() -> dict[str, object]:
    """Return a reusable declared-math perturbation graph fixture."""

    return {
        "contract_version": 2,
        "standard": False,
        "gauge": "conformal_newtonian",
        "variables": {
            "delta_x": {
                "kind": "density_contrast",
                "description": "Synthetic density perturbation.",
                "tensor_character": "scalar_like",
                "rank": 0,
                "spin": 0.0,
            },
            "theta_x": {
                "kind": "velocity_divergence",
                "description": "Synthetic velocity perturbation.",
                "tensor_character": "scalar_like",
                "rank": 0,
                "spin": 0.0,
            },
            "phi_aux": {
                "kind": "metric_potential_phi",
                "gauge_role": "newtonian_potential",
            },
            "psi_aux": {
                "kind": "metric_potential_psi",
                "gauge_role": "curvature_potential",
            },
        },
        "derived": {
            "density_drive": {
                "expression": "delta_x + phi_aux",
                "description": "Synthetic driving term.",
            }
        },
        "equations": {
            "evolve_delta_x": {
                "lhs": {
                    "kind": "derivative",
                    "variable": "delta_x",
                    "wrt": "tau",
                    "order": 1,
                },
                "rhs": "-theta_x + phi_aux",
                "role": "continuity",
            },
            "evolve_theta_x": {
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
            },
            "polarization_source": {
                "expression": "visibility * theta_x",
                "role": "polarization",
            },
        },
        "observables": {
            "temperature": {
                "kind": "transfer_component",
                "projection": "line_of_sight_temperature",
                "source_terms": {"monopole": "monopole_source"},
            },
            "polarization_e": {
                "kind": "transfer_component",
                "projection": "line_of_sight_polarization_e",
                "source_terms": {"polarization": "polarization_source"},
            },
            "TT": {
                "kind": "angular_power_spectrum",
                "primary": "temperature",
                "secondary": "temperature",
            },
            "TE": {
                "kind": "angular_power_spectrum",
                "primary": "temperature",
                "secondary": "polarization_e",
            },
            "EE": {
                "kind": "angular_power_spectrum",
                "primary": "polarization_e",
                "secondary": "polarization_e",
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
        "numerics": {
            "ode_rtol": 1.0e-5,
            "ode_atol": 1.0e-8,
        },
        "validity": {
            "regimes": ["linear", "synthetic"],
            "notes": "Synthetic declared graph for compiler tests.",
        },
        "backend_mapping": {
            "camb": {
                "native_solver_required": True,
                "implemented": True,
            }
        },
    }


class PerturbationContractTestCase(unittest.TestCase):
    """Validate the typed perturbation graph compiler."""

    def _compile(
        self, contract: dict[str, object]
    ) -> PerturbationContractData:
        """Compile one test contract with a stable environment."""

        return compile_perturbation_contract(
            contract,
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )

    def test_module_symbols_are_exported(self) -> None:
        """The module should export the declared graph data symbols."""

        self.assertIs(
            perturbation_contract_module.compile_perturbation_contract,
            compile_perturbation_contract,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationContractData,
            PerturbationContractData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationConstraintData,
            PerturbationConstraintData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationObservableData,
            PerturbationObservableData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationConditionData,
            PerturbationConditionData,
        )
        self.assertIs(
            perturbation_contract_module.evaluate_compiled_expression,
            evaluate_compiled_expression,
        )

    def test_compiled_expression_evaluator_returns_numeric_result(self):
        """Compiled expression plans should evaluate without AST reparse."""

        expression_data = PerturbationCompiledExpressionData(
            expression="delta_x + phi_aux",
            dependencies=("delta_x", "phi_aux"),
            program=(
                ("name", "delta_x"),
                ("name", "phi_aux"),
                ("binary", "add"),
            ),
        )

        result = evaluate_compiled_expression(
            expression_data,
            {"delta_x": 2.0, "phi_aux": 0.5},
        )

        self.assertEqual(result, 2.5)

    def test_standard_contract_compiles(self) -> None:
        """Standard contracts should compile into immutable data."""

        standard_contract_data = self._compile(
            {
                "contract_version": 2,
                "standard": True,
                "gauge": "unspecified",
                "variables": {},
                "derived": {},
                "equations": {},
                "constraints": {},
                "closures": {},
                "sources": {},
                "observables": {},
                "initial_conditions": {},
                "boundary_conditions": {},
                "numerics": {},
                "validity": {
                    "regimes": ["standard_camb"],
                    "notes": "Uses standard backend perturbations.",
                },
                "backend_mapping": {
                    "camb": {"uses_standard_perturbations": True}
                },
            }
        )

        self.assertIsInstance(standard_contract_data, PerturbationContractData)
        self.assertTrue(standard_contract_data.standard)
        self.assertEqual(standard_contract_data.gauge, "unspecified")
        self.assertTrue(
            standard_contract_data.backend_mapping[
                "camb"
            ].uses_standard_perturbations
        )
        self.assertEqual(
            standard_contract_data.manifest_summary["execution_route"],
            {
                "route_id": "backend_standard_perturbations",
                "prediction_engine": "camb",
                "transfer_function_path": "camb.standard",
                "solver": "camb_standard",
                "route_ready_for_execution": True,
                "uses_backend_standard_perturbations": True,
                "uses_native_declared_graph": False,
                "uses_camb_prediction": True,
                "uses_camb_standard_perturbations": True,
                "backend_mapping_implemented": None,
                "backend_mapping_native_solver_required": None,
                "backend_mapping_uses_standard_perturbations": True,
            },
        )

    def test_nonstandard_contract_compiles(self) -> None:
        """Non-standard contracts should preserve graph metadata."""

        contract_data = self._compile(_base_nonstandard_contract())

        self.assertFalse(contract_data.standard)
        self.assertEqual(contract_data.contract_version, 2)
        self.assertEqual(
            contract_data.equations["evolve_delta_x"].lhs.variable,
            "delta_x",
        )
        self.assertIsInstance(
            contract_data.variables["delta_x"],
            PerturbationVariableData,
        )
        self.assertIsInstance(
            contract_data.derived["density_drive"],
            PerturbationDerivedData,
        )
        self.assertIsInstance(
            contract_data.derived["density_drive"].compiled_expression,
            PerturbationCompiledExpressionData,
        )
        self.assertIsInstance(
            contract_data.equations["evolve_delta_x"].lhs,
            PerturbationDerivativeLhsData,
        )
        self.assertIsInstance(
            contract_data.equations["evolve_delta_x"],
            PerturbationEquationData,
        )
        self.assertIsInstance(
            contract_data.equations["evolve_delta_x"].compiled_rhs,
            PerturbationCompiledExpressionData,
        )
        self.assertEqual(
            contract_data.constraints["poisson_phi"].target,
            "phi_aux",
        )
        self.assertIsInstance(
            contract_data.constraints["poisson_phi"],
            PerturbationConstraintData,
        )
        self.assertIsInstance(
            contract_data.constraints["poisson_phi"].compiled_expression,
            PerturbationCompiledExpressionData,
        )
        self.assertIsInstance(
            contract_data.closures["psi_equals_phi"],
            PerturbationClosureData,
        )
        self.assertIsInstance(
            contract_data.closures["psi_equals_phi"].compiled_expression,
            PerturbationCompiledExpressionData,
        )
        self.assertIsInstance(
            contract_data.sources["monopole_source"],
            PerturbationSourceData,
        )
        self.assertIsInstance(
            contract_data.sources["monopole_source"].compiled_expression,
            PerturbationCompiledExpressionData,
        )
        self.assertEqual(
            contract_data.observables["TT"].primary,
            "temperature",
        )
        self.assertIsInstance(
            contract_data.observables["temperature"],
            PerturbationObservableData,
        )
        self.assertEqual(
            contract_data.observables["temperature"].kernel,
            "temperature_mixed_window",
        )
        self.assertEqual(
            contract_data.initial_conditions["delta_seed"].target.order,
            0,
        )
        self.assertIsInstance(
            contract_data.initial_conditions["delta_seed"].target,
            PerturbationConditionTargetData,
        )
        self.assertIsInstance(
            contract_data.initial_conditions["delta_seed"],
            PerturbationConditionData,
        )
        self.assertIsInstance(
            contract_data.initial_conditions["delta_seed"].compiled_expression,
            PerturbationCompiledExpressionData,
        )
        dependency_summary = contract_data.dependency_graph_summary
        self.assertIsInstance(
            contract_data.validity,
            PerturbationValidityData,
        )
        self.assertIsInstance(
            contract_data.backend_mapping["camb"],
            PerturbationBackendMappingData,
        )
        self.assertIsInstance(
            dependency_summary,
            PerturbationDependencyGraphSummaryData,
        )
        self.assertIn("seed", dependency_summary.background_references_used)
        self.assertIn("phi_aux", dependency_summary.evaluation_order)
        self.assertEqual(
            contract_data.manifest_summary["equation_wrt_by_variable"],
            {
                "delta_x": "tau",
                "theta_x": "tau",
            },
        )
        self.assertEqual(
            contract_data.manifest_summary["boundary_condition_anchors"],
            {},
        )
        self.assertEqual(
            contract_data.manifest_summary["transfer_component_contracts"][
                "temperature"
            ]["kernel"],
            "temperature_mixed_window",
        )
        self.assertEqual(
            contract_data.manifest_summary["transfer_component_contracts"][
                "polarization_e"
            ]["source_term_roles"],
            ("polarization",),
        )
        self.assertEqual(
            contract_data.manifest_summary["angular_power_spectrum_targets"][
                "TT"
            ],
            {
                "primary": "temperature",
                "secondary": "temperature",
            },
        )
        self.assertEqual(
            contract_data.manifest_summary["execution_route"],
            {
                "route_id": "native_declared_graph",
                "prediction_engine": "copernican_native_declared_graph",
                "transfer_function_path": (
                    "copernican.lib.likelihoods.cmb." "copernican_cmb_solver"
                ),
                "solver": "declared_math_graph",
                "route_ready_for_execution": True,
                "uses_backend_standard_perturbations": False,
                "uses_native_declared_graph": True,
                "uses_camb_prediction": False,
                "uses_camb_standard_perturbations": False,
                "backend_mapping_implemented": True,
                "backend_mapping_native_solver_required": True,
                "backend_mapping_uses_standard_perturbations": None,
            },
        )

    def test_hybrid_graph_compiles_as_one_graph(self) -> None:
        """Tagged scalar/vector/tensor variables should share one graph."""

        contract = _base_nonstandard_contract()
        contract["variables"]["vector_mode"] = {
            "kind": "custom_vector_mode",
            "rank": 1,
            "spin": 1.0,
            "parity": "odd",
            "tensor_character": "vector_like",
        }
        contract["variables"]["tensor_mode"] = {
            "kind": "custom_tensor_mode",
            "rank": 2,
            "spin": 2.0,
            "parity": "even",
            "tensor_character": "tensor_like",
        }
        contract["equations"]["evolve_vector_mode"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "vector_mode",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-Hconf * vector_mode + tensor_mode",
            "role": "vector_coupling",
        }
        contract["equations"]["evolve_tensor_mode"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "tensor_mode",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-0.5 * tensor_mode + vector_mode",
            "role": "tensor_coupling",
        }
        contract["initial_conditions"]["vector_seed"] = {
            "target": {
                "variable": "vector_mode",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
        contract["initial_conditions"]["tensor_seed"] = {
            "target": {
                "variable": "tensor_mode",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }

        contract_data = self._compile(contract)

        self.assertIn("vector_mode", contract_data.variables)
        self.assertIn("tensor_mode", contract_data.variables)
        self.assertIn(
            "psi_aux",
            contract_data.dependency_graph_summary.evaluation_order,
        )

    def test_missing_initial_conditions_fail(self) -> None:
        """Each evolved variable should require explicit initial data."""

        contract = _base_nonstandard_contract()
        del contract["initial_conditions"]["theta_seed"]

        with self.assertRaisesRegex(
            ValueError,
            "missing required initial conditions",
        ):
            self._compile(contract)

    def test_missing_observables_fail(self) -> None:
        """Non-standard graphs should declare observable mappings."""

        contract = _base_nonstandard_contract()
        contract["observables"] = {}

        with self.assertRaisesRegex(
            ValueError,
            "must declare observables",
        ):
            self._compile(contract)

    def test_duplicate_relation_targets_fail(self) -> None:
        """Duplicate closure or constraint targets should be rejected."""

        contract = _base_nonstandard_contract()
        contract["closures"]["psi_again"] = {
            "target": "psi_aux",
            "expression": "phi_aux",
            "role": "closure",
        }

        with self.assertRaisesRegex(ValueError, "duplicates target 'psi_aux'"):
            self._compile(contract)

    def test_circular_derived_dependencies_fail(self) -> None:
        """Circular derived expressions should fail clearly."""

        contract = _base_nonstandard_contract()
        contract["derived"]["alpha"] = {"expression": "beta"}
        contract["derived"]["beta"] = {"expression": "alpha"}

        with self.assertRaisesRegex(ValueError, "contains a cycle"):
            self._compile(contract)

    def test_unknown_transfer_component_reference_fails(self) -> None:
        """Power spectra should reference declared transfer components."""

        contract = _base_nonstandard_contract()
        contract["observables"]["TT"]["primary"] = "unknown_component"

        with self.assertRaisesRegex(
            ValueError,
            "unknown transfer component 'unknown_component'",
        ):
            self._compile(contract)

    def test_unsupported_projection_fails_during_compilation(self) -> None:
        """Transfer projections should be validated before runtime."""

        contract = _base_nonstandard_contract()
        temperature_observable = contract["observables"]["temperature"]
        temperature_observable["projection"] = "bogus_projection"

        with self.assertRaisesRegex(ValueError, "unsupported projection"):
            self._compile(contract)

    def test_projection_source_role_mismatch_fails(self) -> None:
        """Projection source-role requirements should fail clearly."""

        contract = _base_nonstandard_contract()
        contract["observables"]["polarization_e"]["source_terms"] = {
            "signal": "polarization_source"
        }

        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles",
        ):
            self._compile(contract)

    def test_custom_line_of_sight_projection_compiles_with_kernel(
        self,
    ) -> None:
        """Custom transfer components should record kernel provenance."""

        contract = _base_nonstandard_contract()
        contract["variables"]["tensor_b"] = {
            "kind": "custom_tensor_polarization_source",
            "projection_role": "b_mode",
            "rank": 2,
            "spin": 2.0,
            "parity": "odd",
            "tensor_character": "tensor_like",
        }
        contract["equations"]["evolve_tensor_b"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-0.1 * tensor_b + theta_x",
            "role": "odd_parity_polarization",
        }
        contract["initial_conditions"]["tensor_b_seed"] = {
            "target": {
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }
        contract["sources"]["custom_b_source"] = {
            "expression": "visibility * tensor_b",
            "role": "polarization_b",
        }
        contract["observables"]["custom_b"] = {
            "kind": "transfer_component",
            "projection": "custom_line_of_sight",
            "kernel": "spin2_b_window",
            "source_terms": {"polarization_b": "custom_b_source"},
            "required_projection_roles": ["b_mode"],
        }

        contract_data = self._compile(contract)

        self.assertEqual(
            contract_data.observables["custom_b"].kernel,
            "spin2_b_window",
        )
        self.assertEqual(
            contract_data.observables["custom_b"].required_projection_roles,
            ("b_mode",),
        )

    def test_custom_line_of_sight_requires_kernel(self) -> None:
        """Custom transfer components should fail without a kernel."""

        contract = _base_nonstandard_contract()
        contract["observables"]["temperature"] = {
            "kind": "transfer_component",
            "projection": "custom_line_of_sight",
            "source_terms": {"monopole": "monopole_source"},
        }

        with self.assertRaisesRegex(ValueError, "must declare kernel"):
            self._compile(contract)

    def test_angular_power_targets_reject_projection_metadata(self) -> None:
        """Angular spectra should stay separate from transfer machinery."""

        contract = _base_nonstandard_contract()
        contract["observables"]["TT"]["projection"] = "line_of_sight_signal"

        with self.assertRaisesRegex(
            ValueError,
            "must not declare projection or source_terms",
        ):
            self._compile(contract)

    def test_lensing_projection_requires_declared_potential_role(self) -> None:
        """Lensing projections should reject generic signal-role bindings."""

        contract = _base_nonstandard_contract()
        contract["sources"]["lensing_source"] = {
            "expression": "phi_aux + psi_aux",
            "role": "signal",
        }
        contract["observables"]["lensing"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {"signal": "lensing_source"},
        }

        with self.assertRaisesRegex(
            ValueError,
            "requires the source-term roles: potential",
        ):
            self._compile(contract)

    def test_b_mode_projection_requires_odd_parity_source_ancestry(
        self,
    ) -> None:
        """B-mode projections should reject scalar-like source ancestry."""

        contract = _base_nonstandard_contract()
        contract["sources"]["polarization_b_source"] = {
            "expression": "visibility * theta_x",
            "role": "polarization_b",
        }
        contract["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {"polarization_b": "polarization_b_source"},
        }

        with self.assertRaisesRegex(
            ValueError,
            "requires an odd-parity declared source ancestry",
        ):
            self._compile(contract)

    def test_unsolved_variable_references_fail(self) -> None:
        """Referenced variables must be evolved or algebraically solved."""

        contract = _base_nonstandard_contract()
        contract["variables"]["chi_aux"] = {"kind": "custom_auxiliary_mode"}
        monopole_source = contract["sources"]["monopole_source"]
        monopole_source["expression"] = (
            "visibility * (density_drive + chi_aux)"
        )

        with self.assertRaisesRegex(
            ValueError,
            "without evolution or algebraic definitions",
        ):
            self._compile(contract)

    def test_relation_target_derivative_symbols_compile(self) -> None:
        """Derivative symbols may target algebraic relation outputs."""

        contract = _base_nonstandard_contract()
        contract["derived"]["phi_tau"] = {
            "kind": "derivative_symbol",
            "variable": "phi_aux",
            "wrt": "tau",
            "order": 1,
        }
        monopole_source = contract["sources"]["monopole_source"]
        monopole_source["expression"] = (
            "visibility * (density_drive + phi_tau)"
        )

        contract_data = self._compile(contract)

        self.assertIn("phi_tau", contract_data.derived)

    def test_start_boundary_conditions_satisfy_missing_initial_data(
        self,
    ) -> None:
        """Start boundary conditions may provide native-solver seed data."""

        contract = _base_nonstandard_contract()
        theta_seed = contract["initial_conditions"].pop("theta_seed")
        theta_seed["anchor"] = "start"
        contract["boundary_conditions"]["theta_start"] = theta_seed

        contract_data = self._compile(contract)

        self.assertIn("theta_start", contract_data.boundary_conditions)

    def test_end_boundary_conditions_satisfy_missing_initial_data(
        self,
    ) -> None:
        """End boundary conditions may replace missing start-state slots."""

        contract = _base_nonstandard_contract()
        theta_seed = contract["initial_conditions"].pop("theta_seed")
        theta_seed["anchor"] = "end"
        contract["boundary_conditions"]["theta_end"] = theta_seed

        contract_data = self._compile(contract)

        self.assertIn("theta_end", contract_data.boundary_conditions)
        self.assertEqual(
            contract_data.manifest_summary["boundary_condition_anchors"],
            {"theta_end": "end"},
        )

    def test_duplicate_start_boundary_target_fails(self) -> None:
        """Initial and start-boundary conditions may not target one slot."""

        contract = _base_nonstandard_contract()
        contract["boundary_conditions"]["delta_again"] = {
            "target": {
                "variable": "delta_x",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "seed",
            "anchor": "start",
        }

        with self.assertRaisesRegex(
            ValueError,
            "duplicate targets",
        ):
            self._compile(contract)

    def test_conditions_may_only_target_evolved_state_slots(self) -> None:
        """Initial data may not target purely algebraic relation variables."""

        contract = _base_nonstandard_contract()
        contract["initial_conditions"]["phi_seed"] = {
            "target": {
                "variable": "phi_aux",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.0",
        }

        with self.assertRaisesRegex(
            ValueError,
            "may only target declared differential state slots",
        ):
            self._compile(contract)

    def test_perturbation_dataclasses_are_constructible(self) -> None:
        """The typed perturbation objects should be individually usable."""

        variable_data = PerturbationVariableData(
            name="delta_x",
            kind="density_contrast",
        )
        derived_data = PerturbationDerivedData(
            name="density_drive",
            kind="derived_quantity",
            expression="delta_x + phi_aux",
        )
        lhs_data = PerturbationDerivativeLhsData(
            kind="derivative",
            variable="delta_x",
            wrt="tau",
            order=1,
        )
        equation_data = PerturbationEquationData(
            name="evolve_delta_x",
            lhs=lhs_data,
            rhs="-theta_x + phi_aux",
            role="continuity",
        )
        constraint_data = PerturbationConstraintData(
            name="poisson_phi",
            target="phi_aux",
            expression="0.25 * delta_x",
            role="constraint",
        )
        closure_data = PerturbationClosureData(
            name="psi_equals_phi",
            target="psi_aux",
            expression="phi_aux",
            role="closure",
        )
        source_data = PerturbationSourceData(
            name="monopole_source",
            expression="visibility * density_drive",
            role="monopole",
        )
        observable_data = PerturbationObservableData(
            name="temperature",
            kind="transfer_component",
            projection="line_of_sight_temperature",
        )
        target_data = PerturbationConditionTargetData(
            variable="delta_x",
            wrt="tau",
            order=0,
        )
        condition_data = PerturbationConditionData(
            name="delta_seed",
            target=target_data,
            expression="seed",
        )
        validity_data = PerturbationValidityData(
            regimes=("linear", "synthetic"),
            notes="Synthetic graph.",
        )
        backend_mapping_data = PerturbationBackendMappingData(
            backend="camb",
            native_solver_required=True,
            implemented=True,
        )
        dependency_summary = PerturbationDependencyGraphSummaryData(
            variable_names=("delta_x",),
            derived_names=("density_drive",),
            equation_names=("evolve_delta_x",),
            constraint_names=("poisson_phi",),
            closure_names=("psi_equals_phi",),
            source_names=("monopole_source",),
            observable_names=("temperature",),
            initial_condition_names=("delta_seed",),
            boundary_condition_names=(),
            independent_variables_used=("tau",),
            model_parameters_used=(),
            background_references_used=("H0",),
            derived_dependencies={},
            equation_dependencies={},
            constraint_dependencies={},
            closure_dependencies={},
            source_dependencies={},
            observable_dependencies={},
            initial_condition_dependencies={},
            boundary_condition_dependencies={},
            evaluation_order=("equation:evolve_delta_x",),
        )

        self.assertEqual(variable_data.name, "delta_x")
        self.assertIsInstance(variable_data, PerturbationVariableData)
        self.assertEqual(derived_data.expression, "delta_x + phi_aux")
        self.assertIsInstance(derived_data, PerturbationDerivedData)
        self.assertEqual(equation_data.lhs.order, 1)
        self.assertIsInstance(lhs_data, PerturbationDerivativeLhsData)
        self.assertIsInstance(equation_data, PerturbationEquationData)
        self.assertEqual(constraint_data.target, "phi_aux")
        self.assertIsInstance(constraint_data, PerturbationConstraintData)
        self.assertEqual(closure_data.target, "psi_aux")
        self.assertIsInstance(closure_data, PerturbationClosureData)
        self.assertEqual(source_data.role, "monopole")
        self.assertIsInstance(source_data, PerturbationSourceData)
        self.assertEqual(
            observable_data.projection,
            "line_of_sight_temperature",
        )
        self.assertIsInstance(observable_data, PerturbationObservableData)
        self.assertEqual(condition_data.target.variable, "delta_x")
        self.assertEqual(condition_data.anchor, "start")
        self.assertIsInstance(target_data, PerturbationConditionTargetData)
        self.assertIsInstance(condition_data, PerturbationConditionData)
        self.assertEqual(validity_data.regimes, ("linear", "synthetic"))
        self.assertIsInstance(validity_data, PerturbationValidityData)
        self.assertTrue(backend_mapping_data.implemented)
        self.assertIsInstance(
            backend_mapping_data,
            PerturbationBackendMappingData,
        )
        self.assertEqual(
            dependency_summary.evaluation_order,
            ("equation:evolve_delta_x",),
        )
        self.assertIsInstance(
            dependency_summary,
            PerturbationDependencyGraphSummaryData,
        )


if __name__ == "__main__":
    unittest.main()
