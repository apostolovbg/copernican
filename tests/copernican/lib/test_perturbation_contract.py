"""Tests for the declared perturbation-graph compiler."""

from __future__ import annotations

import unittest

import copernican.lib.perturbation_contract as perturbation_contract_module
from copernican.lib.perturbation_contract import (
    PerturbationBackendMappingData,
    PerturbationClosureData,
    PerturbationCollisionLinearFormData,
    PerturbationCollisionOperatorData,
    PerturbationCollisionTargetSelectorData,
    PerturbationCompiledExpressionData,
    PerturbationConditionData,
    PerturbationConditionTargetData,
    PerturbationConservationRuleData,
    PerturbationConstraintData,
    PerturbationContractData,
    PerturbationDependencyGraphSummaryData,
    PerturbationDerivativeLhsData,
    PerturbationDerivedData,
    PerturbationEquationData,
    PerturbationHierarchyFamilyData,
    PerturbationInitialConditionFamilyData,
    PerturbationInteractionData,
    PerturbationObservableData,
    PerturbationProjectionExtensionData,
    PerturbationProjectionTypingData,
    PerturbationSectorData,
    PerturbationSourceData,
    PerturbationSpeciesData,
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
                "gauge_role": "curvature_potential",
            },
            "psi_aux": {
                "kind": "metric_potential_psi",
                "gauge_role": "newtonian_potential",
            },
        },
        "derived": {
            "density_drive": {
                "expression": "delta_x + phi_aux",
                "description": "Synthetic driving term.",
            },
            "acoustic_k": {
                "expression": "k",
                "description": "Synthetic acoustic wave number.",
            },
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
        "conservation_rules": {},
        "sectors": {
            "scalar": {
                "description": "Synthetic scalar sector.",
                "species": ["photon", "baryon"],
                "hierarchy_families": ["photon_temperature"],
                "supported_gauges": ["conformal_newtonian"],
                "tensor_character": "scalar_like",
            }
        },
        "species": {
            "photon": {
                "sector": "scalar",
                "hierarchy_family": "photon_temperature",
                "collision_operators": ["thomson_drag"],
                "background_reference": "Omega_gamma0",
            },
            "baryon": {
                "sector": "scalar",
                "background_reference": "Omega_b0",
            },
        },
        "hierarchy_families": {
            "photon_temperature": {
                "sector": "scalar",
                "species": ["photon"],
                "closure": "lmax_exponential",
                "default_l_max": 4,
                "multipole_symbol": "theta_gamma_l",
            }
        },
        "collision_operators": {
            "thomson_drag": {
                "sector": "scalar",
                "species": ["photon", "baryon"],
                "expression": "tight_coupling_drag * (theta_x - delta_x)",
            }
        },
        "interactions": {},
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
        "initial_condition_families": {
            "adiabatic_scalar": {
                "sector": "scalar",
                "members": ["delta_seed", "theta_seed"],
            }
        },
        "boundary_conditions": {},
        "projection_typing": {
            "temperature_line_of_sight": {
                "sector": "scalar",
                "kernel": "temperature_mixed_window",
                "source_roles": ["monopole"],
                "observable_kinds": ["transfer_component"],
                "parity": "even",
                "spin": 0.0,
            }
        },
        "projection_extensions": {},
        "accuracy_controls": {
            "ell_refinement": "bounded",
            "k_refinement_levels": [16, 32],
        },
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


def _scalar_metadata_only_contract() -> dict[str, object]:
    """Return one metadata-only scalar contract for hierarchy generation."""

    contract = _base_nonstandard_contract()
    for section_name in (
        "variables",
        "derived",
        "equations",
        "constraints",
        "closures",
        "sources",
        "observables",
        "initial_conditions",
        "boundary_conditions",
    ):
        contract[section_name] = {}
    contract["species"] = {
        "photon": {
            "sector": "scalar",
            "hierarchy_family": "photon_temperature",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_gamma0",
        },
        "baryon": {
            "sector": "scalar",
            "collision_operators": ["thomson_drag"],
            "background_reference": "Omega_b0",
        },
        "cdm": {
            "sector": "scalar",
            "background_reference": "Omega_c0",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "hierarchy_family": "massless_neutrino",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        },
    }
    contract["hierarchy_families"] = {
        "photon_temperature": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 6,
            "multipole_symbol": "theta_gamma_l",
        },
        "photon_polarization_e": {
            "sector": "scalar",
            "species": ["photon"],
            "closure": "free_streaming_scalar",
            "default_l_max": 6,
            "multipole_symbol": "e_gamma_l",
        },
        "massless_neutrino": {
            "sector": "scalar",
            "species": ["massless_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 4,
            "multipole_symbol": "nu_l",
        },
    }
    contract["collision_operators"] = {
        "thomson_drag": {
            "sector": "scalar",
            "species": ["photon", "baryon"],
            "expression": (
                "collision_rate * "
                "((theta_b / acoustic_k) / 3.0 - theta_gamma1)"
            ),
        }
    }
    contract["initial_condition_families"] = {
        "adiabatic_scalar": {
            "sector": "scalar",
            "members": [],
        }
    }
    contract["numerics"].update(
        {
            "photon_hierarchy_l_max": 6,
            "neutrino_hierarchy_l_max": 4,
        }
    )
    contract["validity"] = {
        "regimes": ["linear", "native_scalar_hierarchy"],
    }
    return contract


def _vector_metadata_only_contract() -> dict[str, object]:
    """Return one metadata-only vector contract for hierarchy generation."""

    contract = _base_nonstandard_contract()
    for section_name in (
        "variables",
        "derived",
        "equations",
        "constraints",
        "closures",
        "sources",
        "observables",
        "initial_conditions",
        "boundary_conditions",
    ):
        contract[section_name] = {}
    contract["gauge"] = "conformal_newtonian"
    contract["sectors"] = {
        "vector": {
            "description": "Native vector hierarchy sector.",
            "species": ["photon", "baryon", "cdm", "massless_neutrino"],
            "hierarchy_families": [
                "photon_temperature_vector",
                "photon_polarization_e_vector",
                "photon_polarization_b_vector",
                "massless_neutrino_vector",
            ],
            "supported_gauges": ["conformal_newtonian"],
            "tensor_character": "vector_like",
        }
    }
    contract["species"] = {
        "photon": {
            "sector": "vector",
            "hierarchy_family": "photon_temperature_vector",
            "collision_operators": ["thomson_vector_drag"],
            "background_reference": "Omega_gamma0",
        },
        "baryon": {
            "sector": "vector",
            "collision_operators": ["thomson_vector_drag"],
            "background_reference": "Omega_b0",
        },
        "cdm": {
            "sector": "vector",
            "background_reference": "Omega_c0",
        },
        "massless_neutrino": {
            "sector": "vector",
            "hierarchy_family": "massless_neutrino_vector",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        },
    }
    contract["hierarchy_families"] = {
        "photon_temperature_vector": {
            "sector": "vector",
            "species": ["photon"],
            "closure": "free_streaming_vector",
            "default_l_max": 6,
            "multipole_symbol": "theta_gamma_v_l",
        },
        "photon_polarization_e_vector": {
            "sector": "vector",
            "species": ["photon"],
            "closure": "free_streaming_vector",
            "default_l_max": 6,
            "multipole_symbol": "e_gamma_v_l",
        },
        "photon_polarization_b_vector": {
            "sector": "vector",
            "species": ["photon"],
            "closure": "free_streaming_vector",
            "default_l_max": 6,
            "multipole_symbol": "b_gamma_v_l",
        },
        "massless_neutrino_vector": {
            "sector": "vector",
            "species": ["massless_neutrino"],
            "closure": "free_streaming_vector",
            "default_l_max": 4,
            "multipole_symbol": "nu_v_l",
        },
    }
    contract["collision_operators"] = {}
    contract["conservation_rules"] = {}
    contract["projection_typing"] = {}
    contract["initial_condition_families"] = {
        "regular_vector_mode": {
            "sector": "vector",
            "members": [],
        }
    }
    contract["numerics"].update(
        {
            "photon_hierarchy_l_max": 6,
            "photon_polarization_hierarchy_l_max": 6,
            "neutrino_hierarchy_l_max": 4,
        }
    )
    contract["validity"] = {
        "regimes": ["linear", "native_vector_hierarchy"],
    }
    return contract


def _tensor_metadata_only_contract() -> dict[str, object]:
    """Return one metadata-only tensor contract for hierarchy generation."""

    contract = _base_nonstandard_contract()
    for section_name in (
        "variables",
        "derived",
        "equations",
        "constraints",
        "closures",
        "sources",
        "observables",
        "initial_conditions",
        "boundary_conditions",
    ):
        contract[section_name] = {}
    contract["gauge"] = "conformal_newtonian"
    contract["sectors"] = {
        "tensor": {
            "description": "Native tensor hierarchy sector.",
            "species": ["photon", "massless_neutrino"],
            "hierarchy_families": [
                "photon_temperature_tensor",
                "photon_polarization_e_tensor",
                "photon_polarization_b_tensor",
                "massless_neutrino_tensor",
            ],
            "supported_gauges": ["conformal_newtonian"],
            "tensor_character": "tensor_like",
        }
    }
    contract["species"] = {
        "photon": {
            "sector": "tensor",
            "hierarchy_family": "photon_temperature_tensor",
            "background_reference": "Omega_gamma0",
        },
        "massless_neutrino": {
            "sector": "tensor",
            "hierarchy_family": "massless_neutrino_tensor",
            "background_reference": "Omega_nu0",
            "anisotropic_stress": "supported",
        },
    }
    contract["hierarchy_families"] = {
        "photon_temperature_tensor": {
            "sector": "tensor",
            "species": ["photon"],
            "closure": "free_streaming_tensor",
            "default_l_max": 6,
            "multipole_symbol": "theta_gamma_t_l",
        },
        "photon_polarization_e_tensor": {
            "sector": "tensor",
            "species": ["photon"],
            "closure": "free_streaming_tensor",
            "default_l_max": 6,
            "multipole_symbol": "e_gamma_t_l",
        },
        "photon_polarization_b_tensor": {
            "sector": "tensor",
            "species": ["photon"],
            "closure": "free_streaming_tensor",
            "default_l_max": 6,
            "multipole_symbol": "b_gamma_t_l",
        },
        "massless_neutrino_tensor": {
            "sector": "tensor",
            "species": ["massless_neutrino"],
            "closure": "free_streaming_tensor",
            "default_l_max": 4,
            "multipole_symbol": "nu_t_l",
        },
    }
    contract["collision_operators"] = {}
    contract["conservation_rules"] = {}
    contract["projection_typing"] = {}
    contract["initial_condition_families"] = {
        "tensor_mode": {
            "sector": "tensor",
            "members": [],
        }
    }
    contract["numerics"].update(
        {
            "photon_hierarchy_l_max": 6,
            "photon_polarization_hierarchy_l_max": 6,
            "neutrino_hierarchy_l_max": 4,
        }
    )
    contract["validity"] = {
        "regimes": ["linear", "native_tensor_hierarchy"],
    }
    return contract


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

    def test_scalar_metadata_contract_materializes_runtime_graph(self) -> None:
        """Metadata-only scalar contracts should expand into graph entries."""

        compiled = self._compile(_scalar_metadata_only_contract())

        self.assertFalse(compiled.standard)
        self.assertIn("theta_gamma0", compiled.variables)
        self.assertIn("theta_gamma6", compiled.variables)
        self.assertIn("e_gamma6", compiled.variables)
        self.assertIn("nu_l4", compiled.variables)
        self.assertIn("evolve_theta_gamma6", compiled.equations)
        self.assertIn("evolve_nu_l4", compiled.equations)
        self.assertIn("TT", compiled.observables)
        self.assertIn("TE", compiled.observables)
        self.assertIn("EE", compiled.observables)
        self.assertTrue(
            compiled.manifest_summary["generated_scalar_hierarchy"]
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
            perturbation_contract_module.PerturbationSectorData,
            PerturbationSectorData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationSpeciesData,
            PerturbationSpeciesData,
        )
        self.assertIs(
            perturbation_contract_module.evaluate_compiled_expression,
            evaluate_compiled_expression,
        )

    def test_scalar_hierarchy_omits_undeclared_cdm(self) -> None:
        """Generated scalar graphs should contain only declared species."""

        contract = _scalar_metadata_only_contract()
        contract["species"].pop("cdm")
        compiled = self._compile(contract)

        self.assertTrue(
            compiled.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertNotIn("delta_c", compiled.variables)
        self.assertNotIn("theta_c", compiled.variables)
        self.assertNotIn("evolve_delta_c", compiled.equations)
        self.assertNotIn("evolve_theta_c", compiled.equations)
        self.assertNotIn("observable_delta_c", compiled.derived)
        self.assertNotIn("observable_theta_c", compiled.derived)
        self.assertNotIn(
            "Omega_c0",
            compiled.derived["matter_density_source"].expression,
        )
        self.assertNotIn(
            "Omega_c0",
            compiled.derived["total_momentum_source"].expression,
        )

    def test_scalar_hierarchy_omits_undeclared_massless_neutrinos(
        self,
    ) -> None:
        """Massless-neutrino states should follow the species inventory."""

        contract = _scalar_metadata_only_contract()
        contract["species"].pop("massless_neutrino")
        contract["hierarchy_families"].pop("massless_neutrino")
        compiled = self._compile(contract)

        self.assertNotIn("delta_nu", compiled.variables)
        self.assertNotIn("theta_nu", compiled.variables)
        self.assertNotIn("sigma_nu", compiled.variables)
        self.assertNotIn("evolve_delta_nu", compiled.equations)
        self.assertNotIn("evolve_theta_nu", compiled.equations)
        self.assertNotIn("evolve_sigma_nu", compiled.equations)
        self.assertNotIn("observable_delta_nu", compiled.derived)
        self.assertNotIn("observable_theta_nu", compiled.derived)
        self.assertNotIn(
            "massless_neutrino_fraction",
            compiled.derived["radiation_density_source"].expression,
        )

    def test_scalar_hierarchy_consumes_declared_matter_source_closure(
        self,
    ) -> None:
        """Theory source closures should replace the generic matter term."""

        contract = _scalar_metadata_only_contract()
        contract["species"].pop("cdm")
        contract["sources"] = {
            "relational_matter_density": {
                "expression": "2.0 * Omega_b0 * observable_delta_b",
                "role": "matter_density",
                "description": "Relational matter source closure.",
            },
            "relational_matter_momentum": {
                "expression": "2.0 * Omega_b0 * observable_theta_b",
                "role": "matter_momentum",
                "description": "Relational momentum source closure.",
            },
            "relational_baryon_euler": {
                "expression": (
                    "-Hconf * theta_b + acoustic_k_sq * "
                    "baryon_sound_speed_sq * delta_b + "
                    "baryon_thomson_drag / 2.0 + acoustic_k_sq * Psi"
                ),
                "role": "baryon_euler",
                "description": "Relational baryon Euler closure.",
            },
        }
        compiled = self._compile(contract)

        self.assertIn("relational_matter_density", compiled.sources)
        self.assertIn(
            "2.0 * Omega_b0 * observable_delta_b",
            compiled.derived["matter_density_source"].expression,
        )
        self.assertIn(
            "2.0 * Omega_b0 * observable_theta_b",
            compiled.derived["total_momentum_source"].expression,
        )
        self.assertEqual(
            compiled.equations["evolve_theta_b"].rhs,
            (
                "-Hconf * theta_b + acoustic_k_sq * baryon_sound_speed_sq * "
                "delta_b + baryon_thomson_drag / 2.0 + acoustic_k_sq * Psi"
            ),
        )

    def test_scalar_metadata_contract_tracks_canonical_units_and_roles(
        self,
    ) -> None:
        """Generated scalar metadata should expose units and roles."""

        compiled = self._compile(_scalar_metadata_only_contract())

        self.assertEqual(
            compiled.variables["Phi"].gauge_role,
            "curvature_potential",
        )
        self.assertEqual(
            compiled.variables["Psi"].gauge_role,
            "newtonian_potential",
        )
        self.assertEqual(
            compiled.variables["theta_b"].units,
            "1/Mpc",
        )
        self.assertEqual(
            compiled.variables["theta_gamma2"].units,
            "dimensionless",
        )
        self.assertEqual(
            compiled.derived["acoustic_k"].units,
            "1/Mpc",
        )
        self.assertEqual(
            compiled.derived["einstein_momentum_residual"].units,
            "1/Mpc^3",
        )
        self.assertEqual(
            compiled.sources["temperature_monopole"].units,
            "1/Mpc",
        )
        self.assertEqual(
            compiled.observables["temperature"].units,
            "dimensionless",
        )
        self.assertEqual(compiled.observables["TT"].units, "muK^2")
        self.assertEqual(compiled.observables["TP"].units, "dimensionless")
        self.assertEqual(
            compiled.observables["PP"].units,
            "dimensionless",
        )
        self.assertIn(
            "clpp = [ell(ell+1)]^2 C_ell^{phiphi} / (2*pi).",
            compiled.observables["PP"].notes,
        )
        self.assertEqual(
            compiled.manifest_summary["transfer_component_contracts"][
                "temperature"
            ]["units"],
            "dimensionless",
        )
        self.assertEqual(
            compiled.manifest_summary["angular_power_spectrum_targets"]["PP"][
                "output_role"
            ],
            "potential_power",
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
            contract_data.sectors["scalar"],
            PerturbationSectorData,
        )
        self.assertIsInstance(
            contract_data.species["photon"],
            PerturbationSpeciesData,
        )
        self.assertIsInstance(
            contract_data.hierarchy_families["photon_temperature"],
            PerturbationHierarchyFamilyData,
        )
        self.assertIsInstance(
            contract_data.collision_operators["thomson_drag"],
            PerturbationCollisionOperatorData,
        )
        self.assertIsInstance(
            contract_data.initial_condition_families["adiabatic_scalar"],
            PerturbationInitialConditionFamilyData,
        )
        self.assertIsInstance(
            contract_data.projection_typing["temperature_line_of_sight"],
            PerturbationProjectionTypingData,
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
                "output_role": "temperature_power",
                "units": "muK^2",
            },
        )
        self.assertEqual(
            contract_data.manifest_summary["sector_names"],
            ("scalar",),
        )
        self.assertEqual(
            contract_data.manifest_summary["species_names"],
            ("baryon", "photon"),
        )
        self.assertEqual(
            contract_data.manifest_summary["compilation_ownership"],
            {
                "compiler": (
                    "copernican.lib.model_coder." "compile_native_cmb_runtime"
                ),
                "compiled_upstream": True,
                "hot_path_recompilation_allowed": False,
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

    def test_species_with_unknown_sector_fails(self) -> None:
        """Hierarchy metadata should reject unknown sector references."""

        contract = _base_nonstandard_contract()
        contract["species"]["photon"]["sector"] = "tensor"

        with self.assertRaisesRegex(ValueError, "unknown sector 'tensor'"):
            self._compile(contract)

    def test_initial_condition_family_requires_known_members(self) -> None:
        """Family metadata should reference declared initial conditions."""

        contract = _base_nonstandard_contract()
        contract["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = ["delta_seed", "missing_seed"]

        with self.assertRaisesRegex(
            ValueError,
            "unknown initial conditions: missing_seed",
        ):
            self._compile(contract)

    def test_standard_initial_condition_family_generates_missing_seeds(
        self,
    ) -> None:
        """Standard modes should auto-generate missing start conditions."""

        contract = _base_nonstandard_contract()
        contract["initial_conditions"] = {}
        contract["initial_condition_families"] = {
            "adiabatic_scalar": {
                "sector": "scalar",
                "members": [],
            }
        }

        contract_data = self._compile(contract)

        self.assertIn(
            "adiabatic_scalar_delta_x_tau_0_seed",
            contract_data.initial_conditions,
        )
        self.assertIn(
            "adiabatic_scalar_theta_x_tau_0_seed",
            contract_data.initial_conditions,
        )

    def test_metric_role_initial_conditions_follow_physical_series(
        self,
    ) -> None:
        """Metric-role seeds should use leading-order physical series."""

        newtonian = _base_nonstandard_contract()
        newtonian["constraints"] = {}
        newtonian["closures"] = {}
        newtonian["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = []
        newtonian["equations"]["evolve_phi_aux"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "phi_aux",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "psi_aux",
            "role": "closure",
        }
        newtonian["equations"]["evolve_psi_aux"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "psi_aux",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "phi_aux",
            "role": "closure",
        }
        newtonian_data = self._compile(newtonian)
        self.assertEqual(
            newtonian_data.initial_conditions[
                "adiabatic_scalar_phi_aux_tau_0_seed"
            ].compiled_expression.expression,
            "seed",
        )
        self.assertEqual(
            newtonian_data.initial_conditions[
                "adiabatic_scalar_psi_aux_tau_0_seed"
            ].compiled_expression.expression,
            "seed",
        )

        synchronous = _base_nonstandard_contract()
        synchronous["gauge"] = "synchronous"
        synchronous["sectors"]["scalar"]["supported_gauges"] = [
            "synchronous",
        ]
        synchronous["constraints"] = {}
        synchronous["closures"] = {}
        synchronous["derived"]["acoustic_k"] = {
            "expression": "k",
            "description": "Synthetic acoustic wave number.",
        }
        synchronous["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = []
        synchronous["variables"].pop("phi_aux")
        synchronous["variables"].pop("psi_aux")
        synchronous["variables"]["h_sync_metric"] = {
            "kind": "synchronous_metric_trace",
            "gauge_role": "synchronous_metric_trace",
        }
        synchronous["variables"]["eta_sync_metric"] = {
            "kind": "synchronous_metric_shear",
            "gauge_role": "synchronous_metric_shear",
        }
        synchronous["derived"]["density_drive"] = {
            "expression": "delta_x + h_sync_metric",
            "description": "Synthetic driving term.",
        }
        evolve_delta_x = synchronous["equations"]["evolve_delta_x"]
        evolve_theta_x = synchronous["equations"]["evolve_theta_x"]
        evolve_delta_x["rhs"] = "-theta_x + h_sync_metric"
        evolve_theta_x["rhs"] = "-Hconf * theta_x + k * eta_sync_metric"
        synchronous["equations"]["evolve_h_sync_metric"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "h_sync_metric",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "eta_sync_metric",
            "role": "closure",
        }
        synchronous["equations"]["evolve_eta_sync_metric"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "eta_sync_metric",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "h_sync_metric",
            "role": "closure",
        }
        synchronous_data = self._compile(synchronous)
        self.assertEqual(
            synchronous_data.initial_conditions[
                "adiabatic_scalar_h_sync_metric_tau_0_seed"
            ].compiled_expression.expression,
            "(acoustic_k * eta_initial) * "
            "(acoustic_k * eta_initial) * seed",
        )
        self.assertEqual(
            synchronous_data.initial_conditions[
                "adiabatic_scalar_eta_sync_metric_tau_0_seed"
            ].compiled_expression.expression,
            "2.0 * seed",
        )

    def test_generated_synchronous_route_materializes_gauge_bridge(
        self,
    ) -> None:
        """Generated synchronous metadata should expose stable transforms."""

        contract = _scalar_metadata_only_contract()
        contract["gauge"] = "synchronous"
        compiled = self._compile(contract)

        self.assertIn("gauge_shift_alpha", compiled.variables)
        self.assertIn("Phi", compiled.variables)
        self.assertIn("Psi", compiled.variables)
        self.assertIn("Phi_gi", compiled.variables)
        self.assertIn("evolve_Phi_gi", compiled.equations)
        self.assertIn("evolve_h_sync_metric", compiled.equations)
        self.assertIn("evolve_eta_sync_metric", compiled.equations)
        self.assertIn("evolve_gauge_shift_alpha", compiled.equations)
        self.assertIn("Phi_from_synchronous", compiled.derived)
        self.assertIn("Psi_from_synchronous", compiled.derived)
        self.assertNotIn("evolve_Phi", compiled.equations)
        self.assertEqual(
            compiled.closures["phi_closure"].expression,
            "Phi_gi",
        )
        self.assertEqual(compiled.constraints, {})
        self.assertEqual(
            compiled.closures["psi_closure"].expression,
            "Phi_gi - metric_shear_correction",
        )
        self.assertEqual(
            compiled.derived["gauge_shift_alpha_tau"].expression,
            "Psi - Hconf * gauge_shift_alpha",
        )

    def test_generated_neutrino_velocity_mode_uses_regular_leading_powers(
        self,
    ) -> None:
        """Velocity isocurvature seeds should use regular powers."""

        contract = _scalar_metadata_only_contract()
        contract["initial_condition_families"] = {
            "neutrino_velocity_isocurvature": {
                "sector": "scalar",
                "members": [],
            }
        }
        compiled = self._compile(contract)

        self.assertEqual(
            compiled.initial_conditions["theta_nu_seed"].expression,
            "acoustic_k * seed",
        )
        self.assertEqual(
            compiled.initial_conditions["theta_gamma1_seed"].expression,
            "seed / 3.0",
        )
        self.assertEqual(
            compiled.initial_conditions["sigma_nu_seed"].expression,
            "(acoustic_k * scalar_initial_conformal_time / 6.0) * seed",
        )

    def test_vector_metadata_contract_materializes_runtime_graph(self) -> None:
        """Metadata-only vector contracts should expand into graph entries."""

        compiled = self._compile(_vector_metadata_only_contract())

        self.assertTrue(
            compiled.manifest_summary["generated_vector_hierarchy"]
        )
        self.assertFalse(
            compiled.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertIn("sigma_vector", compiled.variables)
        self.assertIn("theta_gamma_v6", compiled.variables)
        self.assertIn("e_gamma_v6", compiled.variables)
        self.assertIn("b_gamma_v6", compiled.variables)
        self.assertIn("nu_v4", compiled.variables)
        self.assertIn("evolve_sigma_vector", compiled.equations)
        self.assertIn("vector_temperature_source", compiled.sources)
        self.assertIn("TT", compiled.observables)
        self.assertIn("TE", compiled.observables)
        self.assertIn("EE", compiled.observables)
        self.assertIn("BB", compiled.observables)
        self.assertEqual(
            compiled.observables["temperature"].kernel,
            "spherical_bessel_window",
        )
        self.assertEqual(
            compiled.observables["polarization_b"].parity,
            "odd",
        )

    def test_vector_materializer_generates_physical_source_terms(self) -> None:
        """Generated vector sources should expose LOS and drag surfaces."""

        compiled = self._compile(_vector_metadata_only_contract())

        self.assertEqual(
            compiled.collision_operators["thomson_vector_drag"].expression,
            "collision_rate * ((4.0 / 3.0) * v_b_vector - q_gamma_vector)",
        )
        self.assertIn(
            "vector_visibility_polarization_moment_tau",
            compiled.sources["vector_temperature_source"].expression,
        )
        self.assertIn(
            "vector_metric_shear_rhs",
            compiled.sources["vector_temperature_source"].expression,
        )
        self.assertIn(
            "vector_photon_baryon_loading * thomson_vector_drag",
            compiled.conservation_rules[
                "thomson_vector_drag_balance"
            ].expression,
        )

    def test_vector_collision_terms_vanish_on_tight_coupling_surface(
        self,
    ) -> None:
        """Vector Thomson terms should vanish on the tight-coupling surface."""

        compiled = self._compile(_vector_metadata_only_contract())
        drag_context = {
            "collision_rate": 25.0,
            "q_gamma_vector": 2.0,
            "v_b_vector": 1.5,
        }
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.collision_operators[
                        "thomson_vector_drag"
                    ].compiled_expression,
                    drag_context,
                )
            ),
            0.0,
        )
        quadrupole_context = {
            "collision_rate": 25.0,
            "pi_gamma_vector": 0.4,
            "e_gamma_v2": 0.4,
            "vector_polarization_moment": 0.4,
        }
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.collision_operators[
                        "vector_quadrupole_collision"
                    ].compiled_expression,
                    quadrupole_context,
                )
            ),
            0.0,
        )
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.collision_operators[
                        "vector_e_quadrupole_collision"
                    ].compiled_expression,
                    quadrupole_context,
                )
            ),
            0.0,
        )

    def test_vector_free_streaming_hierarchy_uses_camb_coefficients(
        self,
    ) -> None:
        """Vector hierarchy recurrences should follow the CAMB coefficients."""

        compiled = self._compile(_vector_metadata_only_contract())
        context = {
            "acoustic_k": 0.5,
            "collision_rate": 0.0,
            "pi_gamma_vector": 1.2,
            "theta_gamma_v3": -0.1,
            "theta_gamma_v4": -0.4,
            "theta_gamma_v5": 0.2,
            "theta_gamma_v6": -0.3,
            "e_gamma_v2": 0.7,
            "e_gamma_v3": 0.1,
            "e_gamma_v4": -0.2,
            "e_gamma_v5": 0.4,
            "e_gamma_v6": -0.1,
            "b_gamma_v3": 0.6,
            "b_gamma_v6": 0.8,
            "vector_eta_safe": 10.0,
        }
        photon_rhs = float(
            evaluate_compiled_expression(
                compiled.equations["evolve_theta_gamma_v3"].compiled_rhs,
                context,
            )
        )
        self.assertAlmostEqual(
            photon_rhs,
            (3.0 / 7.0) * 0.5 * 1.2 - (15.0 / 28.0) * 0.5 * (-0.4),
        )
        e_rhs = float(
            evaluate_compiled_expression(
                compiled.equations["evolve_e_gamma_v3"].compiled_rhs,
                context,
            )
        )
        self.assertAlmostEqual(
            e_rhs,
            (3.0 / 7.0) * 0.5 * 0.7
            - (45.0 / 112.0) * 0.5 * (-0.2)
            + (1.0 / 6.0) * 0.5 * 0.6,
        )
        terminal_temperature_rhs = float(
            evaluate_compiled_expression(
                compiled.equations["evolve_theta_gamma_v6"].compiled_rhs,
                context,
            )
        )
        self.assertAlmostEqual(
            terminal_temperature_rhs,
            (6.0 / 5.0) * 0.5 * 0.2 - (8.0 / 10.0) * (-0.3),
        )
        self.assertEqual(
            compiled.equations["evolve_e_gamma_v6"].rhs,
            "0.0",
        )
        self.assertEqual(
            compiled.equations["evolve_b_gamma_v6"].rhs,
            "0.0",
        )

    def test_tensor_metadata_contract_materializes_runtime_graph(
        self,
    ) -> None:
        """Metadata-only tensor contracts should expand into graph entries."""

        compiled = self._compile(_tensor_metadata_only_contract())

        self.assertTrue(
            compiled.manifest_summary["generated_tensor_hierarchy"]
        )
        self.assertFalse(
            compiled.manifest_summary["generated_scalar_hierarchy"]
        )
        self.assertFalse(
            compiled.manifest_summary["generated_vector_hierarchy"]
        )
        self.assertIn("h_tensor", compiled.variables)
        self.assertIn("h_tensor_tau", compiled.variables)
        self.assertIn("theta_gamma_t6", compiled.variables)
        self.assertIn("e_gamma_t6", compiled.variables)
        self.assertIn("b_gamma_t6", compiled.variables)
        self.assertIn("nu_t4", compiled.variables)
        self.assertIn("evolve_h_tensor_tau", compiled.equations)
        self.assertIn("tensor_temperature_source", compiled.sources)
        self.assertIn("TT", compiled.observables)
        self.assertIn("TE", compiled.observables)
        self.assertIn("EE", compiled.observables)
        self.assertIn("BB", compiled.observables)
        self.assertEqual(
            compiled.observables["polarization_b"].projection,
            "spin2_b_mode",
        )
        self.assertEqual(
            compiled.observables["polarization_b"].parity,
            "odd",
        )

    def test_tensor_materializer_generates_physical_source_terms(
        self,
    ) -> None:
        """Generated tensor sources should expose metric and stress terms."""

        compiled = self._compile(_tensor_metadata_only_contract())

        self.assertEqual(
            compiled.equations["evolve_h_tensor_tau"].rhs,
            "tensor_metric_wave_rhs",
        )
        self.assertIn(
            "pi_nu_tensor",
            compiled.derived["tensor_total_shear_source"].expression,
        )
        self.assertIn(
            "h_tensor_tau",
            compiled.sources["tensor_temperature_source"].expression,
        )
        self.assertEqual(
            compiled.sources["tensor_temperature_source"].expression,
            "-exp(-tau) * h_tensor_tau + "
            "(15.0 / 8.0) * visibility * tensor_polarization_moment",
        )
        self.assertIn(
            "tensor_polarization_moment",
            compiled.sources["tensor_polarization_e_source"].expression,
        )
        self.assertEqual(
            compiled.derived["tensor_polarization_moment"].expression,
            "0.1 * pi_gamma_tensor + 0.6 * e_gamma_t2",
        )
        self.assertEqual(
            compiled.sources["tensor_polarization_e_source"].expression,
            "(15.0 / 2.0) * sqrt(3.0 / 8.0) * visibility * "
            "tensor_polarization_moment",
        )
        self.assertEqual(
            compiled.sources["tensor_polarization_b_source"].expression,
            "(15.0 / 2.0) * sqrt(3.0 / 8.0) * visibility * "
            "tensor_polarization_moment + "
            "0.0 * visibility * b_gamma_t2",
        )
        tensor_collision = compiled.collision_operators[
            "tensor_thomson_collision"
        ]
        self.assertEqual(
            tensor_collision.integration_strategy,
            "exact",
        )
        self.assertEqual(
            tensor_collision.rate_expression,
            "collision_rate",
        )
        self.assertEqual(
            tensor_collision.exact_form.matrix,
            (
                ("-0.9", "0.6"),
                ("0.1", "-0.4"),
            ),
        )
        self.assertEqual(
            tensor_collision.exact_form.targets,
            (
                PerturbationCollisionTargetSelectorData(
                    variable="pi_gamma_tensor",
                ),
                PerturbationCollisionTargetSelectorData(
                    variable="e_gamma_t2",
                ),
            ),
        )
        self.assertIn(
            PerturbationCollisionTargetSelectorData(
                variable="b_gamma_t2",
            ),
            tensor_collision.exact_form.damping_targets,
        )

    def test_tensor_free_streaming_hierarchy_uses_spin2_coefficients(
        self,
    ) -> None:
        """Tensor hierarchy recurrences should use the spin-2 coefficients."""

        compiled = self._compile(_tensor_metadata_only_contract())
        context = {
            "acoustic_k": 0.5,
            "collision_rate": 0.0,
            "pi_gamma_tensor": 0.25,
            "theta_gamma_t3": -0.2,
            "theta_gamma_t4": 0.4,
            "e_gamma_t2": 0.3,
            "e_gamma_t3": 0.1,
            "e_gamma_t4": -0.2,
            "b_gamma_t3": 0.6,
            "tensor_eta_safe": 12.0,
        }
        photon_rhs = float(
            evaluate_compiled_expression(
                compiled.equations["evolve_theta_gamma_t3"].compiled_rhs,
                context,
            )
        )
        self.assertAlmostEqual(
            photon_rhs,
            (3.0 / 7.0) * 0.5 * 0.25 - (3.0 / 7.0) * 0.5 * 0.4,
        )
        e_rhs = float(
            evaluate_compiled_expression(
                compiled.equations["evolve_e_gamma_t3"].compiled_rhs,
                context,
            )
        )
        self.assertAlmostEqual(
            e_rhs,
            (3.0 / 7.0) * 0.5 * 0.3
            - (9.0 / 28.0) * 0.5 * (-0.2)
            + (1.0 / 3.0) * 0.5 * 0.6,
        )

    def test_vector_and_tensor_sector_metadata_are_inferred(self) -> None:
        """Vector and tensor source ancestries should classify cleanly."""

        vector_contract = _base_nonstandard_contract()
        vector_contract["sectors"]["vector"] = {
            "description": "Synthetic vector sector.",
            "tensor_character": "vector_like",
        }
        vector_contract["variables"]["vector_signal"] = {
            "kind": "custom_vector_mode",
            "spin": 1.0,
            "parity": "even",
            "tensor_character": "vector_like",
        }
        vector_contract["equations"]["evolve_vector_signal"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "0.15 * k * theta_x - 0.25 * Hconf * vector_signal",
            "role": "vector_coupling",
        }
        vector_contract["sources"]["vector_source"] = {
            "expression": "visibility * vector_signal",
            "role": "signal",
        }
        vector_contract["observables"]["vector_signal"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_signal",
            "source_terms": {"signal": "vector_source"},
        }
        vector_contract["observables"]["VV"] = {
            "kind": "angular_power_spectrum",
            "primary": "vector_signal",
            "secondary": "vector_signal",
        }
        vector_contract["initial_conditions"]["vector_signal_seed"] = {
            "target": {
                "variable": "vector_signal",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 90.0",
        }
        vector_data = self._compile(vector_contract)
        self.assertEqual(vector_data.observables["VV"].sector, "vector")
        self.assertEqual(
            vector_data.observables["VV"].tensor_character,
            "vector_like",
        )
        self.assertIn("vector", vector_data.manifest_summary["sector_names"])

        tensor_contract = _base_nonstandard_contract()
        tensor_contract["sectors"]["tensor"] = {
            "description": "Synthetic tensor sector.",
            "tensor_character": "tensor_like",
        }
        tensor_contract["variables"]["tensor_b"] = {
            "kind": "custom_tensor_polarization_source",
            "rank": 2,
            "spin": 2.0,
            "parity": "odd",
            "tensor_character": "tensor_like",
        }
        tensor_contract["equations"]["evolve_tensor_b"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "0.2 * k * density_drive - 0.4 * Hconf * tensor_b",
            "role": "odd_parity_polarization",
        }
        tensor_contract["sources"]["polarization_b_source"] = {
            "expression": "visibility * tensor_b",
            "role": "polarization_b",
        }
        tensor_contract["observables"]["polarization_b"] = {
            "kind": "transfer_component",
            "projection": "spin2_b_mode",
            "source_terms": {"polarization_b": "polarization_b_source"},
        }
        tensor_contract["observables"]["BB"] = {
            "kind": "angular_power_spectrum",
            "primary": "polarization_b",
            "secondary": "polarization_b",
        }
        tensor_contract["initial_conditions"]["tensor_b_seed"] = {
            "target": {
                "variable": "tensor_b",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "(k * eta_initial) * seed / 120.0",
        }
        tensor_data = self._compile(tensor_contract)
        self.assertEqual(tensor_data.observables["BB"].sector, "tensor")
        self.assertEqual(
            tensor_data.observables["BB"].tensor_character,
            "tensor_like",
        )
        self.assertIn("tensor", tensor_data.manifest_summary["sector_names"])

    def test_synchronous_gauge_rejects_newtonian_metric_roles(self) -> None:
        """Gauge-role mixes should fail before runtime."""

        contract = _base_nonstandard_contract()
        contract["gauge"] = "synchronous"
        contract["sectors"]["scalar"]["supported_gauges"] = ["synchronous"]

        with self.assertRaisesRegex(
            ValueError,
            "conflict with gauge 'synchronous'",
        ):
            self._compile(contract)

    def test_momentum_grid_family_requires_numerics_entry(self) -> None:
        """Momentum-grid families should bind to declared numerics entries."""

        contract = _scalar_metadata_only_contract()
        contract["species"]["massive_neutrino"] = {
            "sector": "scalar",
            "hierarchy_family": "massive_neutrino",
            "background_reference": "Omega_nu0",
        }
        contract["hierarchy_families"]["massive_neutrino"] = {
            "sector": "scalar",
            "species": ["massive_neutrino"],
            "closure": "free_streaming_scalar",
            "default_l_max": 4,
            "multipole_symbol": "nu_massive_l",
            "momentum_grid": "massive_neutrino_default",
        }

        with self.assertRaisesRegex(
            ValueError,
            "momentum_grid references unknown",
        ):
            self._compile(contract)

    def test_scalar_hierarchy_uses_time_dependent_einstein_sources(
        self,
    ) -> None:
        """Generated scalar routes should use physical Einstein sources."""

        compiled = self._compile(_scalar_metadata_only_contract())

        self.assertIn(
            "0.6 * acoustic_k * theta_gamma3",
            compiled.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertIn(
            "0.3333333333333333 * acoustic_k * e_gamma3",
            compiled.equations["evolve_e_gamma2"].rhs,
        )
        self.assertEqual(
            compiled.equations["evolve_theta_gamma6"].rhs,
            "1 * acoustic_k * theta_gamma5 - acoustic_k * 7 * theta_gamma6 / "
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 7 * 7)",
        )
        self.assertEqual(
            compiled.equations["evolve_e_gamma6"].rhs,
            "1.5 * acoustic_k * e_gamma5 - "
            "acoustic_k * 9 * e_gamma6 / "
            "sqrt((acoustic_k * eta) * (acoustic_k * eta) + 9 * 9)",
        )
        self.assertIn("matter_density_source", compiled.derived)
        self.assertIn("radiation_density_source", compiled.derived)
        self.assertIn(
            "/ a",
            compiled.derived["matter_density_source"].expression,
        )
        self.assertIn(
            "/ (a * a)",
            compiled.derived["radiation_density_source"].expression,
        )
        self.assertIn(
            "photon_velocity_divergence",
            compiled.derived["total_momentum_source"].expression,
        )
        self.assertIn(
            "Omega_gamma0 * observable_theta_gamma2",
            compiled.derived["total_shear_source"].expression,
        )
        self.assertIn("evolve_Phi", compiled.equations)
        self.assertEqual(
            compiled.equations["evolve_Phi"].rhs,
            "Phi_tau",
        )
        self.assertEqual(
            compiled.initial_conditions["Phi_seed"].expression,
            "(scalar_potential_seed) + metric_shear_correction",
        )
        self.assertEqual(
            compiled.derived["scalar_potential_seed"].expression,
            "(10.0 / (15.0 + 4.0 * scalar_neutrino_fraction)) * seed",
        )
        self.assertEqual(
            compiled.derived["scalar_lapse_seed"].expression,
            "scalar_potential_seed",
        )
        self.assertEqual(
            compiled.initial_conditions["theta_gamma2_seed"].expression,
            "(8.0 / 15.0) * acoustic_k * theta_gamma1 / " "collision_rate",
        )
        self.assertEqual(
            compiled.initial_conditions["e_gamma2_seed"].expression,
            "theta_gamma2 / 4.0",
        )
        self.assertEqual(
            compiled.initial_conditions["e_gamma3_seed"].expression,
            "(3.0 / 28.0) * acoustic_k * theta_gamma2 / collision_rate",
        )
        self.assertEqual(
            compiled.derived["metric_constraint_scale"].expression,
            "acoustic_k_sq",
        )
        self.assertEqual(
            compiled.closures["psi_closure"].expression,
            "Phi - metric_shear_correction",
        )
        self.assertIn(
            "metric_momentum_source_drive",
            compiled.derived["Phi_tau"].expression,
        )
        self.assertIsNone(compiled.derived["Psi_tau"].expression)
        self.assertEqual(compiled.derived["Psi_tau"].variable, "Psi")
        self.assertEqual(compiled.derived["Psi_tau"].wrt, "tau")
        self.assertEqual(compiled.derived["Psi_tau"].order, 1)
        self.assertIn("einstein_energy_residual", compiled.derived)
        self.assertIn("einstein_momentum_residual", compiled.derived)
        self.assertIn("einstein_shear_residual", compiled.derived)
        context = {
            "a": 0.5,
            "Omega_b0": 0.05,
            "Omega_c0": 0.25,
            "Omega_gamma0": 1.0e-4,
            "Omega_nu0": 5.0e-5,
            "delta_b": 1.2,
            "delta_c": 0.8,
            "theta_gamma0": 0.4,
            "delta_nu": 1.1,
            "theta_b": 0.03,
            "theta_c": 0.01,
            "theta_gamma1": 0.02,
            "theta_nu": 0.04,
            "sigma_nu": 0.05,
            "acoustic_k": 0.1,
            "massless_neutrino_fraction": 5.0e-5,
        }
        context["observable_delta_b"] = context["delta_b"]
        context["observable_delta_c"] = context["delta_c"]
        context["observable_theta_gamma0"] = context["theta_gamma0"]
        context["observable_theta_nu"] = context["theta_nu"]
        context["observable_theta_b"] = context["theta_b"]
        context["observable_theta_c"] = context["theta_c"]
        context["observable_delta_nu"] = context["delta_nu"]
        context["observable_theta_gamma1"] = context["theta_gamma1"]
        context["photon_velocity_divergence"] = (
            3.0 * context["acoustic_k"] * context["theta_gamma1"]
        )
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.derived[
                        "matter_density_source"
                    ].compiled_expression,
                    context,
                )
            ),
            (0.25 * 0.8 + 0.05 * 1.2) / 0.5,
        )
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.derived[
                        "radiation_density_source"
                    ].compiled_expression,
                    context,
                )
            ),
            (4.0 * 1.0e-4 * 0.4 + 5.0e-5 * 1.1) / (0.5 * 0.5),
        )
        self.assertAlmostEqual(
            float(
                evaluate_compiled_expression(
                    compiled.derived[
                        "total_momentum_source"
                    ].compiled_expression,
                    context,
                )
            ),
            (
                (0.05 * 0.03 + 0.25 * 0.01) / 0.5
                + (
                    (4.0 / 3.0)
                    * 1.0e-4
                    * context["photon_velocity_divergence"]
                    + (4.0 / 3.0) * 5.0e-5 * 0.04
                )
                / (0.5 * 0.5)
            ),
        )

    def test_scalar_hierarchy_materializes_collision_operators(
        self,
    ) -> None:
        """Generated scalar routes should synthesize collision operators."""

        contract = _scalar_metadata_only_contract()
        contract.pop("collision_operators", None)
        compiled = self._compile(contract)

        self.assertIn("thomson_drag", compiled.collision_operators)
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].integration_strategy,
            "exact",
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].rate_expression,
            "collision_rate",
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].activation_strategy,
            "always",
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].counterpart,
            "baryon_thomson_drag",
        )
        self.assertIsInstance(
            compiled.collision_operators["thomson_drag"].exact_form,
            PerturbationCollisionLinearFormData,
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"]
            .exact_form.targets[0]
            .kind,
            "photon_temperature_dipole",
        )
        self.assertEqual(
            compiled.collision_operators[
                "thomson_drag"
            ].exact_form.damping_coefficient,
            "-1.0",
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].exact_form.matrix,
            (
                ("-1.0", "1.0 / (3.0 * acoustic_k)", "0.0", "0.0"),
                (
                    "3.0 * acoustic_k * photon_baryon_momentum_ratio",
                    "-photon_baryon_momentum_ratio",
                    "0.0",
                    "0.0",
                ),
                ("0.0", "0.0", "-0.9", "0.6"),
                ("0.0", "0.0", "0.1", "-0.4"),
            ),
        )
        self.assertIn("thomson_drag_balance", compiled.conservation_rules)
        self.assertEqual(
            compiled.conservation_rules["thomson_drag_balance"].expression,
            "3.0 * acoustic_k * photon_baryon_momentum_ratio * "
            "thomson_drag + "
            "baryon_thomson_drag",
        )
        self.assertEqual(
            compiled.derived["photon_baryon_momentum_ratio"].expression,
            "(4.0 * Omega_gamma0) / (3.0 * Omega_b0 * a)",
        )

    def test_collision_operator_linear_forms_compile(self) -> None:
        """Collision operators should compile exact and implicit metadata."""

        contract = _base_nonstandard_contract()
        contract["collision_operators"] = {
            "thomson_drag": {
                "sector": "scalar",
                "species": ["photon", "baryon"],
                "expression": "collision_rate * (theta_x - delta_x)",
                "integration_strategy": "exact",
                "activation_strategy": "tight_coupling",
                "rate_expression": "collision_rate",
                "exact_form": {
                    "targets": [
                        {"variable": "delta_x"},
                        {"variable": "theta_x"},
                    ],
                    "matrix": [
                        ["-1.0", "0.5"],
                        ["0.25", "-0.75"],
                    ],
                    "activation_strategy": "tight_coupling",
                },
            },
            "drag_feedback": {
                "sector": "scalar",
                "species": ["baryon"],
                "expression": "-0.25 * collision_rate * theta_x",
                "integration_strategy": "implicit",
                "rate_expression": "collision_rate",
                "linear_block": {
                    "targets": [{"variable": "theta_x"}],
                    "matrix": [["-0.25"]],
                },
            },
        }

        compiled = self._compile(contract)
        exact_drag = compiled.collision_operators["thomson_drag"]
        implicit_drag = compiled.collision_operators["drag_feedback"]

        self.assertEqual(exact_drag.integration_strategy, "exact")
        self.assertEqual(exact_drag.activation_strategy, "tight_coupling")
        self.assertEqual(exact_drag.rate_expression, "collision_rate")
        self.assertIsInstance(
            exact_drag.compiled_rate_expression,
            PerturbationCompiledExpressionData,
        )
        self.assertIsInstance(
            exact_drag.exact_form,
            PerturbationCollisionLinearFormData,
        )
        self.assertIsInstance(
            exact_drag.exact_form.targets[0],
            PerturbationCollisionTargetSelectorData,
        )
        self.assertEqual(
            exact_drag.exact_form.targets[0].variable,
            "delta_x",
        )
        self.assertEqual(
            exact_drag.exact_form.compiled_matrix[0][0].expression,
            "-1.0",
        )
        self.assertEqual(implicit_drag.integration_strategy, "implicit")
        self.assertEqual(implicit_drag.rate_expression, "collision_rate")
        self.assertIsInstance(
            implicit_drag.linear_block,
            PerturbationCollisionLinearFormData,
        )
        self.assertEqual(
            implicit_drag.linear_block.targets[0].variable,
            "theta_x",
        )
        self.assertEqual(
            implicit_drag.linear_block.compiled_matrix[0][0].expression,
            "-0.25",
        )

    def test_scalar_hierarchy_uses_physical_collision_block(self) -> None:
        """Generated scalar photon collisions should use the exact block."""

        contract = _scalar_metadata_only_contract()
        compiled = self._compile(contract)

        self.assertNotIn(
            "collision_rate",
            compiled.equations["evolve_theta_gamma2"].rhs,
        )
        self.assertEqual(
            compiled.equations["evolve_e_gamma0"].rhs,
            "0.0",
        )
        self.assertEqual(
            compiled.equations["evolve_e_gamma1"].rhs,
            "0.0",
        )
        self.assertNotIn(
            "collision_rate",
            compiled.equations["evolve_e_gamma2"].rhs,
        )
        self.assertNotIn(
            "e_gamma1",
            compiled.equations["evolve_e_gamma2"].rhs,
        )
        self.assertIn(
            "0.3333333333333333 * acoustic_k * e_gamma3",
            compiled.equations["evolve_e_gamma2"].rhs,
        )
        self.assertIn(
            "- 0.4285714285714285 * acoustic_k * e_gamma4",
            compiled.equations["evolve_e_gamma3"].rhs,
        )
        self.assertIn(
            "1.5 * acoustic_k * e_gamma5",
            compiled.equations["evolve_e_gamma6"].rhs,
        )
        self.assertIn(
            "- acoustic_k * 9 * e_gamma6 / sqrt((acoustic_k * eta) * "
            "(acoustic_k * eta) + 9 * 9)",
            compiled.equations["evolve_e_gamma6"].rhs,
        )
        self.assertIn(
            "baryon_thomson_drag",
            compiled.equations["evolve_theta_b"].rhs,
        )
        self.assertIn(
            "baryon_sound_speed_sq * delta_b",
            compiled.equations["evolve_theta_b"].rhs,
        )
        self.assertEqual(
            compiled.collision_operators["thomson_drag"].expression,
            "collision_rate * ((theta_b / acoustic_k) / 3.0 - theta_gamma1)",
        )
        self.assertEqual(
            compiled.derived["polarization_moment"].expression,
            "0.1 * theta_gamma2 + 0.6 * e_gamma2",
        )
        self.assertEqual(
            compiled.derived[
                "visibility_polarization_moment_tau_tau"
            ].variable,
            "visibility_polarization_moment",
        )
        self.assertEqual(
            compiled.derived["visibility_polarization_moment_tau_tau"].order,
            2,
        )
        self.assertEqual(
            compiled.sources["temperature_quadrupole"].expression,
            "(5.0 / 2.0) * visibility * polarization_moment",
        )
        self.assertEqual(
            compiled.sources["temperature_quadrupole_derivative"].expression,
            "(15.0 / 2.0) * visibility * polarization_moment",
        )
        self.assertEqual(
            compiled.sources["polarization_source"].expression,
            "(15.0 / 2.0) * visibility * polarization_moment",
        )
        self.assertEqual(
            compiled.sources["temperature_doppler"].expression,
            "visibility * observable_theta_b / acoustic_k",
        )
        self.assertEqual(
            compiled.sources["temperature_isw"].expression,
            "exp(-tau) * (Phi_tau + Psi_tau)",
        )
        self.assertEqual(
            compiled.sources["lensing_potential"].expression,
            "Phi + Psi",
        )
        self.assertNotIn(
            "tight_coupling_drag",
            compiled.equations["evolve_theta_gamma3"].rhs,
        )
        self.assertNotIn(
            "tight_coupling_drag",
            compiled.equations["evolve_e_gamma3"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            compiled.equations["evolve_theta_gamma3"].rhs,
        )
        self.assertNotIn(
            "collision_rate",
            compiled.equations["evolve_e_gamma3"].rhs,
        )

    def test_extended_runtime_physical_scalars_compile(self) -> None:
        """Native graphs may reference the documented physical scalars."""

        contract = _base_nonstandard_contract()
        contract["derived"]["density_drive"]["expression"] = (
            "delta_x + phi_aux + "
            "(rho_b0_kg_m3 / rho_crit0_kg_m3) + "
            "(1.0e-9 * H0_km_s_Mpc) + "
            "(1.0e-9 * Tcmb_K) + "
            "(1.0e-24 * n_H0_m3)"
        )

        contract_data = self._compile(contract)

        self.assertIn(
            "rho_b0_kg_m3",
            contract_data.dependency_graph_summary.background_references_used,
        )
        self.assertIn(
            "H0_km_s_Mpc",
            contract_data.dependency_graph_summary.background_references_used,
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
        contract["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = ["delta_seed"]

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

    def test_mode_family_selectors_are_rejected(self) -> None:
        """Declared graphs should not accept theory-family selectors."""

        contract = _base_nonstandard_contract()
        contract["mode_families"] = {"scalar": ["delta_x", "theta_x"]}

        with self.assertRaisesRegex(
            ValueError,
            "Unknown perturbation contract key\\(s\\): mode_families",
        ):
            self._compile(contract)

    def test_hidden_backend_selectors_are_rejected(self) -> None:
        """Non-standard backend mappings should stay selector-free."""

        contract = _base_nonstandard_contract()
        contract["backend_mapping"]["camb"]["theory_selector"] = "lcdm_like"

        with self.assertRaisesRegex(
            ValueError,
            "Non-standard perturbation mappings may only declare",
        ):
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

    def test_transfer_component_metadata_tracks_lensing_cross_roles(
        self,
    ) -> None:
        """Compiled observables should expose output roles for scaling."""

        contract = _base_nonstandard_contract()
        contract["sources"]["lensing_source"] = {
            "expression": "phi_aux + psi_aux",
            "role": "potential",
        }
        contract["observables"]["lensing"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_lensing_potential",
            "source_terms": {"potential": "lensing_source"},
        }
        contract["observables"]["TP"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "lensing",
        }

        contract_data = self._compile(contract)

        self.assertEqual(
            contract_data.observables["temperature"].output_role,
            "temperature",
        )
        self.assertEqual(
            contract_data.observables["lensing"].output_role,
            "potential",
        )
        self.assertEqual(
            contract_data.observables["TP"].output_role,
            "temperature_potential_cross",
        )

    def test_mixed_sector_cross_spectrum_fails(self) -> None:
        """Angular spectra should reject mixed scalar and vector targets."""

        contract = _base_nonstandard_contract()
        contract["variables"]["vector_mode"] = {
            "kind": "custom_vector_mode",
            "spin": 1.0,
            "tensor_character": "vector_like",
        }
        contract["equations"]["evolve_vector_mode"] = {
            "lhs": {
                "kind": "derivative",
                "variable": "vector_mode",
                "wrt": "tau",
                "order": 1,
            },
            "rhs": "-0.25 * vector_mode + theta_x",
            "role": "vector_coupling",
        }
        contract["initial_conditions"]["vector_seed"] = {
            "target": {
                "variable": "vector_mode",
                "wrt": "tau",
                "order": 0,
            },
            "expression": "0.01 * seed",
        }
        contract["sources"]["vector_source"] = {
            "expression": "visibility * vector_mode",
            "role": "signal",
        }
        contract["observables"]["vector_signal"] = {
            "kind": "transfer_component",
            "projection": "line_of_sight_signal",
            "source_terms": {"signal": "vector_source"},
        }
        contract["observables"]["TV"] = {
            "kind": "angular_power_spectrum",
            "primary": "temperature",
            "secondary": "vector_signal",
        }

        with self.assertRaisesRegex(ValueError, "incompatible sectors"):
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
        contract["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = ["delta_seed"]

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
        contract["initial_condition_families"]["adiabatic_scalar"][
            "members"
        ] = ["delta_seed"]

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

    def test_extensions_and_conservation_rules_compile(self) -> None:
        """Native extension rules should compile into typed metadata."""

        contract = _base_nonstandard_contract()
        contract["interactions"]["photon_baryon_slip"] = {
            "sector": "scalar",
            "species": ["photon", "baryon"],
            "expression": "thomson_drag + 0.1 * density_drive",
        }
        theta_equation = contract["equations"]["evolve_theta_x"]
        theta_equation["rhs"] += " + photon_baryon_slip"
        contract["conservation_rules"]["metric_balance"] = {
            "expression": "psi_aux - phi_aux",
            "tolerance": 1.0e-9,
        }
        contract["projection_extensions"]["signal_derivative_extension"] = {
            "base_projection": "line_of_sight_signal_derivative",
            "kernel": "spherical_bessel_derivative_window",
            "required_roles": ["signal"],
            "allowed_roles": ["signal"],
        }
        contract["sources"]["signal_source"] = {
            "expression": "visibility * theta_x",
            "role": "signal",
        }
        contract["observables"]["signal_derivative"] = {
            "kind": "transfer_component",
            "projection": "signal_derivative_extension",
            "source_terms": {"signal": "signal_source"},
        }

        contract_data = self._compile(contract)

        self.assertIsInstance(
            contract_data.interactions["photon_baryon_slip"],
            PerturbationInteractionData,
        )
        self.assertIsInstance(
            contract_data.conservation_rules["metric_balance"],
            PerturbationConservationRuleData,
        )
        self.assertIsInstance(
            contract_data.projection_extensions["signal_derivative_extension"],
            PerturbationProjectionExtensionData,
        )
        self.assertEqual(
            contract_data.observables["signal_derivative"].projection,
            "line_of_sight_signal_derivative",
        )
        self.assertEqual(
            contract_data.manifest_summary["interaction_names"],
            ("photon_baryon_slip",),
        )
        self.assertEqual(
            contract_data.manifest_summary["projection_extension_names"],
            ("signal_derivative_extension",),
        )
        self.assertEqual(
            contract_data.manifest_summary["transfer_component_contracts"][
                "signal_derivative"
            ]["declared_projection"],
            "signal_derivative_extension",
        )

    def test_conservation_rule_requires_positive_tolerance(self) -> None:
        """Conservation rules should fail when tolerance is non-positive."""

        contract = _base_nonstandard_contract()
        contract["conservation_rules"]["metric_balance"] = {
            "expression": "psi_aux - phi_aux",
            "tolerance": 0.0,
        }

        with self.assertRaisesRegex(ValueError, "must be a positive float"):
            self._compile(contract)

    def test_projection_extension_cannot_shadow_builtin_projection(
        self,
    ) -> None:
        """Projection extensions should stay distinct from built-ins."""

        contract = _base_nonstandard_contract()
        contract["projection_extensions"]["line_of_sight_temperature"] = {
            "base_projection": "line_of_sight_signal",
        }

        with self.assertRaisesRegex(ValueError, "collides with a built-in"):
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
        sector_data = PerturbationSectorData(name="scalar")
        species_data = PerturbationSpeciesData(
            name="photon",
            sector="scalar",
        )
        hierarchy_family_data = PerturbationHierarchyFamilyData(
            name="photon_temperature",
            sector="scalar",
        )
        collision_operator_data = PerturbationCollisionOperatorData(
            name="thomson_drag",
            sector="scalar",
        )
        interaction_data = PerturbationInteractionData(
            name="photon_baryon_slip",
            sector="scalar",
        )
        conservation_rule_data = PerturbationConservationRuleData(
            name="density_balance",
            expression="delta_x - delta_x",
            tolerance=1.0e-9,
        )
        initial_condition_family_data = PerturbationInitialConditionFamilyData(
            name="adiabatic_scalar",
            sector="scalar",
        )
        projection_extension_data = PerturbationProjectionExtensionData(
            name="temperature_signal_extension",
            base_projection="line_of_sight_signal",
        )
        projection_typing_data = PerturbationProjectionTypingData(
            name="temperature_line_of_sight",
            sector="scalar",
        )
        dependency_summary = PerturbationDependencyGraphSummaryData(
            variable_names=("delta_x",),
            derived_names=("density_drive",),
            equation_names=("evolve_delta_x",),
            constraint_names=("poisson_phi",),
            closure_names=("psi_equals_phi",),
            interaction_names=("photon_baryon_slip",),
            conservation_rule_names=("density_balance",),
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
            interaction_dependencies={},
            conservation_rule_dependencies={},
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
        self.assertEqual(sector_data.name, "scalar")
        self.assertIsInstance(sector_data, PerturbationSectorData)
        self.assertEqual(species_data.name, "photon")
        self.assertIsInstance(species_data, PerturbationSpeciesData)
        self.assertEqual(hierarchy_family_data.name, "photon_temperature")
        self.assertIsInstance(
            hierarchy_family_data,
            PerturbationHierarchyFamilyData,
        )
        self.assertEqual(collision_operator_data.name, "thomson_drag")
        self.assertIsInstance(
            collision_operator_data,
            PerturbationCollisionOperatorData,
        )
        self.assertEqual(interaction_data.name, "photon_baryon_slip")
        self.assertIsInstance(
            interaction_data,
            PerturbationInteractionData,
        )
        self.assertEqual(conservation_rule_data.name, "density_balance")
        self.assertIsInstance(
            conservation_rule_data,
            PerturbationConservationRuleData,
        )
        self.assertEqual(
            initial_condition_family_data.name,
            "adiabatic_scalar",
        )
        self.assertIsInstance(
            initial_condition_family_data,
            PerturbationInitialConditionFamilyData,
        )
        self.assertEqual(
            projection_extension_data.base_projection,
            "line_of_sight_signal",
        )
        self.assertIsInstance(
            projection_extension_data,
            PerturbationProjectionExtensionData,
        )
        self.assertEqual(
            projection_typing_data.name,
            "temperature_line_of_sight",
        )
        self.assertIsInstance(
            projection_typing_data,
            PerturbationProjectionTypingData,
        )
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
