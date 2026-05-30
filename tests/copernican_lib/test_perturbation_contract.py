"""Tests for the perturbation contract compiler."""

from __future__ import annotations

import unittest

import copernican_lib.perturbation_contract as perturbation_contract_module
from copernican_lib.perturbation_contract import (
    PerturbationBackendMappingData,
    PerturbationClosureData,
    PerturbationContractData,
    PerturbationDependencyGraphSummaryData,
    PerturbationDerivativeLhsData,
    PerturbationDerivedData,
    PerturbationEquationData,
    PerturbationSourceData,
    PerturbationValidityData,
    PerturbationVariableData,
    compile_perturbation_contract,
)


class PerturbationContractTestCase(unittest.TestCase):
    """Validate the typed perturbation contract compiler."""

    def _make_nonstandard_contract(
        self,
        *,
        equals: object = "0",
        lhs: object | None = None,
    ) -> dict[str, object]:
        """Return a reusable non-standard perturbation contract fixture."""

        contract: dict[str, object] = {
            "contract_version": 1,
            "standard": False,
            "gauge": "conformal_newtonian",
            "variables": {
                "delta_x": {
                    "kind": "density_contrast",
                    "description": "Template density contrast.",
                },
                "theta_x": {
                    "kind": "velocity_divergence",
                    "description": "Template velocity divergence.",
                },
                "rho_x": {
                    "kind": "background_density",
                    "description": "Template density source.",
                },
                "sigma_x": {
                    "kind": "anisotropic_stress",
                    "description": "Template stress source.",
                },
            },
            "derived": {
                "Phi_tau": {
                    "kind": "derivative_symbol",
                    "variable": "Phi",
                    "wrt": "tau",
                    "order": 1,
                    "description": "Template derivative symbol.",
                },
                "delta_rho_eff": {
                    "expression": "rho_x * delta_x",
                    "description": "Template effective density.",
                },
            },
            "equations": {
                "continuity_x": {
                    "lhs": {
                        "kind": "derivative",
                        "variable": "delta_x",
                        "wrt": "tau",
                        "order": 1,
                    },
                    "rhs": "-theta_x + 3 * Phi_tau",
                }
            },
            "closures": {
                "no_anisotropic_stress": {
                    "expression": "sigma_x",
                    "equals": equals,
                }
            },
            "sources": {"poisson": {"expression": "delta_rho_eff + delta_x"}},
            "validity": {
                "regimes": ["linear", "scalar"],
                "notes": "Declared for first-order scalar perturbations.",
            },
            "backend_mapping": {
                "camb": {
                    "native_solver_required": True,
                    "implemented": False,
                }
            },
        }
        if lhs is not None:
            contract["equations"]["continuity_x"]["lhs"] = lhs
        return contract

    def test_module_symbols_are_exported(self) -> None:
        """The module should export the declared data symbols."""

        self.assertIs(
            perturbation_contract_module.compile_perturbation_contract,
            compile_perturbation_contract,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationContractData,
            PerturbationContractData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationVariableData,
            PerturbationVariableData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationDerivedData,
            PerturbationDerivedData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationDerivativeLhsData,
            PerturbationDerivativeLhsData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationEquationData,
            PerturbationEquationData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationClosureData,
            PerturbationClosureData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationSourceData,
            PerturbationSourceData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationValidityData,
            PerturbationValidityData,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationBackendMappingData,
            PerturbationBackendMappingData,
        )
        summary_data = (
            perturbation_contract_module.PerturbationDependencyGraphSummaryData
        )
        self.assertIs(summary_data, PerturbationDependencyGraphSummaryData)

    def test_standard_contract_compiles(self) -> None:
        """Standard contracts should compile into immutable data."""

        standard_contract_data = compile_perturbation_contract(
            {
                "contract_version": 1,
                "standard": True,
                "gauge": "unspecified",
                "variables": {},
                "derived": {},
                "equations": {},
                "closures": {},
                "sources": {},
                "validity": {
                    "regimes": ["standard_camb"],
                    "notes": "Uses standard backend perturbations.",
                },
                "backend_mapping": {
                    "camb": {"uses_standard_perturbations": True}
                },
            },
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )

        self.assertIsInstance(standard_contract_data, PerturbationContractData)
        self.assertTrue(standard_contract_data.standard)
        self.assertEqual(standard_contract_data.gauge, "unspecified")
        self.assertTrue(
            standard_contract_data.backend_mapping[
                "camb"
            ].uses_standard_perturbations
        )

    def test_nonstandard_contract_compiles(self) -> None:
        """Non-standard contracts should preserve typed equation metadata."""

        nonstandard_contract_data = compile_perturbation_contract(
            self._make_nonstandard_contract(),
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )

        self.assertIsInstance(
            nonstandard_contract_data, PerturbationContractData
        )
        self.assertFalse(nonstandard_contract_data.standard)
        self.assertEqual(
            nonstandard_contract_data.equations["continuity_x"].lhs.variable,
            "delta_x",
        )
        self.assertEqual(
            nonstandard_contract_data.derived["Phi_tau"].kind,
            "derivative_symbol",
        )
        dependency_summary = nonstandard_contract_data.dependency_graph_summary
        self.assertIn("tau", dependency_summary.independent_variables_used)

    def test_numeric_closure_equals_is_rejected(self) -> None:
        """Closure equality expressions must remain string literals."""

        with self.assertRaises(ValueError):
            compile_perturbation_contract(
                self._make_nonstandard_contract(equals=0),
                model_name="TemplateModel",
                backend="camb",
                parameter_names=("H0",),
                latex_names=("H_0",),
                background_reference_names=("H0",),
            )

    def test_quoted_closure_equals_compiles(self) -> None:
        """Quoted closure equality expressions should compile cleanly."""

        contract_data = compile_perturbation_contract(
            self._make_nonstandard_contract(equals="0"),
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )
        self.assertEqual(
            contract_data.closures["no_anisotropic_stress"].equals, "0"
        )

    def test_string_equation_lhs_is_rejected(self) -> None:
        """Free-text equation left-hand sides are not accepted."""

        with self.assertRaises(ValueError):
            compile_perturbation_contract(
                self._make_nonstandard_contract(lhs="delta_x"),
                model_name="TemplateModel",
                backend="camb",
                parameter_names=("H0",),
                latex_names=("H_0",),
                background_reference_names=("H0",),
            )

    def test_perturbation_dataclasses_are_constructible(self) -> None:
        """The typed perturbation objects should be individually usable."""

        variable_data = PerturbationVariableData(
            name="delta_x",
            kind="density_contrast",
        )
        derived_symbol_data = PerturbationDerivedData(
            name="Phi_tau",
            kind="derivative_symbol",
            variable="Phi",
            wrt="tau",
            order=1,
        )
        lhs_data = PerturbationDerivativeLhsData(
            kind="derivative",
            variable="delta_x",
            wrt="tau",
            order=1,
        )
        equation_data = PerturbationEquationData(
            name="continuity_x",
            lhs=lhs_data,
            rhs="-theta_x + 3 * Phi_tau",
        )
        closure_data = PerturbationClosureData(
            name="no_anisotropic_stress",
            expression="sigma_x",
            equals="0",
        )
        source_data = PerturbationSourceData(
            name="poisson",
            expression="delta_x",
        )
        validity_data = PerturbationValidityData(
            regimes=("linear",),
            notes="Template",
        )
        backend_mapping_data = PerturbationBackendMappingData(
            backend="camb",
            uses_standard_perturbations=True,
        )
        dependency_graph_data = PerturbationDependencyGraphSummaryData(
            variable_names=("delta_x",),
            derived_expression_names=("delta_rho_eff",),
            derivative_symbol_names=("Phi_tau",),
            equation_names=("continuity_x",),
            closure_names=("no_anisotropic_stress",),
            source_names=("poisson",),
            independent_variables_used=("tau",),
            model_parameters_used=("H0",),
            background_references_used=("Phi",),
            derived_expression_dependencies={"delta_rho_eff": ("delta_x",)},
            equation_dependencies={"continuity_x": ("delta_x",)},
            closure_dependencies={"no_anisotropic_stress": ("sigma_x",)},
            source_dependencies={"poisson": ("delta_x",)},
        )

        self.assertEqual(variable_data.kind, "density_contrast")
        self.assertEqual(derived_symbol_data.wrt, "tau")
        self.assertEqual(equation_data.lhs.variable, "delta_x")
        self.assertEqual(closure_data.equals, "0")
        self.assertEqual(source_data.expression, "delta_x")
        self.assertEqual(validity_data.regimes, ("linear",))
        self.assertTrue(backend_mapping_data.uses_standard_perturbations)
        self.assertIn("tau", dependency_graph_data.independent_variables_used)


if __name__ == "__main__":
    unittest.main()
