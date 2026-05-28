"""Tests for the perturbation contract compiler."""

from __future__ import annotations

import unittest

import copernican_lib.perturbation_contract as perturbation_contract_module
from copernican_lib.perturbation_contract import (
    PerturbationBackendMappingIR,
    PerturbationClosureIR,
    PerturbationContractIR,
    PerturbationDependencyGraphSummaryIR,
    PerturbationDerivativeLHSIR,
    PerturbationDerivedIR,
    PerturbationEquationIR,
    PerturbationSourceIR,
    PerturbationValidityIR,
    PerturbationVariableIR,
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
                    "solver": "template_native_solver",
                    "implemented": False,
                }
            },
        }
        if lhs is not None:
            contract["equations"]["continuity_x"]["lhs"] = lhs
        return contract

    def test_module_symbols_are_exported(self) -> None:
        """The module should export the declared IR symbols."""

        self.assertIs(
            perturbation_contract_module.compile_perturbation_contract,
            compile_perturbation_contract,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationContractIR,
            PerturbationContractIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationVariableIR,
            PerturbationVariableIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationDerivedIR,
            PerturbationDerivedIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationDerivativeLHSIR,
            PerturbationDerivativeLHSIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationEquationIR,
            PerturbationEquationIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationClosureIR,
            PerturbationClosureIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationSourceIR,
            PerturbationSourceIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationValidityIR,
            PerturbationValidityIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationBackendMappingIR,
            PerturbationBackendMappingIR,
        )
        self.assertIs(
            perturbation_contract_module.PerturbationDependencyGraphSummaryIR,
            PerturbationDependencyGraphSummaryIR,
        )

    def test_standard_contract_compiles(self) -> None:
        """Standard contracts should compile into immutable IR."""

        standard_contract_ir = compile_perturbation_contract(
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

        self.assertIsInstance(standard_contract_ir, PerturbationContractIR)
        self.assertTrue(standard_contract_ir.standard)
        self.assertEqual(standard_contract_ir.gauge, "unspecified")
        self.assertTrue(
            standard_contract_ir.backend_mapping[
                "camb"
            ].uses_standard_perturbations
        )

    def test_nonstandard_contract_compiles(self) -> None:
        """Non-standard contracts should preserve typed equation metadata."""

        nonstandard_contract_ir = compile_perturbation_contract(
            self._make_nonstandard_contract(),
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )

        self.assertIsInstance(nonstandard_contract_ir, PerturbationContractIR)
        self.assertFalse(nonstandard_contract_ir.standard)
        self.assertEqual(
            nonstandard_contract_ir.equations["continuity_x"].lhs.variable,
            "delta_x",
        )
        self.assertEqual(
            nonstandard_contract_ir.derived["Phi_tau"].kind,
            "derivative_symbol",
        )
        dependency_summary = nonstandard_contract_ir.dependency_graph_summary
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

        contract_ir = compile_perturbation_contract(
            self._make_nonstandard_contract(equals="0"),
            model_name="TemplateModel",
            backend="camb",
            parameter_names=("H0",),
            latex_names=("H_0",),
            background_reference_names=("H0",),
        )
        self.assertEqual(
            contract_ir.closures["no_anisotropic_stress"].equals, "0"
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

        variable_ir = PerturbationVariableIR(
            name="delta_x",
            kind="density_contrast",
        )
        derived_symbol_ir = PerturbationDerivedIR(
            name="Phi_tau",
            kind="derivative_symbol",
            variable="Phi",
            wrt="tau",
            order=1,
        )
        lhs_ir = PerturbationDerivativeLHSIR(
            kind="derivative",
            variable="delta_x",
            wrt="tau",
            order=1,
        )
        equation_ir = PerturbationEquationIR(
            name="continuity_x",
            lhs=lhs_ir,
            rhs="-theta_x + 3 * Phi_tau",
        )
        closure_ir = PerturbationClosureIR(
            name="no_anisotropic_stress",
            expression="sigma_x",
            equals="0",
        )
        source_ir = PerturbationSourceIR(
            name="poisson",
            expression="delta_x",
        )
        validity_ir = PerturbationValidityIR(
            regimes=("linear",),
            notes="Template",
        )
        backend_mapping_ir = PerturbationBackendMappingIR(
            backend="camb",
            uses_standard_perturbations=True,
        )
        dependency_graph_ir = PerturbationDependencyGraphSummaryIR(
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

        self.assertEqual(variable_ir.kind, "density_contrast")
        self.assertEqual(derived_symbol_ir.wrt, "tau")
        self.assertEqual(equation_ir.lhs.variable, "delta_x")
        self.assertEqual(closure_ir.equals, "0")
        self.assertEqual(source_ir.expression, "delta_x")
        self.assertEqual(validity_ir.regimes, ("linear",))
        self.assertTrue(backend_mapping_ir.uses_standard_perturbations)
        self.assertIn("tau", dependency_graph_ir.independent_variables_used)


if __name__ == "__main__":
    unittest.main()
