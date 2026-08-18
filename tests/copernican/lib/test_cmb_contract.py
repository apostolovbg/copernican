"""Smoke tests for :mod:`copernican.lib.cmb_contract`."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from copernican.lib import cmb_contract
from copernican.lib.model_coder import compile_native_cmb_runtime


class TestCMBContractExports(unittest.TestCase):
    """Verify the contract module mirrors the native adapter helpers."""

    def test_reexports_match_expected_helpers(self) -> None:
        """The route-neutral evaluators should be the only public surface."""

        self.assertTrue(hasattr(cmb_contract, "CMBContractEvaluator"))
        self.assertTrue(hasattr(cmb_contract, "CMBParameterEvaluator"))
        self.assertFalse(hasattr(cmb_contract, "CMB_BACKEND_CAPABILITIES"))
        self.assertTrue(
            hasattr(cmb_contract, "_validate_cmb_contract_definition")
        )
        self.assertIn("CMBContractEvaluator", cmb_contract.__all__)
        self.assertNotIn("CMB_BACKEND_CAPABILITIES", cmb_contract.__all__)
        self.assertIn(
            "_validate_cmb_contract_definition",
            cmb_contract.__all__,
        )
        self.assertTrue(callable(cmb_contract.CMBCapabilityAudit))
        self.assertTrue(callable(cmb_contract.CMBObservableCapability))
        self.assertTrue(callable(cmb_contract.CMBObservableRequirement))
        self.assertTrue(callable(cmb_contract.audit_cmb_capabilities))
        self.assertTrue(callable(cmb_contract.build_cmb_capability_matrix))
        self.assertTrue(callable(cmb_contract.require_cmb_capability))
        self.assertIn("CMBCapabilityAudit", cmb_contract.__all__)
        self.assertIn("CMBObservableCapability", cmb_contract.__all__)
        self.assertIn("CMBObservableRequirement", cmb_contract.__all__)
        self.assertIn("audit_cmb_capabilities", cmb_contract.__all__)
        self.assertIn("build_cmb_capability_matrix", cmb_contract.__all__)
        self.assertIn("require_cmb_capability", cmb_contract.__all__)

    def test_observable_requirements_define_the_public_capability_surface(
        self,
    ) -> None:
        """The audit must define one minimum contract per public spectrum."""

        requirements = cmb_contract.CMB_OBSERVABLE_REQUIREMENTS
        self.assertEqual(
            tuple(requirements),
            ("TT", "TE", "EE", "BB", "PP", "TP", "EP"),
        )
        self.assertEqual(
            requirements["TE"].required_transfer_roles,
            ("temperature", "polarization_e"),
        )
        self.assertEqual(
            requirements["PP"].required_transfer_roles,
            ("potential", "potential"),
        )
        self.assertEqual(
            requirements["EP"].required_sectors,
            ("scalar",),
        )

    @staticmethod
    def _compile_model(filename: str):
        """Compile one bundled model for capability-audit assertions."""

        model_path = (
            Path(__file__).parents[3] / "copernican" / "models" / filename
        )
        model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
        parameters = tuple(
            str(entry.get("python_var") or entry.get("name"))
            for entry in model.get("parameters", [])
        )
        latex_names = tuple(
            str(entry.get("latex_name", ""))
            for entry in model.get("parameters", [])
        )
        return compile_native_cmb_runtime(
            model_name=model["model_name"],
            parameter_names=parameters,
            latex_names=latex_names,
            cmb_contract=model["cmb"],
        ).perturbation_data

    def test_audit_reports_declared_model_capabilities_and_route(self) -> None:
        """The audit must expose all standard spectra and declared ontology."""

        audit = cmb_contract.audit_cmb_capabilities(
            self._compile_model("model_lcdm.yml")
        )
        self.assertEqual(audit.model_name, "LambdaCDM")
        self.assertEqual(audit.sectors, ("scalar",))
        self.assertIn("cdm", audit.species)
        self.assertIn("massless_neutrino", audit.hierarchy_families)
        self.assertIn("Omega_c0", audit.background_references)
        self.assertIn("ell_max", audit.numerical_controls)
        self.assertIn("linear", audit.validity_regimes)
        self.assertEqual(audit.generated_hierarchies, ("scalar",))
        self.assertEqual(
            audit.supported_observables,
            ("TT", "TE", "EE", "BB", "PP", "TP", "EP"),
        )
        self.assertEqual(audit.unsupported_observables, ())
        self.assertEqual(
            audit.execution_solver_id,
            "ccmbs_numpy",
        )
        self.assertEqual(
            audit.execution_runtime_module,
            "copernican.lib.likelihoods.cmb.copernican_cmb_solver",
        )
        self.assertEqual(
            audit.to_mapping()["supported_observables"],
            audit.supported_observables,
        )
        self.assertEqual(
            audit.to_mapping()["capability_matrix"]["EP"]["declared_sector"],
            "scalar",
        )

    def test_capability_matrix_is_model_keyed_and_route_neutral(self) -> None:
        """Corpus rows must be derived from declarations, not model labels."""

        lcdm = self._compile_model("model_lcdm.yml")
        qrsf = self._compile_model("model_qrsf.yml")
        matrix = cmb_contract.build_cmb_capability_matrix(
            {"lcdm-file.yml": lcdm, "qrsf-file.yml": qrsf}
        )
        self.assertEqual(tuple(matrix), ("LambdaCDM", "QRSF"))
        self.assertNotIn("cdm", matrix["QRSF"].species)
        self.assertEqual(
            matrix["QRSF"].supported_observables,
            matrix["LambdaCDM"].supported_observables,
        )
        self.assertEqual(
            matrix["QRSF"].execution_solver_id,
            matrix["LambdaCDM"].execution_solver_id,
        )

    def test_full_bundled_cmb_corpus_has_machine_testable_rows(self) -> None:
        """Every CMB-enabled bundled model must satisfy the public matrix."""

        contracts = {}
        for model_path in sorted(
            (Path(__file__).parents[3] / "copernican" / "models").glob(
                "model_*.yml"
            )
        ):
            model = yaml.safe_load(model_path.read_text(encoding="utf-8"))
            if model.get("valid_for_cmb"):
                contracts[model_path.name] = self._compile_model(
                    model_path.name
                )
        matrix = cmb_contract.build_cmb_capability_matrix(contracts)
        self.assertEqual(len(matrix), 10)
        self.assertIn("USMFv2", matrix)
        for audit in matrix.values():
            self.assertEqual(audit.unsupported_observables, ())
            self.assertEqual(
                audit.supported_observables,
                ("TT", "TE", "EE", "BB", "PP", "TP", "EP"),
            )

    def test_unsupported_capabilities_fail_before_execution(self) -> None:
        """Unknown or incomplete spectra must have explicit diagnostics."""

        audit = cmb_contract.audit_cmb_capabilities(
            self._compile_model("model_lcdm.yml")
        )
        with self.assertRaisesRegex(
            ValueError, "supported public observables"
        ):
            cmb_contract.require_cmb_capability(audit, "XX")
        row = cmb_contract.require_cmb_capability(audit, "te")
        self.assertTrue(row.available)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
