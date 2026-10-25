"""Tests for the bundled CCMBS contract inventory audit."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

from copernican.lib.likelihoods.cmb.contracts_audit import (
    CMBContractAudit,
    CMBModelDeclarationDecision,
    CMBSourceGraphAudit,
    _audit_declaration_plugin,
    _audit_source_graph_plugin,
    assert_bundled_cmb_contracts,
    assert_bundled_cmb_declarations,
    assert_bundled_cmb_source_graphs,
    audit_bundled_cmb_contracts,
    audit_bundled_cmb_declarations,
    audit_bundled_cmb_source_graphs,
)
from copernican.lib.perturbation_contract import (
    validate_generated_scalar_source_graph,
)


class CMBContractAuditTestCase(unittest.TestCase):
    """Verify that every bundled CMB declaration passes structural checks."""

    def test_bundled_contract_inventory_is_valid(self):
        """All bundled CMB contracts are present and structurally valid."""

        audits = audit_bundled_cmb_contracts()
        self.assertEqual(len(audits), 10)
        assert_bundled_cmb_contracts(audits)
        self.assertTrue(all(audit.valid for audit in audits))

    def test_audit_records_serialize_public_fields(self):
        """Audit records expose their validity and serialized fields."""

        audit = CMBContractAudit(
            model_filename="example.yml",
            model_name="Example",
            valid_for_cmb=True,
            contract_version=2,
            gauge="conformal_newtonian",
            sectors=("scalar",),
            hierarchy_families=("photon",),
            spectra=("TT", "TE", "EE"),
        )
        self.assertIsInstance(audit, CMBContractAudit)
        self.assertTrue(audit.valid)
        self.assertTrue(audit.to_dict()["valid"])

    def test_generated_source_graphs_are_complete(self) -> None:
        """Generated hierarchies expose source and metric metadata."""

        audits = audit_bundled_cmb_source_graphs()
        self.assertEqual(len(audits), 10)
        assert_bundled_cmb_source_graphs(audits)
        self.assertIsInstance(audits[0], CMBSourceGraphAudit)
        self.assertTrue(callable(assert_bundled_cmb_source_graphs))
        self.assertTrue(all(audit.valid for audit in audits))
        self.assertTrue(
            all(
                set(("Phi", "Psi")) <= set(audit.metric_state_names)
                for audit in audits
                if audit.generated_scalar_hierarchy
            )
        )
        self.assertTrue(
            all(
                set(("Phi_history_tau", "Phi_tau", "Psi_tau"))
                <= set(audit.metric_derivative_names)
                for audit in audits
                if audit.generated_scalar_hierarchy
            )
        )

    def test_bundled_declarations_have_explicit_theory_routes(self) -> None:
        """Every bundle is classified without an LCDM surrogate route."""

        decisions = audit_bundled_cmb_declarations()
        self.assertEqual(len(decisions), 10)
        assert_bundled_cmb_declarations(decisions)
        self.assertTrue(
            all(decision.decision == "ready" for decision in decisions)
        )
        self.assertEqual(
            {decision.execution_route for decision in decisions},
            {"generated_scalar_hierarchy", "explicit_scalar_graph"},
        )
        usmf2 = next(
            decision
            for decision in decisions
            if decision.model_filename == "model_usmf2.yml"
        )
        self.assertEqual(usmf2.execution_route, "explicit_scalar_graph")
        self.assertFalse(usmf2.generated_scalar_hierarchy)
        self.assertEqual(usmf2.theory_specific_source_names, ())
        qrsf = next(
            decision
            for decision in decisions
            if decision.model_filename == "model_qrsf.yml"
        )
        self.assertEqual(
            qrsf.theory_specific_source_names,
            (
                "qrsf_baryon_euler",
                "qrsf_matter_density",
                "qrsf_matter_momentum",
            ),
        )
        self.assertTrue(qrsf.source_rationales)
        self.assertTrue(
            all(decision.to_dict()["valid"] for decision in decisions)
        )

    def test_explicit_route_rejects_missing_projection_source_role(
        self,
    ) -> None:
        """An explicit graph cannot hide a missing temperature source."""

        plugin = SimpleNamespace(
            MODEL_FILENAME="broken-explicit.yml",
            MODEL_NAME="Broken Explicit",
            valid_for_cmb=True,
            CMB_PERTURBATION_DATA=SimpleNamespace(
                manifest_summary={"generated_scalar_hierarchy": False},
                variables={"Phi": object(), "Psi": object()},
                species={},
                sources={
                    "temperature_monopole": SimpleNamespace(
                        role="monopole",
                        description="Explicit monopole source",
                    )
                },
            ),
        )
        contract = CMBContractAudit(
            model_filename="broken-explicit.yml",
            model_name="Broken Explicit",
            valid_for_cmb=True,
            contract_version=2,
            gauge="conformal_newtonian",
            sectors=("scalar",),
            hierarchy_families=("photon_temperature",),
            spectra=("EE", "TE", "TT"),
        )
        graph = CMBSourceGraphAudit(
            model_filename="broken-explicit.yml",
            generated_scalar_hierarchy=False,
            metric_state_names=("Phi", "Psi"),
            metric_derivative_names=(),
            source_roles=("monopole",),
            closure_targets=(),
            compiled_source_count=1,
        )
        decision = _audit_declaration_plugin(
            plugin,
            contract_audit=contract,
            source_graph_audit=graph,
        )
        self.assertIsInstance(decision, CMBModelDeclarationDecision)
        self.assertEqual(decision.decision, "rejected")
        self.assertIn("missing source role", " ".join(decision.issues))

    def test_source_graph_audit_rejects_missing_metric_derivative(
        self,
    ) -> None:
        """Missing Phi/Psi history derivatives fail explicitly."""

        plugin = SimpleNamespace(
            MODEL_FILENAME="broken.yml",
            CMB_PERTURBATION_DATA=SimpleNamespace(
                manifest_summary={"generated_scalar_hierarchy": True},
                variables={"Phi": object(), "Psi": object()},
                derived={
                    "Phi_tau": SimpleNamespace(
                        expression="Psi",
                        kind="metric_potential_time_derivative",
                        dependencies=("Psi",),
                    )
                },
                sources={},
                closures={},
                initial_conditions={},
                initial_condition_families={},
            ),
        )

        audit = _audit_source_graph_plugin(plugin)
        self.assertFalse(audit.valid)
        self.assertIn("Phi_history_tau", " ".join(audit.issues))
        self.assertTrue(callable(_audit_source_graph_plugin))

    def test_source_graph_audit_shares_runtime_metric_requirements(self):
        """Audits and execution reject each missing Einstein energy input."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb

        contract = test_cmb._prepare_declared_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        perturbation_data = contract["perturbation_data"]
        plugin = SimpleNamespace(
            MODEL_FILENAME="generated.yml",
            CMB_PERTURBATION_DATA=perturbation_data,
        )
        self.assertTrue(_audit_source_graph_plugin(plugin).valid)
        validate_generated_scalar_source_graph(perturbation_data)
        for dependency in (
            "acoustic_k_sq",
            "Phi",
            "Hconf",
            "Psi",
            "einstein_gravity_strength",
            "total_density_source",
        ):
            with self.subTest(dependency=dependency):
                derived = dict(perturbation_data.derived)
                derivative = derived["Phi_tau"]
                derived["Phi_tau"] = replace(
                    derivative,
                    dependencies=tuple(
                        name
                        for name in derivative.dependencies
                        if name != dependency
                    ),
                )
                broken = replace(perturbation_data, derived=derived)
                plugin.CMB_PERTURBATION_DATA = broken
                with self.assertRaisesRegex(ValueError, dependency):
                    validate_generated_scalar_source_graph(broken)
                audit = _audit_source_graph_plugin(plugin)
                self.assertFalse(audit.valid)
                self.assertIn(dependency, " ".join(audit.issues))


if __name__ == "__main__":
    unittest.main()
