"""Tests for the bundled CCMBS contract inventory audit."""

import unittest
from types import SimpleNamespace

from copernican.lib.likelihoods.cmb.contracts_audit import (
    CMBContractAudit,
    CMBSourceGraphAudit,
    _audit_source_graph_plugin,
    assert_bundled_cmb_contracts,
    assert_bundled_cmb_source_graphs,
    audit_bundled_cmb_contracts,
    audit_bundled_cmb_source_graphs,
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


if __name__ == "__main__":
    unittest.main()
