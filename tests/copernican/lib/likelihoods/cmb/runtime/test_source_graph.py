"""Focused tests for the universal CCMBS source graph compiler."""

import unittest
from types import SimpleNamespace

from copernican.lib.likelihoods.cmb.runtime.source_graph import (
    DeclaredSourceGraph,
    compile_declared_source_graph,
)


class SourceGraphCompilerTestCase(unittest.TestCase):
    """Exercise model-independent source and projection graph assembly."""

    @staticmethod
    def _graph_contract(*, model_name: str | None = None) -> object:
        """Return a minimal complete scalar source-to-spectrum graph."""

        source = SimpleNamespace(
            role="monopole",
            expression="theta_gamma0 + Phi",
            dependencies=("theta_gamma0", "Phi"),
            units="dimensionless",
            domain="scalar",
        )
        transfer = SimpleNamespace(
            kind="transfer_component",
            projection="line_of_sight_temperature",
            kernel="temperature_mixed_window",
            sector="scalar",
            output_role="temperature",
            parity="even",
            spin=0.0,
            units="dimensionless",
            source_terms={"monopole": "temperature_monopole"},
            required_projection_roles=(),
        )
        spectrum = SimpleNamespace(
            kind="angular_power_spectrum",
            primary="temperature",
            secondary="temperature",
            sector="scalar",
        )
        values = {
            "sources": {"temperature_monopole": source},
            "observables": {"temperature": transfer, "TT": spectrum},
            "projection_extensions": {},
        }
        if model_name is not None:
            values["model_name"] = model_name
        return SimpleNamespace(**values)

    def test_graph_digest_ignores_model_name_and_records_kernel_route(self):
        """One route graph is stable when only the theory label changes."""

        first = compile_declared_source_graph(
            self._graph_contract(),
            requested_spectra=("TT",),
        )
        second = compile_declared_source_graph(
            self._graph_contract(model_name="renamed-theory"),
            requested_spectra=("TT",),
        )

        self.assertIsInstance(first, DeclaredSourceGraph)
        self.assertEqual(first.to_dict()["digest"], first.digest)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(len(first.source_nodes), 1)
        self.assertEqual(len(first.transfer_routes), 1)
        self.assertTrue(first.transfer_routes[0]["active"])
        self.assertEqual(
            first.transfer_routes[0]["source_kernel_kinds"]["monopole"],
            "spherical_bessel",
        )
        self.assertEqual(first.spectrum_edges[0]["name"], "TT")

    def test_graph_rejects_unknown_history_before_projection(self):
        """A route may not fabricate a missing source history."""

        transfer = SimpleNamespace(
            kind="transfer_component",
            projection="line_of_sight_temperature",
            kernel="temperature_mixed_window",
            sector="scalar",
            source_terms={"monopole": "missing"},
            required_projection_roles=(),
        )
        graph_contract = SimpleNamespace(
            sources={},
            observables={"temperature": transfer},
            projection_extensions={},
        )
        with self.assertRaisesRegex(ValueError, "unknown source histories"):
            compile_declared_source_graph(graph_contract)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
