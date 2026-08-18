"""Tests for the selectable CMB solver contract and registry."""

import unittest

import numpy

from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.likelihoods.cmb.contracts import (
    CMBResult,
    CMBSolverCapabilities,
    CMBSolverProtocol,
)
from copernican.lib.likelihoods.cmb.errors import UnsupportedCapabilityError
from copernican.lib.likelihoods.cmb.solvers.registry import (
    CMB_SOLVER_REGISTRY,
    available_cmb_solvers,
    register_cmb_solver,
    resolve_cmb_solver,
)


class _FakeSolver:
    """Small deterministic backend used to exercise solver injection."""

    solver_id = "test_solver"
    solver_label = "Test solver"

    def capabilities(self):
        """Return capabilities without depending on declared model assets."""

        return {
            "solver_id": self.solver_id,
            "solver_label": self.solver_label,
            "execution_backend": "test",
            "supported_spectra": ("TT",),
        }

    def prepare(self, contract):
        """Return the test contract unchanged."""

        return contract

    def evaluate(self, prepared, ells, *, spectra, workload):
        """Return a deterministic spectrum with request metadata."""

        del workload
        value = float(prepared["value"])
        return CMBResult(
            spectra=numpy.full(len(tuple(ells)), value),
            requested_ells=tuple(ells),
            requested_spectra=tuple(spectra),
            solver_id=self.solver_id,
            solver_label=self.solver_label,
        )

    def evaluate_batch(self, prepared, ells, *, spectra, workload):
        """Evaluate test contracts in their supplied order."""

        return tuple(
            self.evaluate(
                item,
                ells,
                spectra=spectra,
                workload=workload,
            )
            for item in prepared
        )

    def cleanup(self):
        """Release no resources for the test backend."""


class TestCMBSolverContract(unittest.TestCase):
    """Exercise selection, ordered evaluation, and typed failure boundaries."""

    def test_default_solver_is_ccmbs_and_reports_capabilities(self):
        """The registry exposes CCMBS as the stable default backend."""

        solver = resolve_cmb_solver()
        self.assertEqual(solver.solver_id, "ccmbs_numpy")
        self.assertIn("ccmbs_numpy", available_cmb_solvers())
        self.assertEqual(solver.capabilities()["execution_backend"], "cpu")

    def test_contract_symbols_expose_the_solver_lifecycle(self):
        """Public contract symbols retain typed lifecycle entry points."""

        capabilities = CMBSolverCapabilities(
            solver_id="test",
            solver_label="Test",
        )
        self.assertIsInstance(capabilities, CMBSolverCapabilities)
        self.assertEqual(capabilities.to_mapping()["solver_id"], "test")
        self.assertTrue(callable(CMBResult.raise_for_failure))
        self.assertTrue(callable(CMBResult.to_dict))
        self.assertTrue(callable(CMBSolverProtocol.prepare))
        self.assertTrue(callable(CMBSolverProtocol.evaluate))
        self.assertTrue(callable(CMBSolverProtocol.evaluate_batch))
        self.assertTrue(callable(CMBSolverProtocol.cleanup))

    def test_public_wrapper_injects_registered_solver(self):
        """A manifest-style solver selection controls scalar evaluation."""

        previous = CMB_SOLVER_REGISTRY.get(_FakeSolver.solver_id)
        register_cmb_solver(_FakeSolver(), replace=True)
        try:
            spectrum = cmb.compute_cmb_spectrum_from_contract(
                {"value": 2.5},
                (2, 4),
                solver="test_solver",
            )
        finally:
            if previous is None:
                CMB_SOLVER_REGISTRY.pop(_FakeSolver.solver_id, None)
            else:
                CMB_SOLVER_REGISTRY[_FakeSolver.solver_id] = previous
        numpy.testing.assert_array_equal(spectrum, [2.5, 2.5])

    def test_result_serializes_order_and_solver_provenance(self):
        """Results retain request order and stable backend identity."""

        result = _FakeSolver().evaluate(
            {"value": 1.0},
            (8, 3),
            spectra=("TT",),
            workload="test",
        )
        payload = result.to_dict()
        self.assertEqual(payload["requested_ells"], (8, 3))
        self.assertEqual(payload["solver_id"], "test_solver")
        self.assertTrue(result.success)

    def test_unknown_solver_fails_as_typed_capability_error(self):
        """Unknown manifest selections fail before any numerical work."""

        with self.assertRaises(UnsupportedCapabilityError):
            resolve_cmb_solver("missing_solver")


if __name__ == "__main__":
    unittest.main()
