"""Tests for the optional Taichi accelerator boundary."""

import unittest
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.errors import EngineCapabilityError
from copernican.lib.likelihoods.cmb.solvers.ccmbs_taichi import (
    CCMBSTaichiSolver,
    apply_taichi_two_state_collision,
    taichi_device_probe,
)
from copernican.lib.likelihoods.cmb.solvers.registry import (
    available_cmb_solvers,
    resolve_cmb_solver,
)


class TestCCMBSTaichiBoundary(unittest.TestCase):
    """Check explicit device identity and fail-closed optional behavior."""

    def test_registry_exposes_optional_backend_without_importing_taichi(self):
        """The optional solver is discoverable without eager device import."""

        self.assertIn("ccmbs_taichi", available_cmb_solvers())
        solver = resolve_cmb_solver("ccmbs_taichi")
        self.assertIsInstance(solver, CCMBSTaichiSolver)
        capabilities = solver.capabilities()
        self.assertEqual(capabilities["execution_backend"], "accelerator")
        self.assertEqual(
            capabilities["implementation"],
            "taichi_fixed_shape_collision_pilot",
        )
        self.assertFalse(capabilities["device_probe"]["taichi_imported"])

    def test_selected_backend_reports_typed_full_route_gap(self):
        """Explicit accelerator selection cannot silently use the CPU path."""

        solver = CCMBSTaichiSolver()
        self.assertTrue(callable(solver.prepare))
        self.assertTrue(callable(solver.evaluate))
        self.assertTrue(callable(solver.evaluate_batch))
        self.assertTrue(callable(solver.cleanup))
        self.assertIn("apply", apply_taichi_two_state_collision.__name__)
        result = solver.evaluate(
            solver.prepare({"model_name": "fixture"}),
            (2, 20),
            spectra=("TT",),
            workload="test",
        )
        self.assertIsInstance(result.failure, EngineCapabilityError)
        self.assertEqual(result.solver_id, "ccmbs_taichi")
        self.assertFalse(result.success)
        batch = solver.evaluate_batch(
            ({"model_name": "fixture"},),
            (2, 20),
            spectra=("TT",),
            workload="test",
        )
        self.assertEqual(len(batch), 1)
        self.assertIsInstance(batch[0].failure, EngineCapabilityError)
        self.assertIsNone(solver.cleanup())

    def test_missing_optional_runtime_is_typed(self):
        """A missing Taichi package is non-applicability, not CPU fallback."""

        matrices = numpy.asarray(
            (((-1.0, 0.2), (0.1, -0.5)),),
            dtype=float,
        )
        states = numpy.asarray(((0.4, -0.2),), dtype=float)
        scales = numpy.asarray((0.75,), dtype=float)
        with mock.patch(
            "importlib.util.find_spec",
            return_value=None,
        ):
            with self.assertRaises(EngineCapabilityError) as raised:
                apply_taichi_two_state_collision(
                    matrices,
                    states,
                    scales,
                    0.125,
                )
        self.assertIn("not installed", str(raised.exception))
        self.assertEqual(
            raised.exception.context["device_probe"]["backend"],
            "taichi",
        )

    def test_device_probe_is_path_free_and_deterministic(self):
        """Capability evidence contains no machine-local filesystem paths."""

        first = taichi_device_probe()
        second = taichi_device_probe()
        self.assertEqual(first, second)
        self.assertNotIn("/Users/", repr(first))
        self.assertNotIn("/home/", repr(first))


if __name__ == "__main__":
    unittest.main()
