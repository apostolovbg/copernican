"""Coverage for the single public CCMBS engine boundary."""

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy

from copernican.lib.likelihoods.cmb.errors import EngineCapabilityError
from copernican.lib.likelihoods.cmb.solvers import ccmbs as ccmbs_engine
from copernican.lib.likelihoods.cmb.solvers.ccmbs import CCMBS


class TestCCMBSEngine(unittest.TestCase):
    """Check public identity and internal backend selection evidence."""

    def test_engine_owns_backend_selection(self):
        """The public engine reports CPU fallback provenance."""

        solver = CCMBS()
        with mock.patch(
            "copernican.lib.likelihoods.cmb.solvers.ccmbs._device_probe",
            return_value={"backend": "taichi", "taichi_installed": False},
        ):
            capabilities = solver.capabilities()
        self.assertEqual(capabilities["solver_id"], "ccmbs")
        self.assertEqual(capabilities["execution_backend"], "cpu")
        provenance = capabilities["device_probe"]
        self.assertEqual(provenance["selected_backend"], "cpu")
        self.assertTrue(provenance["fallback"])
        self.assertEqual(
            provenance["fallback_reason"],
            "accelerator_unavailable_or_not_installed",
        )

    def test_missing_optional_runtime_fails_inside_engine(self):
        """An unavailable optional device is a typed internal capability
        gap.
        """

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
                ccmbs_engine._apply_accelerated_collision(
                    matrices,
                    states,
                    scales,
                    0.125,
                )
        self.assertIn("not installed", str(raised.exception))
        self.assertEqual(
            raised.exception.context["device_probe"]["backend"],
            "accelerator",
        )

    def test_device_probe_is_path_free_and_deterministic(self):
        """Engine device evidence contains no machine-local file paths."""

        with mock.patch(
            "importlib.util.find_spec",
            return_value=None,
        ):
            first = ccmbs_engine._device_probe()
            second = ccmbs_engine._device_probe()
        self.assertEqual(first, second)
        self.assertNotIn("/Users/", repr(first))
        self.assertNotIn("/home/", repr(first))

    def test_prepare_uses_the_engine_contract_boundary(self):
        """Preparation stays behind the single CCMBS engine boundary."""

        prepared = {"runtime_signature": "test-signature"}
        target = (
            "copernican.lib.likelihoods.cmb.cmb."
            "prepare_cmb_execution_contract"
        )
        with mock.patch(
            target,
            return_value=prepared,
        ) as normalizer:
            result = CCMBS().prepare({"model_name": "test"})
        self.assertEqual(result, prepared)
        normalizer.assert_called_once_with({"model_name": "test"})

    def test_evaluate_returns_a_typed_failure_for_invalid_preparation(self):
        """Evaluation does not leak an untyped preparation failure."""

        result = CCMBS().evaluate(
            object(),
            (2,),
            spectra=("TT",),
            workload="test",
        )
        self.assertFalse(result.success, "evaluate returned success")
        self.assertIsNotNone(result.failure)

    def test_evaluate_batch_preserves_prepared_request_order(self):
        """Batch evaluation delegates in input order."""

        solver = CCMBS()
        with mock.patch.object(
            solver,
            "evaluate",
            side_effect=("first", "second"),
        ):
            result = solver.evaluate_batch(
                (object(), object()),
                (2,),
                spectra=("TT",),
                workload="test",
            )
        # evaluate_batch
        self.assertEqual(
            result,
            ("first", "second"),
            "evaluate_batch changed request order",
        )

    def test_cleanup_completes_without_a_public_backend_resource(self):
        """Cleanup remains a no-op at the public engine boundary."""

        self.assertIsNone(CCMBS().cleanup())

    def test_fixed_shape_accelerator_pilot_matches_cpu_reference(self):
        """The internal fixed-shape pilot validates against CPU results."""

        class FakeField:
            def __init__(self, shape):
                self.value = numpy.zeros(shape, dtype=float)

            def from_numpy(self, value):
                self.value[...] = value

            def to_numpy(self):
                return self.value.copy()

            def __getitem__(self, index):
                return self.value[index]

            def __setitem__(self, index, value):
                self.value[index] = value

        accelerator = SimpleNamespace(
            f32=float,
            field=lambda dtype, shape: FakeField(shape),
            kernel=lambda function: function,
            sqrt=numpy.sqrt,
            sinh=numpy.sinh,
            cosh=numpy.cosh,
            sin=numpy.sin,
            cos=numpy.cos,
            exp=numpy.exp,
        )
        matrices = numpy.asarray(
            (((-1.0, 0.2), (0.1, -0.5)),),
            dtype=float,
        )
        states = numpy.asarray(((0.4, -0.2),), dtype=float)
        scales = numpy.asarray((0.75,), dtype=float)
        with mock.patch.object(
            ccmbs_engine,
            "_load_accelerator",
            return_value=(accelerator, "fake"),
        ):
            result = ccmbs_engine._apply_accelerated_collision(
                matrices,
                states,
                scales,
                0.125,
            )
        expected = ccmbs_engine._cpu_two_state_collision(
            matrices,
            states,
            scales,
            0.125,
        )
        self.assertTrue(
            numpy.allclose(result, expected),
            "apply pilot diverged from the CPU reference",
        )


if __name__ == "__main__":
    unittest.main()
