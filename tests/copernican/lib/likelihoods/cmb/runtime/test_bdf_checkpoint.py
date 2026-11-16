"""Verify exact continuation of the pinned continuous BDF solver."""

import subprocess  # nosec B404 - fixed local Python test producers.
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix

from copernican.lib.likelihoods.cmb import evidence_assets
from copernican.lib.likelihoods.cmb.durable_history import (
    BDFCheckpoint,
    DurableHistories,
)
from copernican.lib.likelihoods.cmb.runtime import bdf_checkpoint as bdf


class BDFCheckpointTestCase(unittest.TestCase):
    """Check sampled histories, solver state and corruption handling."""

    def test_restart_matches_original_at_every_order_and_lu_backend(self):
        """Resume accepted steps at orders one through five without drift."""

        eta = numpy.linspace(0.0, 3.0, 101)
        initial = numpy.array([1.0, 0.7, -0.2])
        matrix = numpy.array(
            [[-0.2, 0.1, 0.0], [0.0, -2.0, 0.2], [0.1, 0.0, -9.0]]
        )
        for sparse in (False, True):
            for order in range(1, 6):
                with self.subTest(sparse=sparse, order=order):
                    calls = [0]

                    def rhs(epoch, values):
                        calls[0] += 1
                        return matrix @ values + numpy.sin(epoch)

                    def jac(epoch, values):
                        return csc_matrix(matrix) if sparse else matrix

                    def restore_calls(count):
                        calls[0] = count

                    original = solve_ivp(
                        rhs,
                        (eta[0], eta[-1]),
                        initial,
                        method="BDF",
                        jac=jac,
                        t_eval=eta,
                        rtol=1e-7,
                        atol=1e-9,
                    )
                    original_calls = calls[0]
                    self.assertTrue(original.success)
                    with tempfile.TemporaryDirectory() as root:
                        identity = {
                            "physical": "a" * 64,
                            "numerical": "b" * 64,
                        }
                        store = DurableHistories(Path(root), identity, "first")

                        class Interrupted(BDFCheckpoint):
                            def due(self):
                                return True

                            def record(self, history, evaluated, state):
                                if state["order"] == order:
                                    super().record(history, evaluated, state)
                                    raise RuntimeError("interrupted")

                        calls[0] = 0
                        with self.assertRaisesRegex(
                            RuntimeError, "interrupted"
                        ):
                            bdf.integrate_bdf_checkpointed(
                                rhs,
                                jac,
                                eta,
                                initial,
                                rtol=1e-7,
                                atol=1e-9,
                                checkpoint=Interrupted(
                                    store, ("mode",), interval_seconds=0
                                ),
                                rhs_calls=lambda: calls[0],
                                restore_rhs_calls=restore_calls,
                                initial_rhs_calls=0,
                            )
                        calls[0] = 0
                        checkpoint = BDFCheckpoint(
                            DurableHistories(Path(root), identity, "resume"),
                            ("mode",),
                            interval_seconds=1000,
                        )
                        resumed = bdf.integrate_bdf_checkpointed(
                            rhs,
                            jac,
                            eta,
                            initial,
                            rtol=1e-7,
                            atol=1e-9,
                            checkpoint=checkpoint,
                            rhs_calls=lambda: calls[0],
                            restore_rhs_calls=restore_calls,
                            initial_rhs_calls=0,
                        )
                        numpy.testing.assert_array_equal(resumed, original.y)
                        self.assertEqual(calls[0], original_calls)
                        self.assertGreater(checkpoint.restored_steps, 0)
                        self.assertGreater(checkpoint.restored_nfev, 0)

    def test_matrix_validation_and_factor_identity(self):
        """Reject malformed sparse structures and changed numerical factors."""

        for matrix in (numpy.eye(3), csc_matrix(numpy.eye(3))):
            encoded = bdf._encode_bdf_matrix(matrix)
            decoded = bdf._decode_bdf_matrix(encoded, 3)
            actual = (
                decoded.toarray() if hasattr(decoded, "toarray") else decoded
            )
            numpy.testing.assert_array_equal(actual, numpy.eye(3))
            self.assertTrue(bdf._identical_bdf_product(encoded, encoded))
        sparse = bdf._encode_bdf_matrix(csc_matrix(numpy.eye(3)))
        sparse["indptr"][1] = 5
        with self.assertRaisesRegex(ValueError, "sparse matrix"):
            bdf._decode_bdf_matrix(sparse, 3)
        self.assertFalse(
            bdf._identical_bdf_product(
                {"factor": numpy.array([1.0])}, {"factor": numpy.array([2.0])}
            )
        )
        self.assertEqual(len(bdf.bdf_implementation_identity()), 64)

    def test_traced_lu_requires_identical_reconstructed_factors(self):
        """A numerically different reconstructed factorization is rejected."""

        eta = numpy.linspace(0.0, 3.0, 101)
        initial = numpy.array([1.0, 0.5])
        matrix = csc_matrix([[-1.0, 0.1], [0.0, -4.0]])

        def rhs(epoch, values):
            return matrix @ values + numpy.sin(epoch)

        def jac(epoch, values):
            return matrix

        original_step = bdf.BDF.step

        def observe_step(solver):
            self.assertEqual(solver.lu.__name__, "traced_lu")
            return original_step(solver)

        class Interrupted(BDFCheckpoint):
            def due(self):
                return True

            def record(self, history, evaluated, state):
                if state["factors"] is not None:
                    super().record(history, evaluated, state)
                    raise RuntimeError("interrupted")

        with tempfile.TemporaryDirectory() as root:
            identity = {"physical": "a" * 64, "numerical": "b" * 64}
            store = DurableHistories(Path(root), identity, "first")
            args = (rhs, jac, eta, initial)
            kwargs = {
                "rtol": 1e-7,
                "atol": 1e-9,
                "rhs_calls": lambda: 0,
                "restore_rhs_calls": lambda value: None,
                "initial_rhs_calls": 0,
            }
            with (
                mock.patch.object(bdf.BDF, "step", observe_step),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                bdf.integrate_bdf_checkpointed(
                    *args,
                    checkpoint=Interrupted(
                        store, ("mode",), interval_seconds=0
                    ),
                    **kwargs,
                )
            with (
                mock.patch.object(
                    bdf, "_identical_bdf_product", return_value=False
                ),
                self.assertRaisesRegex(ValueError, "LU factors differ"),
            ):
                bdf.integrate_bdf_checkpointed(
                    *args, checkpoint=BDFCheckpoint(store, ("mode",)), **kwargs
                )

    def test_checkpoint_chunks_are_atomic_and_fail_closed(self):
        """Accept a step with no new samples and reject corrupted chunks."""

        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            identity = {"physical": "a" * 64, "numerical": "b" * 64}
            store = DurableHistories(root, identity, "first")
            checkpoint = BDFCheckpoint(store, ("mode",), interval_seconds=0)
            history = numpy.arange(8.0).reshape(2, 4)
            first = {"accepted_steps": 1, "t": 0.1, "evaluated": 1, "nfev": 2}
            second = {"accepted_steps": 2, "t": 0.2, "evaluated": 1, "nfev": 4}
            checkpoint.record(history, 1, first)
            pointer = evidence_assets.contained_path(
                root, checkpoint.stem + ".bdf-index.json"
            )
            old_index = pointer.read_bytes()
            with (
                mock.patch.object(
                    evidence_assets, "write_json_artifact", side_effect=OSError
                ),
                self.assertRaises(OSError),
            ):
                checkpoint.record(history, 1, second)
            self.assertEqual(pointer.read_bytes(), old_index)
            checkpoint.record(history, 1, second)
            restored = BDFCheckpoint(store, ("mode",))
            output = numpy.empty_like(history)
            self.assertEqual(restored.restore(output), second)
            numpy.testing.assert_array_equal(output[:, :1], history[:, :1])
            self.assertEqual(restored.restored_steps, 2)
            self.assertEqual(restored.restored_samples, 1)
            self.assertIsNone(BDFCheckpoint(store, ("other",)).restore(output))
            artifact = restored.entries[-1]["artifact"]
            path = evidence_assets.contained_path(root, artifact["path"])
            path.write_bytes(path.read_bytes() + b"corrupted")
            with self.assertRaises(ValueError):
                restored.restore(output)

    def test_bdf_restart_in_fresh_process(self):
        """A terminated producer continues the same sparse SciPy history."""

        script = """
import os, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csc_matrix
from copernican.lib.likelihoods.cmb.durable_history import (
    BDFCheckpoint, DurableHistories,
)
from copernican.lib.likelihoods.cmb.runtime.bdf_checkpoint import (
    integrate_bdf_checkpointed,
)
root, mode = Path(sys.argv[1]), sys.argv[2]
eta = np.linspace(0., 3., 101)
initial = np.array([1., 0.5])
matrix = csc_matrix([[-1., 0.1], [0., -4.]])
calls = [0]
def rhs(t, y):
    calls[0] += 1
    return matrix @ y + np.sin(t)
def jac(t, y):
    return matrix
class Stop(BDFCheckpoint):
    def due(self):
        return True
    def record(self, history, evaluated, state):
        if state['order'] >= 4 and state['factors'] is not None:
            super().record(history, evaluated, state)
            os._exit(17)
if mode == 'original':
    result = solve_ivp(rhs, (eta[0], eta[-1]), initial, method='BDF',
                       jac=jac, t_eval=eta, rtol=1e-7, atol=1e-9).y
else:
    store = DurableHistories(
        root, {'physical':'a'*64, 'numerical':'b'*64}, mode
    )
    checkpoint = (Stop(store, ('mode',), interval_seconds=0)
                  if mode == 'stop' else BDFCheckpoint(store, ('mode',)))
    result = integrate_bdf_checkpointed(
        rhs, jac, eta, initial, rtol=1e-7, atol=1e-9, checkpoint=checkpoint,
        rhs_calls=lambda: calls[0],
        restore_rhs_calls=lambda n: calls.__setitem__(0,n),
        initial_rhs_calls=0)
np.save(root / (mode + '.npy'), result)
(root / (mode + '.count')).write_text(str(calls[0]))
"""
        with tempfile.TemporaryDirectory() as root:
            for mode, expected in (
                ("stop", 17),
                ("resume", 0),
                ("original", 0),
            ):
                completed = subprocess.run(  # nosec B603 - fixed script.
                    [sys.executable, "-c", script, root, mode],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                self.assertEqual(
                    completed.returncode, expected, completed.stderr
                )
            root = Path(root)
            numpy.testing.assert_array_equal(
                numpy.load(root / "resume.npy"),
                numpy.load(root / "original.npy"),
            )
            self.assertEqual(
                (root / "resume.count").read_text(),
                (root / "original.count").read_text(),
            )


if __name__ == "__main__":
    unittest.main()
