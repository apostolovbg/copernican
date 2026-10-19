"""Direct tests for declared collision operators."""

import unittest
import warnings

import numpy

from copernican.lib.likelihoods.cmb.runtime import collisions
from copernican.lib.perturbation_contract import evaluate_compiled_expression
from tests.copernican.lib import test_perturbation_contract as contract_tests


class CollisionOperatorTestCase(unittest.TestCase):
    """Exercise stiff collision handling separately from mode evolution."""

    def test_scalar_thomson_action_conserves_momentum_and_damps_slip(self):
        """Use total momentum and exponential slip, independently of expm."""

        compiled = contract_tests.PerturbationContractTestCase()._compile(
            contract_tests._scalar_metadata_only_contract()
        )
        form = compiled.collision_operators["thomson_drag"].exact_form
        k_value, loading = 0.2, 2.25
        context = {
            "acoustic_k": k_value,
            "photon_baryon_momentum_ratio": loading,
        }
        matrix = numpy.asarray(
            [
                [evaluate_compiled_expression(entry, context) for entry in row]
                for row in form.compiled_matrix
            ]
        )
        initial = numpy.asarray((2.0, -0.5, 0.0, 0.0))
        photon_velocity = 3 * k_value * initial[0]
        momentum = loading * photon_velocity + initial[1]
        equilibrium = momentum / (1 + loading)
        for depth in (0.0, 0.5, 1e6):
            with self.subTest(depth=depth):
                result = collisions._exact_linear_collision_step(
                    operator_matrix=matrix, dt=depth, target_state=initial
                )
                expected_velocity = equilibrium + (
                    photon_velocity - equilibrium
                ) * numpy.exp(-(1 + loading) * depth)
                numpy.testing.assert_allclose(
                    result,
                    (
                        expected_velocity / (3 * k_value),
                        momentum - loading * expected_velocity,
                        0,
                        0,
                    ),
                    rtol=1e-12,
                    atol=1e-12,
                )
        for eigenmode, decay in (((1.0, 1.0), 0.3), ((-6.0, 1.0), 1.0)):
            initial = numpy.asarray((0.0, 0.0, *eigenmode))
            result = collisions._exact_linear_collision_step(
                operator_matrix=matrix, dt=0.5, target_state=initial
            )
            expected = initial * numpy.exp(-decay * 0.5)
            numpy.testing.assert_allclose(result, expected, rtol=1e-12)
            wrong_sign = collisions._exact_linear_collision_step(
                operator_matrix=-matrix, dt=0.5, target_state=initial
            )
            self.assertGreater(
                float(numpy.max(abs(wrong_sign - expected))), 0.1
            )

    def test_batched_collision_overflow_is_handled_without_runtime_warnings(
        self,
    ):
        """Rejected stiff collision rows must not flood worker stderr."""

        blocks = numpy.asarray(
            [[[1.0e3, 0.0], [0.0, -1.0e3]]],
            dtype=float,
        )
        states = numpy.asarray([[1.0, 1.0]], dtype=float)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always", RuntimeWarning)
            result = collisions._exact_batched_two_state_blocks(
                blocks,
                states,
            )

        self.assertIsNone(result)
        self.assertFalse(
            any(item.category is RuntimeWarning for item in captured)
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
