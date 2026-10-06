"""Direct tests for declared collision operators."""

import unittest
import warnings

import numpy

from copernican.lib.likelihoods.cmb.runtime import collisions


class CollisionOperatorTestCase(unittest.TestCase):
    """Exercise stiff collision handling separately from mode evolution."""

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
