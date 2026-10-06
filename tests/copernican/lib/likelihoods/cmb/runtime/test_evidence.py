"""Direct tests for stable CCMBS evidence identities."""

import unittest

import numpy

from copernican.lib.likelihoods.cmb.runtime import evidence


class EvidenceTestCase(unittest.TestCase):
    """Exercise evidence digests independently from solver execution."""

    def test_projection_digest_binds_shape_and_finite_values(self):
        """A digest changes with products and rejects invalid evidence."""

        baseline = evidence._projection_array_digest((1.0, 2.0))
        self.assertNotEqual(
            baseline,
            evidence._projection_array_digest((1.0, 3.0)),
        )
        with self.assertRaisesRegex(ValueError, "finite and nonempty"):
            evidence._projection_array_digest(numpy.asarray((numpy.nan,)))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
