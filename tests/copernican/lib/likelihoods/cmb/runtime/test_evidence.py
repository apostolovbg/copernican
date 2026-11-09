"""Direct tests for stable CCMBS evidence identities."""

import unittest

import numpy

from copernican.lib.likelihoods.cmb.runtime import evidence


class EvidenceTestCase(unittest.TestCase):
    """Exercise evidence digests independently from solver execution."""

    def test_capture_observes_actual_arrays_and_restores_nested_context(self):
        rows = []
        array = numpy.arange(4.0)
        self.assertFalse(evidence.has_completed_cmb_product_capture())
        evidence.publish_completed_cmb_products("ignored", {"eta": array})
        with evidence.capture_completed_cmb_products(
            lambda kind, payload: rows.append((kind, payload))
        ):
            self.assertTrue(evidence.has_completed_cmb_product_capture())
            evidence.publish_completed_cmb_products("native", {"eta": array})
            self.assertIs(rows[0][1]["eta"], array)
            self.assertEqual(rows[0][0], "native")
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with evidence.capture_completed_cmb_products(lambda *_: None):
                    raise RuntimeError("injected cleanup")
            evidence.publish_completed_cmb_products("restored", {"eta": array})
        self.assertEqual([kind for kind, _ in rows], ["native", "restored"])
        self.assertFalse(evidence.has_completed_cmb_product_capture())

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
