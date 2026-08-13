"""Tests for canonical CMB output metadata and observation surfaces."""

import unittest

import numpy
import pandas

from copernican.lib import cmb_output


class CMBOutputTestCase(unittest.TestCase):
    """Exercise exact spectrum naming and row-preserving block assembly."""

    def test_public_symbols_are_exposed(self) -> None:
        """The canonical CMB output API must remain importable."""

        self.assertIsInstance(cmb_output.CMBObservationBlock, type)
        self.assertIsInstance(cmb_output.CMBSpectrumMetadata, type)
        self.assertTrue(callable(cmb_output.CMBSpectrumMetadata.as_mapping))
        self.assertTrue(callable(cmb_output.assemble_cmb_theory_vector))
        self.assertTrue(callable(cmb_output.canonical_cmb_spectrum_name))
        self.assertTrue(callable(cmb_output.cmb_observation_blocks))
        self.assertTrue(callable(cmb_output.cmb_theory_values_for_block))
        self.assertTrue(callable(cmb_output.compose_cmb_spectrum_name))
        self.assertTrue(callable(cmb_output.describe_cmb_spectrum))
        self.assertTrue(callable(cmb_output.observed_cmb_spectrum_names))
        self.assertTrue(callable(cmb_output.split_cmb_spectrum_name))

    def test_spectrum_metadata_keeps_physical_surfaces_separate(self) -> None:
        """Sector, lensing, and diagnostic metadata must remain distinct."""

        scalar = cmb_output.describe_cmb_spectrum("scalar_tt")
        tensor = cmb_output.describe_cmb_spectrum("lensed_tensor_bb")
        lensing = cmb_output.describe_cmb_spectrum("phiphi")
        diagnostic = cmb_output.describe_cmb_spectrum("diagnostic_constraint")

        self.assertEqual(scalar.canonical_name, "scalar_TT")
        self.assertEqual(scalar.sector, "scalar")
        self.assertEqual(scalar.lensing_state, "unlensed")
        self.assertEqual(tensor.canonical_name, "lensed_tensor_BB")
        self.assertEqual(tensor.sector, "tensor")
        self.assertEqual(tensor.lensing_state, "lensed")
        self.assertEqual(lensing.canonical_name, "PP")
        self.assertEqual(lensing.observable_family, "lensing")
        self.assertEqual(lensing.units, "dimensionless")
        self.assertEqual(diagnostic.observable_family, "diagnostic")

    def test_long_form_blocks_preserve_repeated_noncontiguous_rows(
        self,
    ) -> None:
        """Observation blocks must preserve table and covariance row order."""

        observations = pandas.DataFrame(
            {
                "ell": [30, 20, 30, 25, 20],
                "spectrum": ["TE", "TT", "TT", "TE", "TE"],
                "Dl_obs": [3.0, 1.0, 2.0, 4.0, 5.0],
            }
        )

        te_block, tt_block = cmb_output.cmb_observation_blocks(observations)

        self.assertEqual(te_block.metadata.canonical_name, "TE")
        numpy.testing.assert_array_equal(te_block.row_indices, [0, 3, 4])
        numpy.testing.assert_array_equal(te_block.ells, [30, 25, 20])
        self.assertEqual(tt_block.metadata.canonical_name, "TT")
        numpy.testing.assert_array_equal(tt_block.row_indices, [1, 2])
        numpy.testing.assert_array_equal(tt_block.ells, [20, 30])
        with self.assertRaises(ValueError):
            te_block.ells[0] = 2

    def test_theory_vectors_accept_full_and_compact_spectrum_blocks(
        self,
    ) -> None:
        """Theory assembly must support solver and post-processing shapes."""

        observations = pandas.DataFrame(
            {
                "ell": [30, 20, 30, 25, 20],
                "spectrum": ["TE", "TT", "TT", "TE", "TE"],
                "Dl_obs": numpy.zeros(5),
            }
        )
        blocks = cmb_output.cmb_observation_blocks(observations)
        theory = {
            "TE": numpy.asarray([10.0, -1.0, -1.0, 40.0, 50.0]),
            "TT": numpy.asarray([20.0, 30.0]),
        }

        vector = cmb_output.assemble_cmb_theory_vector(
            theory,
            blocks,
            total_row_count=len(observations),
        )

        numpy.testing.assert_array_equal(
            vector, [10.0, 20.0, 30.0, 40.0, 50.0]
        )

        with self.assertRaisesRegex(ValueError, "duplicate canonical"):
            cmb_output.assemble_cmb_theory_vector(
                {"PP": numpy.ones(5), "phiphi": numpy.ones(5)},
                blocks,
                total_row_count=len(observations),
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
