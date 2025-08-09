"""Validate the BOSS DR12 BAO parser.

This test confirms that the parser combines the two covariance matrices into
a single 9x9 inverse covariance matrix and produces one row per observable at
each of the three redshift bins. It also ensures that missing covariance files
are reported via a ``None`` return value.
"""

import importlib.util
import os
import shutil
import tempfile
import unittest
from pathlib import Path


class BossDR12ParserTestCase(unittest.TestCase):
    """Exercise ``parse_boss_dr12`` under normal and failure modes."""

    @classmethod
    def setUpClass(cls):
        """Import the parser module once for use across all test methods."""
        # Dynamically import the parser directly from the data directory. This
        # avoids mutating ``sys.path`` and keeps the tests self-contained.
        base = Path(__file__).resolve().parents[1]
        cls.data_dir = base / "data" / "bao" / "bossdr12"
        spec = importlib.util.spec_from_file_location(
            "cosmo_parser_bossdr12", cls.data_dir / "cosmo_parser_bossdr12.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Keep a reference to the imported module so its functions remain
        # unbound and callable without additional ``self`` arguments.
        cls.parser = module

    def test_dataframe_shape_and_covariance(self):
        """Return nine observables with a 9x9 inverse covariance."""
        df = self.parser.parse_boss_dr12(str(self.data_dir))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 9)
        cov_inv = df.attrs.get("covariance_matrix_inv")
        self.assertIsNotNone(cov_inv)
        self.assertEqual(cov_inv.shape, (9, 9))

    def test_missing_covariance_files(self):
        """Dropping a covariance file triggers graceful error handling."""
        # Remove the dM/Hz covariance and expect ``None``.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.data_dir, tmp, dirs_exist_ok=True)
            os.remove(os.path.join(tmp, "BAO_consensus_covtot_dM_Hz.txt"))
            self.assertIsNone(self.parser.parse_boss_dr12(tmp))

        # Repeat for the D_V/F_AP covariance matrix.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.data_dir, tmp, dirs_exist_ok=True)
            os.remove(os.path.join(tmp, "BAO_consensus_covtot_dV_FAP.txt"))
            self.assertIsNone(self.parser.parse_boss_dr12(tmp))


if __name__ == "__main__":
    unittest.main()
