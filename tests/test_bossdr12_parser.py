# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

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

import numpy as np

import copernican_lib.engine_plugin_validation as engine_plugin_validation
import copernican_lib.model_coder as model_coder
import copernican_lib.model_spec_validator as model_spec_validator
from copernican_lib.statistics import chi_squared_bao


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

        # Build a validated ΛCDM plugin used for BAO predictions.
        models_dir = base / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)
        engine_plugin_validation.validate_plugin(cls.plugin)

    def test_dataframe_shape_and_covariance(self):
        """Return nine observables with a 9x9 inverse covariance."""
        df = self.parser.parse_boss_dr12(str(self.data_dir))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 9)
        cov_inv = df.attrs.get("covariance_matrix_inv")
        self.assertIsNotNone(cov_inv)
        self.assertEqual(cov_inv.shape, (9, 9))

    def test_observable_values_match_release(self):
        """Check parsed values against the published BOSS DR12 results."""
        df = self.parser.parse_boss_dr12(str(self.data_dir))
        self.assertIsNotNone(df)
        expected = {
            "DM_over_rs": {0.38: 10.234064, 0.51: 13.365949, 0.61: 15.608878},
            "DH_over_rs": {0.38: 24.980578, 0.51: 22.316563, 0.61: 20.498625},
            "DV_over_rs": {0.38: 9.980710, 0.51: 12.668700, 0.61: 14.496600},
        }
        for obs, mapping in expected.items():
            for z, val in mapping.items():
                mask = (df["redshift"] == z) & (df["observable_type"] == obs)
                self.assertAlmostEqual(df.loc[mask, "value"].item(), val, 6)

    def test_chi_squared_bao_residuals_small(self):
        """Residuals stay near zero for reasonable ΛCDM parameters."""
        df = self.parser.parse_boss_dr12(str(self.data_dir))
        params = (67.66, 0.31, 0.041, 5e-5, 3.044, 1090)
        rs = self.plugin.get_sound_horizon_rs_Mpc(*params)
        z = df["redshift"].to_numpy(dtype=float)
        obs_type = df["observable_type"].to_numpy()
        obs_val = df["value"].to_numpy(dtype=float)
        obs_err = df["error"].to_numpy(dtype=float)
        cov_inv = df.attrs.get("covariance_matrix_inv")
        chi2 = chi_squared_bao(
            z,
            obs_type,
            obs_val,
            obs_err,
            self.plugin,
            params,
            rs,
            covariance_matrix_inv=cov_inv,
        )
        self.assertLess(chi2, 10.0)

        pred = np.full_like(obs_val, np.nan, dtype=float)
        mask = obs_type == "DM_over_rs"
        if np.any(mask):
            pred[mask] = (
                self.plugin.get_comoving_distance_Mpc(z[mask], *params) / rs
            )
        mask = obs_type == "DH_over_rs"
        if np.any(mask):
            hz = self.plugin.get_Hz_per_Mpc(z[mask], *params)
            pred[mask] = (
                self.plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
                / hz
                / rs
            )
        mask = obs_type == "DV_over_rs"
        if np.any(mask):
            dv = self.plugin.get_DV_Mpc(z[mask], *params)
            pred[mask] = dv / rs

        resid = obs_val - pred
        self.assertLess(np.max(np.abs(resid)), 1.0)

    def test_missing_covariance_files(self):
        """Dropping a covariance file triggers graceful error handling."""
        # Remove the dM/Hz covariance and expect ``None``.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.data_dir, tmp, dirs_exist_ok=True)
            os.remove(os.path.join(tmp, "BAO_consensus_covtot_dM_Hz.txt"))
            with self.assertLogs(level="ERROR") as cm:
                self.assertIsNone(self.parser.parse_boss_dr12(tmp))
            self.assertIn("dM/Hz covariance", "".join(cm.output))

        # Repeat for the D_V/F_AP covariance matrix.
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copytree(self.data_dir, tmp, dirs_exist_ok=True)
            os.remove(os.path.join(tmp, "BAO_consensus_covtot_dV_FAP.txt"))
            with self.assertLogs(level="ERROR") as cm:
                self.assertIsNone(self.parser.parse_boss_dr12(tmp))
            self.assertIn("D_V/F_AP covariance", "".join(cm.output))


if __name__ == "__main__":
    unittest.main()
