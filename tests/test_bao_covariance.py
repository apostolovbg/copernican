# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from copernican_lib import engine_interface
from engines.cosmo_engine_comb import chi_squared_bao


class BaoCovarianceTestCase(unittest.TestCase):
    """Ensure BAO chi-squared uses the covariance matrix when available."""

    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1]
        data_dir = base / "data" / "bao" / "bossdr12"
        spec = importlib.util.spec_from_file_location(
            "cosmo_parser_bossdr12", data_dir / "cosmo_parser_bossdr12.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        cls.df = module.parse_boss_dr12(str(data_dir))

        model_data = {
            "model_name": "Dummy",
            "description": "desc",
            "abstract": "abs",
            "parameters": [],
            "equations": {"sne": [], "bao": []},
        }

        def _zero(z, *_):
            return np.zeros_like(z, dtype=float)

        def _huge(z, *_):
            return np.full_like(z, 1e18, dtype=float)

        funcs = {
            "distance_modulus_model": _zero,
            "get_comoving_distance_Mpc": _zero,
            "get_luminosity_distance_Mpc": _zero,
            "get_angular_diameter_distance_Mpc": _zero,
            "get_Hz_per_Mpc": _huge,
            "get_DV_Mpc": _zero,
            "get_sound_horizon_rs_Mpc": lambda *_: 150.0,
        }
        cls.plugin = engine_interface.build_plugin(model_data, funcs)
        engine_interface.validate_plugin(cls.plugin)

    def test_covariance_changes_chi2(self):
        """Using the covariance matrix yields a distinct chi-squared value."""
        rs = 150.0
        z = self.df["redshift"].to_numpy(dtype=float)
        obs_type = self.df["observable_type"].to_numpy()
        obs_val = self.df["value"].to_numpy(dtype=float)
        obs_err = self.df["error"].to_numpy(dtype=float)
        cov_inv = self.df.attrs.get("covariance_matrix_inv")

        chi2_cov = chi_squared_bao(
            z,
            obs_type,
            obs_val,
            obs_err,
            self.plugin,
            (),
            rs,
            covariance_matrix_inv=cov_inv,
        )

        chi2_diag = chi_squared_bao(
            z,
            obs_type,
            obs_val,
            obs_err,
            self.plugin,
            (),
            rs,
        )

        pred = np.zeros(len(z), dtype=float)
        mask = obs_type == "DH_over_rs"
        if np.any(mask):
            hz = self.plugin.get_Hz_per_Mpc(z[mask])
            pred[mask] = 299792.458 / hz / rs
        resid = obs_val - pred

        chi2_cov_manual = float(resid @ cov_inv @ resid)
        chi2_diag_manual = float(np.sum((resid / obs_err) ** 2))

        self.assertAlmostEqual(chi2_cov, chi2_cov_manual)
        self.assertAlmostEqual(chi2_diag, chi2_diag_manual)
        self.assertNotEqual(chi2_cov, chi2_diag)


if __name__ == "__main__":
    unittest.main()
