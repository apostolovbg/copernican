# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

import importlib.util
import unittest
from pathlib import Path

import numpy as np

from copernican_lib import engine_plugin_validation
from copernican_lib.statistics import chi_squared_bao


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
        cls.plugin = engine_plugin_validation.build_plugin(model_data, funcs)
        engine_plugin_validation.validate_plugin(cls.plugin)

    def test_covariance_changes_chi2(self):
        """Using the covariance matrix yields a distinct chi-squared value."""
        rs = 150.0
        redshifts_array = self.df["redshift"].to_numpy(dtype=float)
        observable_types_array = self.df["observable_type"].to_numpy()
        observable_values_array = self.df["value"].to_numpy(dtype=float)
        observable_errors_array = self.df["error"].to_numpy(dtype=float)
        cov_inv = self.df.attrs.get("covariance_matrix_inv")

        chi2_cov = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            (),
            rs,
            covariance_matrix_inv=cov_inv,
        )

        chi2_diag = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            (),
            rs,
        )

        pred = np.zeros(len(redshifts_array), dtype=float)
        mask = observable_types_array == "DH_over_rs"
        if np.any(mask):
            hz = self.plugin.get_Hz_per_Mpc(redshifts_array[mask])
            pred[mask] = 299792.458 / hz / rs
        resid = observable_values_array - pred

        chi2_cov_manual = float(resid @ cov_inv @ resid)
        chi2_diag_manual = float(
            np.sum((resid / observable_errors_array) ** 2)
        )

        self.assertAlmostEqual(chi2_cov, chi2_cov_manual)
        self.assertAlmostEqual(chi2_diag, chi2_diag_manual)
        self.assertNotEqual(chi2_cov, chi2_diag)


if __name__ == "__main__":
    unittest.main()
