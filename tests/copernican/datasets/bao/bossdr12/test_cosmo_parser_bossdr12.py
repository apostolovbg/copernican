"""Validate the BOSS DR12 BAO parser."""

import importlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as numpy_module

import copernican_lib.engine_adapter as engine_plugin_validation
import copernican_lib.model_coder as model_coder
import copernican_lib.model_spec_validator as model_spec_validator
from copernican_lib.statistics import chi_squared_bao


class BossDR12ParserTestCase(unittest.TestCase):
    """Exercise ``parse_boss_dr12`` under normal and failure modes."""

    @classmethod
    def setUpClass(cls):
        """Import the parser module once for use across all test methods."""
        base = Path(__file__).resolve().parents[5]
        cls.data_dir = base / "copernican" / "datasets" / "bao" / "bossdr12"
        cls.parser = importlib.import_module(
            "copernican.datasets.bao.bossdr12.cosmo_parser_bossdr12"
        )

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
        bossdr12_dataframe = self.parser.parse_boss_dr12(str(self.data_dir))
        self.assertIsNotNone(bossdr12_dataframe)
        self.assertEqual(len(bossdr12_dataframe), 9)
        cov_inv = bossdr12_dataframe.attrs.get("covariance_matrix_inv")
        self.assertIsNotNone(cov_inv)
        self.assertEqual(cov_inv.shape, (9, 9))

    def test_observable_values_match_release(self):
        """Check parsed values against the published BOSS DR12 results."""
        bossdr12_dataframe = self.parser.parse_boss_dr12(str(self.data_dir))
        self.assertIsNotNone(bossdr12_dataframe)
        expected = {
            "DM_over_rs": {0.38: 10.234064, 0.51: 13.365949, 0.61: 15.608878},
            "DH_over_rs": {0.38: 24.980578, 0.51: 22.316563, 0.61: 20.498625},
            "DV_over_rs": {0.38: 9.980710, 0.51: 12.668700, 0.61: 14.496600},
        }
        for obs, mapping in expected.items():
            for redshift, expected_value in mapping.items():
                row_mask = (bossdr12_dataframe["redshift"] == redshift) & (
                    bossdr12_dataframe["observable_type"] == obs
                )
                self.assertAlmostEqual(
                    bossdr12_dataframe.loc[row_mask, "value"].item(),
                    expected_value,
                    6,
                )

    def test_chi_squared_bao_residuals_small(self):
        """Residuals stay near zero for reasonable ΛCDM parameters."""
        bossdr12_dataframe = self.parser.parse_boss_dr12(str(self.data_dir))
        params = (67.66, 0.31, 0.041, 5e-5, 3.044, 1090)
        sound_horizon_mpc = self.plugin.get_sound_horizon_rs_Mpc(*params)
        redshifts_array = bossdr12_dataframe["redshift"].to_numpy(dtype=float)
        observable_types_array = bossdr12_dataframe[
            "observable_type"
        ].to_numpy()
        observable_values_array = bossdr12_dataframe["value"].to_numpy(
            dtype=float
        )
        observable_errors_array = bossdr12_dataframe["error"].to_numpy(
            dtype=float
        )
        cov_inv = bossdr12_dataframe.attrs.get("covariance_matrix_inv")
        chi2 = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            params,
            sound_horizon_mpc,
            covariance_matrix_inv=cov_inv,
        )
        self.assertLess(chi2, 10.0)

        predicted_values = numpy_module.full_like(
            observable_values_array,
            numpy_module.nan,
            dtype=float,
        )
        row_mask = observable_types_array == "DM_over_rs"
        if numpy_module.any(row_mask):
            predicted_values[row_mask] = (
                self.plugin.get_comoving_distance_Mpc(
                    redshifts_array[row_mask], *params
                )
                / sound_horizon_mpc
            )
        row_mask = observable_types_array == "DH_over_rs"
        if numpy_module.any(row_mask):
            hubble_rate = self.plugin.get_Hz_per_Mpc(
                redshifts_array[row_mask],
                *params,
            )
            predicted_values[row_mask] = (
                self.plugin.FIXED_PARAMS.get("C_LIGHT_KM_S", 299792.458)
                / hubble_rate
                / sound_horizon_mpc
            )
        row_mask = observable_types_array == "DV_over_rs"
        if numpy_module.any(row_mask):
            volume_distance = self.plugin.get_DV_Mpc(
                redshifts_array[row_mask],
                *params,
            )
            predicted_values[row_mask] = volume_distance / sound_horizon_mpc

        residual_values = observable_values_array - predicted_values
        self.assertLess(
            numpy_module.max(numpy_module.abs(residual_values)),
            1.0,
        )

    def test_missing_covariance_files(self):
        """Dropping a covariance file triggers graceful error handling."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            shutil.copytree(self.data_dir, temporary_dir, dirs_exist_ok=True)
            os.remove(
                os.path.join(temporary_dir, "BAO_consensus_covtot_dM_Hz.txt")
            )
            with self.assertLogs(level="ERROR") as log_capture:
                self.assertIsNone(self.parser.parse_boss_dr12(temporary_dir))
            self.assertIn("dM/Hz covariance", "".join(log_capture.output))

        with tempfile.TemporaryDirectory() as temporary_dir:
            shutil.copytree(self.data_dir, temporary_dir, dirs_exist_ok=True)
            os.remove(
                os.path.join(temporary_dir, "BAO_consensus_covtot_dV_FAP.txt")
            )
            with self.assertLogs(level="ERROR") as log_capture:
                self.assertIsNone(self.parser.parse_boss_dr12(temporary_dir))
            self.assertIn("D_V/F_AP covariance", "".join(log_capture.output))


if __name__ == "__main__":
    unittest.main()
