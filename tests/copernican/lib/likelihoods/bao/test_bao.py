"""Unit tests for BAO helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import numpy as numpy_module

import copernican.lib.model_adapter as model_plugin_validation
from copernican.lib import dataset_registry
from copernican.lib.likelihoods import bao
from copernican.lib.likelihoods.cmb import cmb
from copernican.lib.statistics import chi_squared_bao


class BAOCovarianceTestCase(unittest.TestCase):
    """Ensure BAO chi-squared uses the covariance matrix when available."""

    @classmethod
    def setUpClass(cls) -> None:
        base = Path(__file__).resolve().parents[5]
        data_dir = base / "copernican" / "datasets" / "bao" / "bossdr12"
        spec = importlib.util.spec_from_file_location(
            "copernican.datasets.bao.bossdr12.dataset_parser_bossdr12",
            data_dir / "dataset_parser_bossdr12.py",
        )
        module = importlib.util.module_from_spec(spec)
        if spec.loader is None:
            raise RuntimeError("Unable to load BOSS DR12 parser module.")
        spec.loader.exec_module(module)
        cls.bao_dataframe = module.parse_boss_dr12(str(data_dir))

        model_data = {
            "model_name": "Dummy",
            "description": "desc",
            "abstract": "abs",
            "parameters": [],
            "equations": {"sne": [], "bao": []},
            "valid_for_cmb": False,
        }

        def _zero(z, *_):
            return numpy_module.zeros_like(z, dtype=float)

        def _huge(z, *_):
            return numpy_module.full_like(z, 1e18, dtype=float)

        funcs = {
            "distance_modulus_model": _zero,
            "get_comoving_distance_Mpc": _zero,
            "get_luminosity_distance_Mpc": _zero,
            "get_angular_diameter_distance_Mpc": _zero,
            "get_Hz_per_Mpc": _huge,
            "get_DV_Mpc": _zero,
            "get_sound_horizon_rs_Mpc": lambda *_: 150.0,
        }
        cls.plugin = model_plugin_validation.build_plugin(model_data, funcs)
        model_plugin_validation.validate_plugin(cls.plugin)

    def test_covariance_changes_chi2(self) -> None:
        """Using the covariance matrix yields a distinct chi-squared value."""
        sound_horizon_mpc = 150.0
        redshifts_array = self.bao_dataframe["redshift"].to_numpy(dtype=float)
        observable_types_array = self.bao_dataframe[
            "observable_type"
        ].to_numpy()
        observable_values_array = self.bao_dataframe["value"].to_numpy(
            dtype=float
        )
        observable_errors_array = self.bao_dataframe["error"].to_numpy(
            dtype=float
        )
        cov_inv = self.bao_dataframe.attrs.get("covariance_matrix_inv")

        chi2_cov = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            (),
            sound_horizon_mpc,
            covariance_matrix_inv=cov_inv,
        )

        chi2_diag = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            (),
            sound_horizon_mpc,
        )

        predicted_values = numpy_module.zeros(
            len(redshifts_array), dtype=float
        )
        mask = observable_types_array == "DH_over_rs"
        if numpy_module.any(mask):
            hubble_rate_mpc = self.plugin.get_Hz_per_Mpc(redshifts_array[mask])
            predicted_values[mask] = (
                299792.458 / hubble_rate_mpc / sound_horizon_mpc
            )
        residual_values = observable_values_array - predicted_values

        chi2_cov_manual = float(residual_values @ cov_inv @ residual_values)
        normalized = residual_values / observable_errors_array
        chi2_diag_manual = float(numpy_module.sum(normalized**2))

        self.assertAlmostEqual(chi2_cov, chi2_cov_manual)
        self.assertAlmostEqual(chi2_diag, chi2_diag_manual)
        self.assertNotEqual(chi2_cov, chi2_diag)

    def test_bao_is_unchanged_when_cmb_entrypoint_is_unavailable(self) -> None:
        """BAO evaluation must not depend on the CMB solver entrypoint."""

        sound_horizon_mpc = 150.0
        redshifts_array = self.bao_dataframe["redshift"].to_numpy(dtype=float)
        observable_types_array = self.bao_dataframe[
            "observable_type"
        ].to_numpy()
        observable_values_array = self.bao_dataframe["value"].to_numpy(
            dtype=float
        )
        observable_errors_array = self.bao_dataframe["error"].to_numpy(
            dtype=float
        )
        cov_inv = self.bao_dataframe.attrs.get("covariance_matrix_inv")
        baseline = chi_squared_bao(
            redshifts_array,
            observable_types_array,
            observable_values_array,
            observable_errors_array,
            self.plugin,
            (),
            sound_horizon_mpc,
            covariance_matrix_inv=cov_inv,
        )
        with mock.patch.object(
            cmb,
            "compute_cmb_spectrum_from_contract",
            side_effect=RuntimeError("CMB solver deliberately unavailable"),
        ):
            isolated = chi_squared_bao(
                redshifts_array,
                observable_types_array,
                observable_values_array,
                observable_errors_array,
                self.plugin,
                (),
                sound_horizon_mpc,
                covariance_matrix_inv=cov_inv,
            )
        self.assertEqual(isolated, baseline)

    def test_isolation_evidence_preserves_values_covariance_and_failures(self):
        """The final boundary records exact fixed-background BAO parity."""

        baseline = {
            "chi2": 12.5,
            "loglike": -6.25,
            "predictions": numpy_module.array([1.0, 2.0]),
            "covariance_matrix_inv": numpy_module.eye(2),
            "typed_failure": None,
        }
        evidence = bao.assess_bao_cmb_isolation(baseline, dict(baseline))
        self.assertTrue(evidence["available"])
        self.assertTrue(evidence["converged"])
        self.assertTrue(evidence["covariance_preserved"])
        self.assertTrue(evidence["typed_failures_preserved"])

    def test_isolation_evidence_rejects_changed_bao_output(self):
        """A changed fixed-background value is a scientific regression."""

        baseline = {"chi2": 12.5, "covariance": "full"}
        isolated = {"chi2": 12.6, "covariance": "full"}
        evidence = bao.assess_bao_cmb_isolation(baseline, isolated)
        self.assertFalse(evidence["converged"])
        self.assertFalse(evidence["values_preserved"])


class BAOPublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the BAO helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(bao.BAOLike))
        self.assertTrue(callable(chi_squared_bao))
        self.assertTrue(hasattr(dataset_registry, "load_bao_data"))

    def test_loglike_and_state_symbols_are_exposed(self) -> None:
        loglike = bao.BAOLike.loglike
        state = bao.BAOLike.state
        self.assertTrue(callable(loglike))
        self.assertTrue(hasattr(state, "__get__"))


if __name__ == "__main__":
    unittest.main()
