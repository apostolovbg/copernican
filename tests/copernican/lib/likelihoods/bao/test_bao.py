"""Unit tests for BAO helpers."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_loglike_prefers_drag_epoch_sound_horizon(self):
        """BAO ratios select the drag helper when a plugin supplies one."""

        def _ones(redshift, *_params):
            return numpy_module.ones_like(redshift, dtype=float)

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_ones,
            get_Hz_per_Mpc=_ones,
            get_DV_Mpc=_ones,
            get_angular_diameter_distance_Mpc=_ones,
            get_sound_horizon_rs_Mpc=lambda *_: 100.0,
            get_sound_horizon_rs_drag_Mpc=lambda *_: 200.0,
            get_bao_drag_redshift=lambda *_: 1020.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )
        self.assertEqual(like.loglike(()), 0.0)
        self.assertEqual(like.state["metadata"]["sound_horizon_epoch"], "drag")
        self.assertEqual(like.state["metadata"]["z_drag"], 1020.0)

    def test_drag_helper_is_authoritative_over_recombination_helper(self):
        """A broken recombination helper cannot poison a valid BAO ruler."""

        def _ones(redshift, *_params):
            return numpy_module.ones_like(redshift, dtype=float)

        legacy = mock.Mock(
            side_effect=AssertionError(
                "BAO must not evaluate the recombination helper"
            )
        )
        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_ones,
            get_Hz_per_Mpc=_ones,
            get_DV_Mpc=_ones,
            get_angular_diameter_distance_Mpc=_ones,
            get_sound_horizon_rs_Mpc=legacy,
            get_sound_horizon_rs_drag_Mpc=lambda *_: 200.0,
            get_bao_drag_redshift=lambda *_: 1020.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), 0.0)
        legacy.assert_not_called()
        self.assertEqual(
            like.state["metadata"]["sound_horizon_source"],
            "model_plugin.get_sound_horizon_rs_drag_Mpc",
        )

    def test_invalid_drag_helper_is_not_replaced_by_recombination(self):
        """Reject invalid drag instead of silently changing the epoch."""

        def _ones(redshift, *_params):
            return numpy_module.ones_like(redshift, dtype=float)

        legacy = mock.Mock(return_value=100.0)
        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_ones,
            get_Hz_per_Mpc=_ones,
            get_DV_Mpc=_ones,
            get_angular_diameter_distance_Mpc=_ones,
            get_sound_horizon_rs_Mpc=legacy,
            get_sound_horizon_rs_drag_Mpc=lambda *_: numpy_module.nan,
            get_bao_drag_redshift=lambda *_: 1020.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), float("-inf"))
        legacy.assert_not_called()
        self.assertIn("error", like.state["metadata"])
        self.assertEqual(
            like.state["metadata"]["failure_type"], "invalid_value"
        )
        self.assertEqual(like.state["metadata"]["failure_stage"], "drag")

    def test_signature_selects_no_parameter_helpers_without_retry(self):
        """No-parameter plugins work without masking helper TypeErrors."""

        def _distance(redshift):
            return numpy_module.ones_like(redshift, dtype=float)

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_distance,
            get_Hz_per_Mpc=_distance,
            get_DV_Mpc=_distance,
            get_angular_diameter_distance_Mpc=_distance,
            get_sound_horizon_rs_drag_Mpc=lambda: 200.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), 0.0)
        self.assertEqual(like.state["metadata"]["sound_horizon_epoch"], "drag")

    def test_helper_type_error_is_not_retried_with_wrong_signature(self):
        """A helper TypeError is an execution failure, not an arity probe."""

        calls = []

        def _broken_distance(redshift, *_params):
            calls.append(tuple(_params))
            raise TypeError("invalid distance mathematics")

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_broken_distance,
            get_Hz_per_Mpc=lambda redshift, *_params: numpy_module.ones_like(
                redshift, dtype=float
            ),
            get_sound_horizon_rs_drag_Mpc=lambda *_params: 200.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike((70.0,)), float("-inf"))
        self.assertEqual(calls, [(70.0,)])
        self.assertEqual(
            like.state["metadata"]["failure_type"], "invalid_background"
        )
        self.assertEqual(like.state["metadata"]["failure_stage"], "background")

    def test_invalid_drag_redshift_is_a_typed_bao_failure(self):
        """A canonical drag ruler requires a physical positive endpoint."""

        def _ones(redshift, *_params):
            return numpy_module.ones_like(redshift, dtype=float)

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_ones,
            get_Hz_per_Mpc=_ones,
            get_DV_Mpc=_ones,
            get_angular_diameter_distance_Mpc=_ones,
            get_sound_horizon_rs_drag_Mpc=lambda *_: 200.0,
            get_bao_drag_redshift=lambda *_: 0.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), float("-inf"))
        self.assertEqual(
            like.state["metadata"]["failure_type"], "invalid_value"
        )
        self.assertEqual(like.state["metadata"]["failure_stage"], "drag")

    def test_invalid_dataset_shape_is_a_typed_bao_failure(self):
        """Mismatched BAO input arrays fail before model execution."""

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=lambda redshift: redshift,
            get_Hz_per_Mpc=lambda redshift: redshift + 1.0,
            get_sound_horizon_rs_drag_Mpc=lambda: 200.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([]),
            observable_errors=numpy_module.array([]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), float("-inf"))
        self.assertEqual(
            like.state["metadata"]["failure_type"], "invalid_dataset"
        )
        self.assertEqual(like.state["metadata"]["failure_stage"], "setup")

    def test_nonfinite_background_is_a_typed_bao_failure(self):
        """Non-finite model backgrounds cannot produce BAO ratios."""

        def _nan_distance(redshift, *_params):
            return numpy_module.full_like(redshift, numpy_module.nan)

        def _ones(redshift, *_params):
            return numpy_module.ones_like(redshift, dtype=float)

        plugin = SimpleNamespace(
            get_comoving_distance_Mpc=_nan_distance,
            get_Hz_per_Mpc=_ones,
            get_sound_horizon_rs_drag_Mpc=lambda *_: 200.0,
            FIXED_PARAMS={"C_LIGHT_KM_S": 299792.458},
        )
        like = bao.BAOLike(
            redshifts=numpy_module.array([0.5]),
            observable_types=numpy_module.array(["DH_over_rs"]),
            observable_values=numpy_module.array([1498.96229]),
            observable_errors=numpy_module.array([1.0]),
            model_plugin=plugin,
        )

        self.assertEqual(like.loglike(()), float("-inf"))
        self.assertEqual(
            like.state["metadata"]["failure_type"], "nonfinite_background"
        )
        self.assertEqual(like.state["metadata"]["failure_stage"], "background")


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
