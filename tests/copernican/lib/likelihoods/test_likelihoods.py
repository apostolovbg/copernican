"""Unit tests for likelihood helper classes."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy
import pandas

import copernican.lib.dataset_registry as dataset_registry
import copernican.lib.likelihoods as likelihoods
import copernican.lib.model_adapter as model_plugin_validation
import copernican.lib.model_coder as model_coder
import copernican.lib.model_spec_validator as model_spec_validator


class TestImportModule(unittest.TestCase):
    """Exercise the module import path."""

    def test_import_module(self) -> None:
        self.assertEqual(
            likelihoods.__name__,
            "copernican.lib.likelihoods",
        )


class LikelihoodTestCase(unittest.TestCase):
    """Validate the standalone likelihood helpers."""

    @classmethod
    def setUpClass(cls):
        """Load a validated reference plugin for likelihood evaluation."""

        repo_root = Path(__file__).resolve().parents[4]
        os.environ.setdefault("VIRTUAL_ENV", str(repo_root / ".venv"))

        models_dir = repo_root / "copernican" / "models"
        yaml_path = models_dir / "model_lcdm.yml"
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_path = model_spec_validator.validate_and_cache_model(
                yaml_path,
                cache_dir,
            )
            funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = model_plugin_validation.build_plugin(parsed, funcs)
        model_plugin_validation.validate_plugin(cls.plugin)

    def _prepare_sne(self):
        sne_df = dataset_registry.load_sne_data("jla_2014").head(3).copy()
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            cov = attrs["covariance_matrix_inv"]
            attrs["covariance_matrix_inv"] = cov[:3, :3]
            diag = attrs["diag_errors_for_plot"]
            attrs["diag_errors_for_plot"] = diag[:3]
        return sne_df

    @staticmethod
    def _static_sne_model(z_values, *params):
        baseline = numpy.array([10.0, 11.0, 12.0], dtype=float)
        return baseline[: z_values.shape[0]].copy()

    def _make_intercept_sne_df(
        self,
        residuals: numpy.ndarray,
        *,
        covariance_matrix_inv: numpy.ndarray | None = None,
        diag_errors: numpy.ndarray | None = None,
        requires_intercept: bool = False,
    ) -> pandas.DataFrame:
        z_values = numpy.array([0.1, 0.2, 0.3], dtype=float)
        mu_model = self._static_sne_model(z_values)
        observations_df = pandas.DataFrame(
            {
                "zcmb": z_values,
                "mu_obs": mu_model + numpy.asarray(residuals, dtype=float),
                "e_mu_obs": (
                    numpy.asarray(diag_errors, dtype=float)
                    if diag_errors is not None
                    else numpy.full(z_values.shape, 0.1, dtype=float)
                ),
            }
        )
        if covariance_matrix_inv is not None:
            observations_df.attrs["covariance_matrix_inv"] = (
                covariance_matrix_inv
            )
        if requires_intercept:
            observations_df.attrs["requires_sne_intercept_marginalization"] = (
                True
            )
            observations_df.attrs["sne_intercept_name"] = "Delta_mu"
        return observations_df

    def _prepare_bao(self):
        bao_df = dataset_registry.load_bao_data("boss_dr12_bao").head(3).copy()
        cov_inv = bao_df.attrs.get("covariance_matrix_inv")
        if cov_inv is not None:
            bao_df.attrs["covariance_matrix_inv"] = cov_inv[:3, :3]
        return bao_df

    def test_joint_like_prepares_enabled_worker_runtime(self):
        """Worker preparation should run once for each enabled component."""

        enabled_prepare = mock.Mock()
        disabled_prepare = mock.Mock()
        enabled = mock.Mock(
            enabled=True,
            prepare_worker_runtime=enabled_prepare,
        )
        disabled = mock.Mock(
            enabled=False,
            prepare_worker_runtime=disabled_prepare,
        )
        joint = likelihoods.JointLike(
            {"enabled": enabled, "disabled": disabled}
        )

        joint.prepare_worker_runtime()
        enabled_prepare.assert_called_once_with()
        disabled_prepare.assert_not_called()

    def test_component_loglikes_are_finite(self):
        """SNe, BAO and CMB helpers should produce finite log-likelihoods."""

        params = self.plugin.INITIAL_GUESSES

        sne_df = self._prepare_sne()
        sne_like = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            sne_df,
        )
        self.assertTrue(numpy.isfinite(sne_like.loglike(params)))
        self.assertTrue(numpy.isfinite(sne_like.state["chi2"]))

        bao_df = self._prepare_bao()
        rs_value = self.plugin.get_sound_horizon_rs_Mpc(*params)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
            rs_override=rs_value,
        )
        self.assertTrue(numpy.isfinite(bao_like.loglike(params)))
        self.assertTrue(numpy.isfinite(bao_like.state["chi2"]))

        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        cmb_like = likelihoods.CMBLike(cmb_df, self.plugin)
        self.assertTrue(numpy.isfinite(cmb_like.loglike(params)))
        self.assertTrue(numpy.isfinite(cmb_like.state["chi2"]))

    def test_bao_loglike_uses_model_background_helpers(self):
        """BAO evaluation must not invoke the CMB runtime."""

        class TrackingPlugin:
            """Proxy rejecting CMB access and tracking distance helpers."""

            def __init__(self, base_plugin):
                self._base = base_plugin
                self.calls = {
                    "dm": 0,
                    "hz": 0,
                    "dv": 0,
                    "da": 0,
                    "rs": 0,
                    "rs_drag": 0,
                }

            def __getattr__(self, name):
                return getattr(self._base, name)

            def get_cmb_declared_runtime(self, *_args, **_kwargs):
                raise AssertionError("BAO must not query the CMB runtime")

            def get_comoving_distance_Mpc(self, *args, **kwargs):
                self.calls["dm"] += 1
                return self._base.get_comoving_distance_Mpc(*args, **kwargs)

            def get_Hz_per_Mpc(self, *args, **kwargs):
                self.calls["hz"] += 1
                return self._base.get_Hz_per_Mpc(*args, **kwargs)

            def get_DV_Mpc(self, *args, **kwargs):
                self.calls["dv"] += 1
                return self._base.get_DV_Mpc(*args, **kwargs)

            def get_angular_diameter_distance_Mpc(self, *args, **kwargs):
                self.calls["da"] += 1
                return self._base.get_angular_diameter_distance_Mpc(
                    *args, **kwargs
                )

            def get_sound_horizon_rs_Mpc(self, *args, **kwargs):
                self.calls["rs"] += 1
                return self._base.get_sound_horizon_rs_Mpc(*args, **kwargs)

            def get_sound_horizon_rs_drag_Mpc(self, *args, **kwargs):
                self.calls["rs_drag"] += 1
                return self._base.get_sound_horizon_rs_drag_Mpc(
                    *args, **kwargs
                )

        params = self.plugin.INITIAL_GUESSES
        bao_df = self._prepare_bao()
        tracking_plugin = TrackingPlugin(self.plugin)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=tracking_plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
        )

        loglike = bao_like.loglike(params)

        self.assertTrue(numpy.isfinite(loglike))
        self.assertGreater(tracking_plugin.calls["dm"], 0)
        self.assertGreater(tracking_plugin.calls["hz"], 0)
        self.assertGreater(tracking_plugin.calls["rs_drag"], 0)
        self.assertEqual(tracking_plugin.calls["rs"], 0)

    def test_bao_fixed_background_survives_cmb_solver_failure(self):
        """A fixed BAO background remains evaluable without CCMBS."""

        params = self.plugin.INITIAL_GUESSES
        bao_df = self._prepare_bao()
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
        )
        baseline = bao_like.loglike(params)
        with mock.patch(
            "copernican.lib.likelihoods.cmb.cmb."
            "compute_cmb_spectrum_from_contract",
            side_effect=AssertionError("CMB solver must not run for BAO"),
        ):
            isolated = bao_like.loglike(params)

        self.assertTrue(numpy.isfinite(baseline))
        self.assertEqual(isolated, baseline)
        self.assertTrue(numpy.isfinite(bao_like.state["chi2"]))

    def test_bao_loglike_rejects_divergent_sound_horizon(self):
        """Divergent sound-horizon integrals must abort BAO predictions."""

        divergent_helper = model_coder._SoundHorizonFromExpression(
            lambda *full_params: model_coder.robust_quad(
                lambda z_val: 1.0 / (1.0 + z_val),
                full_params[-1],
                numpy.inf,
            )[0]
        )

        class DivergentSoundHorizonPlugin:
            """Proxy injecting a divergent ``rs_expression`` for regression."""

            def __init__(self, base_plugin):
                self._base = base_plugin

            def __getattr__(self, name):
                return getattr(self._base, name)

            def get_cmb_declared_runtime(self, *_args, **_kwargs):
                raise AssertionError("BAO must not query the CMB runtime")

            def get_sound_horizon_rs_Mpc(self, *params):
                return self._base.get_sound_horizon_rs_Mpc(*params)

            def get_sound_horizon_rs_drag_Mpc(self, *params):
                return divergent_helper(*params)

        params = self.plugin.INITIAL_GUESSES
        bao_df = self._prepare_bao()
        divergent_plugin = DivergentSoundHorizonPlugin(self.plugin)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=divergent_plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
        )

        loglike = bao_like.loglike(params)

        self.assertEqual(loglike, float("-inf"))
        self.assertIn("error", bao_like.state["metadata"])

    def test_joint_loglike_matches_component_sum(self):
        """The joint likelihood should equal the sum of its components."""

        params = self.plugin.INITIAL_GUESSES
        sne_like = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            self._prepare_sne(),
        )
        bao_df = self._prepare_bao()
        rs_value = self.plugin.get_sound_horizon_rs_Mpc(*params)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
            rs_override=rs_value,
        )
        cmb_like = likelihoods.CMBLike(
            dataset_registry.load_cmb_data("planck_2018_lite"),
            self.plugin,
        )

        joint = likelihoods.JointLike(
            {"sne": sne_like, "bao": bao_like, "cmb": cmb_like},
            config={"sne": True, "bao": True, "cmb": True},
        )
        joint_loglike = joint.loglike(params)
        self.assertTrue(numpy.isfinite(joint_loglike))

        component_states = joint.state["metadata"]["components"]
        component_sum = sum(
            state["loglike"] for state in component_states.values()
        )
        chi2_sum = sum(state["chi2"] for state in component_states.values())

        self.assertAlmostEqual(joint_loglike, component_sum, places=8)
        self.assertAlmostEqual(joint.state["chi2"], chi2_sum, places=8)

    def test_joint_loglike_batch_uses_batch_capable_cmb_component(self):
        """Joint batches preserve order and combine non-CMB components."""

        sne_like = mock.Mock(enabled=True)
        sne_like.loglike.side_effect = (-1.0, -2.0)
        cmb_like = mock.Mock(enabled=True)
        cmb_like.loglike_batch.return_value = (-3.0, -4.0)
        joint = likelihoods.JointLike(
            {"sne": sne_like, "cmb": cmb_like},
            config={"sne": True, "cmb": True},
        )

        values = joint.loglike_batch(((1.0,), (2.0,)))

        self.assertEqual(values, (-4.0, -6.0))
        cmb_like.loglike_batch.assert_called_once_with([(1.0,), (2.0,)])

    def test_joint_like_respects_toggles(self):
        """Disabled datasets contribute zero log-likelihood and χ²."""

        params = self.plugin.INITIAL_GUESSES
        sne_like = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            self._prepare_sne(),
        )
        bao_df = self._prepare_bao()
        rs_value = self.plugin.get_sound_horizon_rs_Mpc(*params)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
            rs_override=rs_value,
        )

        joint = likelihoods.JointLike(
            {"sne": sne_like, "bao": bao_like},
            config={"sne": True, "bao": False},
        )
        loglike = joint.loglike(params)
        state = joint.state["metadata"]["components"]

        self.assertTrue(numpy.isfinite(loglike))
        self.assertAlmostEqual(loglike, state["sne"]["loglike"], places=8)
        self.assertAlmostEqual(state["bao"]["chi2"], 0.0, places=8)
        self.assertFalse(state["bao"]["metadata"]["enabled"])

    def test_likelihoods_reuse_cached_inputs(self):
        """Mutating sources after construction keeps outputs stable."""

        params = self.plugin.INITIAL_GUESSES

        sne_df = self._prepare_sne()
        sne_like = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            sne_df,
        )
        sne_baseline = sne_like.loglike(params)
        # Mutate the DataFrame in place; cached arrays must shield the helper.
        sne_df.loc[:, "mu_obs"] = 0.0
        sne_df.loc[:, "zcmb"] = 0.0
        self.assertAlmostEqual(sne_like.loglike(params), sne_baseline)
        sne_mutated = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            sne_df,
        )
        self.assertNotAlmostEqual(
            sne_mutated.loglike(params),
            sne_baseline,
        )

        bao_df = self._prepare_bao()
        redshifts_array = bao_df["redshift"].to_numpy(dtype=float)
        observable_types_array = bao_df["observable_type"].to_numpy()
        observable_values_array = bao_df["value"].to_numpy(dtype=float)
        observable_errors_array = bao_df["error"].to_numpy(dtype=float)
        cov = bao_df.attrs.get("covariance_matrix_inv")
        rs_value = self.plugin.get_sound_horizon_rs_Mpc(*params)
        bao_like = likelihoods.BAOLike(
            redshifts=redshifts_array,
            observable_types=observable_types_array,
            observable_values=observable_values_array,
            observable_errors=observable_errors_array,
            model_plugin=self.plugin,
            covariance_matrix_inv=cov,
            rs_override=rs_value,
        )
        bao_baseline = bao_like.loglike(params)
        redshifts_array[:] = 0.0
        observable_values_array[:] = observable_values_array + 50.0
        self.assertAlmostEqual(bao_like.loglike(params), bao_baseline)
        bao_mutated = likelihoods.BAOLike(
            redshifts=redshifts_array,
            observable_types=observable_types_array,
            observable_values=observable_values_array,
            observable_errors=observable_errors_array,
            model_plugin=self.plugin,
            covariance_matrix_inv=cov,
            rs_override=rs_value,
        )
        self.assertNotAlmostEqual(
            bao_mutated.loglike(params),
            bao_baseline,
        )

        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite").copy()
        cmb_like = likelihoods.CMBLike(cmb_df, self.plugin)
        cmb_baseline = cmb_like.loglike(params)
        cmb_df.loc[:, "Dl_obs"] = 0.0
        self.assertAlmostEqual(cmb_like.loglike(params), cmb_baseline)
        cmb_mutated = dataset_registry.load_cmb_data("planck_2018_lite").copy()
        cmb_mutated.loc[:, "Dl_obs"] += 5.0
        cmb_like_mutated = likelihoods.CMBLike(cmb_mutated, self.plugin)
        self.assertNotAlmostEqual(
            cmb_like_mutated.loglike(params),
            cmb_baseline,
        )

    def test_sne_intercept_marginalization_for_full_covariance(self):
        """Union-style intercept shifts should be removed with full COV."""

        cov_inv = numpy.array(
            [
                [4.0, 0.2, 0.0],
                [0.2, 3.0, 0.1],
                [0.0, 0.1, 2.5],
            ],
            dtype=float,
        )
        baseline = self._make_intercept_sne_df(
            numpy.zeros(3, dtype=float),
            covariance_matrix_inv=cov_inv,
            requires_intercept=True,
        )
        shifted = self._make_intercept_sne_df(
            numpy.full(3, 1.25, dtype=float),
            covariance_matrix_inv=cov_inv,
            requires_intercept=True,
        )

        baseline_like = likelihoods.SNeLike(
            self._static_sne_model,
            baseline,
        )
        shifted_like = likelihoods.SNeLike(
            self._static_sne_model,
            shifted,
        )

        baseline_loglike = baseline_like.loglike(())
        shifted_loglike = shifted_like.loglike(())

        self.assertAlmostEqual(shifted_loglike, baseline_loglike, places=8)
        self.assertTrue(
            shifted_like.state["metadata"]["sne_intercept_marginalized"]
        )
        self.assertAlmostEqual(
            shifted_like.state["metadata"]["sne_intercept_delta_mu"],
            -1.25,
            places=8,
        )

    def test_sne_intercept_marginalization_for_diagonal_fallback(self):
        """Union-style intercept shifts should be removed with diagonal COV."""

        diag_errors = numpy.array([0.1, 0.2, 0.3], dtype=float)
        baseline = self._make_intercept_sne_df(
            numpy.zeros(3, dtype=float),
            diag_errors=diag_errors,
            requires_intercept=True,
        )
        shifted = self._make_intercept_sne_df(
            numpy.full(3, -0.75, dtype=float),
            diag_errors=diag_errors,
            requires_intercept=True,
        )

        baseline_like = likelihoods.SNeLike(
            self._static_sne_model,
            baseline,
        )
        shifted_like = likelihoods.SNeLike(
            self._static_sne_model,
            shifted,
        )

        baseline_loglike = baseline_like.loglike(())
        shifted_loglike = shifted_like.loglike(())

        self.assertAlmostEqual(shifted_loglike, baseline_loglike, places=8)
        self.assertTrue(
            shifted_like.state["metadata"]["sne_intercept_marginalized"]
        )
        self.assertAlmostEqual(
            shifted_like.state["metadata"]["sne_intercept_delta_mu"],
            0.75,
            places=8,
        )

    def test_sne_intercept_is_disabled_by_default(self):
        """Ordinary datasets should keep the raw residual convention."""

        raw_df = self._make_intercept_sne_df(
            numpy.full(3, 1.0, dtype=float),
            diag_errors=numpy.array([0.1, 0.1, 0.1], dtype=float),
            requires_intercept=False,
        )
        raw_like = likelihoods.SNeLike(self._static_sne_model, raw_df)

        loglike = raw_like.loglike(())

        self.assertTrue(numpy.isfinite(loglike))
        self.assertFalse(
            raw_like.state["metadata"]["sne_intercept_marginalized"]
        )
        self.assertGreater(raw_like.state["chi2"], 0.0)

    def test_sne_intercept_shaped_residuals_still_have_cost(self):
        """Non-constant residual structure should not collapse to zero."""

        shaped_df = self._make_intercept_sne_df(
            numpy.array([1.0, -1.0, 0.5], dtype=float),
            diag_errors=numpy.array([0.1, 0.2, 0.3], dtype=float),
            requires_intercept=True,
        )
        shaped_like = likelihoods.SNeLike(self._static_sne_model, shaped_df)

        loglike = shaped_like.loglike(())

        self.assertTrue(numpy.isfinite(loglike))
        self.assertTrue(
            shaped_like.state["metadata"]["sne_intercept_marginalized"]
        )
        self.assertGreater(shaped_like.state["chi2"], 0.0)


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the shared likelihood API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(likelihoods.BAOLike))
        self.assertTrue(callable(likelihoods.CMBLike))
        self.assertTrue(callable(likelihoods.SNeLike))
        self.assertTrue(callable(likelihoods.JointLike))
        self.assertTrue(hasattr(likelihoods, "LikelihoodProtocol"))
        self.assertTrue(hasattr(likelihoods, "LikelihoodState"))
        self.assertTrue(hasattr(likelihoods.LikelihoodState, "as_mapping"))


if __name__ == "__main__":
    unittest.main()
