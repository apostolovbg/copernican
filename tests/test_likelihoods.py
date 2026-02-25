"""Unit tests for likelihood helper classes.

"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

import copernican_lib.dataset_registry as dataset_registry
import copernican_lib.engine_plugin_validation as engine_plugin_validation
import copernican_lib.likelihoods as likelihoods
import copernican_lib.model_coder as model_coder
import copernican_lib.model_spec_validator as model_spec_validator


class LikelihoodTestCase(unittest.TestCase):
    """Validate the standalone likelihood helpers."""

    @classmethod
    def setUpClass(cls):
        """Load a validated ΛCDM plugin for likelihood evaluation."""

        repo_root = Path(__file__).resolve().parents[1]
        os.environ.setdefault("VIRTUAL_ENV", str(repo_root / ".venv"))

        models_dir = repo_root / "models"
        yaml_path = models_dir / "cosmo_model_lcdm.yml"
        cache_dir = models_dir / "cache"
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path, cache_dir
        )
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_plugin_validation.build_plugin(parsed, funcs)
        engine_plugin_validation.validate_plugin(cls.plugin)

    def _prepare_sne(self):
        sne_df = dataset_registry.load_sne_data("jla_2014").head(3).copy()
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            cov = attrs["covariance_matrix_inv"]
            attrs["covariance_matrix_inv"] = cov[:3, :3]
            diag = attrs["diag_errors_for_plot"]
            attrs["diag_errors_for_plot"] = diag[:3]
        return sne_df

    def _prepare_bao(self):
        bao_df = dataset_registry.load_bao_data("boss_dr12_bao").head(3).copy()
        cov_inv = bao_df.attrs.get("covariance_matrix_inv")
        if cov_inv is not None:
            bao_df.attrs["covariance_matrix_inv"] = cov_inv[:3, :3]
        return bao_df

    def test_component_loglikes_are_finite(self):
        """SNe, BAO and CMB helpers should produce finite log-likelihoods."""

        params = self.plugin.INITIAL_GUESSES

        sne_df = self._prepare_sne()
        sne_like = likelihoods.SNeLike(
            self.plugin.distance_modulus_model,
            sne_df,
        )
        self.assertTrue(np.isfinite(sne_like.loglike(params)))
        self.assertTrue(np.isfinite(sne_like.state["chi2"]))

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
        self.assertTrue(np.isfinite(bao_like.loglike(params)))
        self.assertTrue(np.isfinite(bao_like.state["chi2"]))

        cmb_df = dataset_registry.load_cmb_data("planck_2018_lite")
        cmb_like = likelihoods.CMBLike(cmb_df, self.plugin)
        self.assertTrue(np.isfinite(cmb_like.loglike(params)))
        self.assertTrue(np.isfinite(cmb_like.state["chi2"]))

    def test_bao_loglike_falls_back_without_camb(self):
        """BAO helper should reuse model distance functions when CAMB fails."""

        class TrackingPlugin:
            """Proxy raising ``get_camb_params`` and tracking fallbacks."""

            def __init__(self, base_plugin):
                self._base = base_plugin
                self.calls = {"dm": 0, "hz": 0, "dv": 0, "da": 0, "rs": 0}

            def __getattr__(self, name):
                return getattr(self._base, name)

            def get_camb_params(self, *_args, **_kwargs):
                raise RuntimeError("CAMB unavailable for test fallback path")

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

        params = self.plugin.INITIAL_GUESSES
        bao_df = self._prepare_bao()
        fallback_plugin = TrackingPlugin(self.plugin)
        bao_like = likelihoods.BAOLike(
            redshifts=bao_df["redshift"].to_numpy(dtype=float),
            observable_types=bao_df["observable_type"].to_numpy(),
            observable_values=bao_df["value"].to_numpy(dtype=float),
            observable_errors=bao_df["error"].to_numpy(dtype=float),
            model_plugin=fallback_plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
        )

        loglike = bao_like.loglike(params)

        self.assertTrue(np.isfinite(loglike))
        self.assertGreater(fallback_plugin.calls["dm"], 0)
        self.assertGreater(fallback_plugin.calls["hz"], 0)
        self.assertGreater(fallback_plugin.calls["rs"], 0)

    def test_bao_loglike_rejects_divergent_sound_horizon(self):
        """Divergent sound-horizon integrals must abort BAO predictions."""

        divergent_helper = model_coder._SoundHorizonFromExpression(
            lambda *full_params: model_coder.robust_quad(
                lambda z_val: 1.0 / (1.0 + z_val),
                full_params[-1],
                np.inf,
            )[0]
        )

        class DivergentSoundHorizonPlugin:
            """Proxy injecting a divergent ``rs_expression`` for regression."""

            def __init__(self, base_plugin):
                self._base = base_plugin

            def __getattr__(self, name):
                return getattr(self._base, name)

            def get_camb_params(self, *_args, **_kwargs):
                raise RuntimeError("CAMB disabled to exercise fallback path")

            def get_sound_horizon_rs_Mpc(self, *params):
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
        self.assertTrue(np.isfinite(joint_loglike))

        component_states = joint.state["metadata"]["components"]
        component_sum = sum(
            state["loglike"] for state in component_states.values()
        )
        chi2_sum = sum(state["chi2"] for state in component_states.values())

        self.assertAlmostEqual(joint_loglike, component_sum, places=8)
        self.assertAlmostEqual(joint.state["chi2"], chi2_sum, places=8)

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

        self.assertTrue(np.isfinite(loglike))
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


if __name__ == "__main__":
    unittest.main()
