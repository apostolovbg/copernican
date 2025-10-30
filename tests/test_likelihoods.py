"""Unit tests for likelihood helper classes.

**Last Updated:** 2025-02-14
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

import copernican_lib.data_loaders as data_loaders
import copernican_lib.engine_interface as engine_interface
import copernican_lib.likelihoods as likelihoods
import copernican_lib.model_coder as model_coder
import copernican_lib.model_parser as model_parser


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
        cache_path = model_parser.parse_model(yaml_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_interface.build_plugin(parsed, funcs)
        engine_interface.validate_plugin(cls.plugin)

    def _prepare_sne(self):
        sne_df = data_loaders.load_sne_data("jla_2014").head(3).copy()
        if sne_df.attrs.get("covariance_matrix_inv") is not None:
            attrs = sne_df.attrs
            cov = attrs["covariance_matrix_inv"]
            attrs["covariance_matrix_inv"] = cov[:3, :3]
            diag = attrs["diag_errors_for_plot"]
            attrs["diag_errors_for_plot"] = diag[:3]
        return sne_df

    def _prepare_bao(self):
        bao_df = data_loaders.load_bao_data("compound_bao_set").head(3).copy()
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
            z=bao_df["redshift"].to_numpy(dtype=float),
            obs_type=bao_df["observable_type"].to_numpy(),
            obs_val=bao_df["value"].to_numpy(dtype=float),
            obs_err=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
            rs_override=rs_value,
        )
        self.assertTrue(np.isfinite(bao_like.loglike(params)))
        self.assertTrue(np.isfinite(bao_like.state["chi2"]))

        cmb_df = data_loaders.load_cmb_data("planck_2018_lite")
        cmb_like = likelihoods.CMBLike(cmb_df, self.plugin)
        self.assertTrue(np.isfinite(cmb_like.loglike(params)))
        self.assertTrue(np.isfinite(cmb_like.state["chi2"]))

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
            z=bao_df["redshift"].to_numpy(dtype=float),
            obs_type=bao_df["observable_type"].to_numpy(),
            obs_val=bao_df["value"].to_numpy(dtype=float),
            obs_err=bao_df["error"].to_numpy(dtype=float),
            model_plugin=self.plugin,
            covariance_matrix_inv=bao_df.attrs.get("covariance_matrix_inv"),
            rs_override=rs_value,
        )
        cmb_like = likelihoods.CMBLike(
            data_loaders.load_cmb_data("planck_2018_lite"),
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
            z=bao_df["redshift"].to_numpy(dtype=float),
            obs_type=bao_df["observable_type"].to_numpy(),
            obs_val=bao_df["value"].to_numpy(dtype=float),
            obs_err=bao_df["error"].to_numpy(dtype=float),
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


if __name__ == "__main__":
    unittest.main()
