"""Integration tests for the ensemble MCMC engine.

**Last Updated:** 2025-11-10
"""

import logging
import math
import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import emcee
import numpy as np
import pandas as pd
import xarray as xr
from emcee import moves

from copernican_lib import (
    chain_io,
    engine_interface,
    model_coder,
    model_parser,
)
from copernican_lib import progress as progress_helpers
from copernican_lib.progress import (
    BatchProgressBar,
    StepProgressEmitter,
    configure_sampler_progress_reporting,
)
from copernican_lib.utils import set_random_seed
from engines import cosmo_engine_mcmc
from engines.cosmo_engine_mcmc import (
    _ActiveLogProbability,
    _build_sne_logposterior,
    _classify_parameter_bounds,
    _estimate_condition_number,
    _initialise_active_walkers,
    _reseed_invalid_walkers,
)


def _build_model_plugin(yaml_filename: str):
    """Return a validated plugin for ``yaml_filename``.

    Tests construct plugins from disk instead of hard-coding dummy classes so
    that they exercise the same parsing pathway as the production workflow.
    """

    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    yaml_path = os.path.join(models_dir, yaml_filename)
    cache_dir = os.path.join(models_dir, "cache")
    cache_path = model_parser.parse_model(yaml_path, cache_dir)
    func_dict, parsed = model_coder.generate_callables(cache_path)
    return engine_interface.build_plugin(parsed, func_dict)


def _build_short_chain_plugin():
    """Return a lightweight plugin for the autocorrelation guard test."""

    def _distance_modulus_model(z, omega_m, omega_lambda):
        z = np.asarray(z, dtype=float)
        return (
            5.0 * np.log10(1.0 + z)
            + float(omega_m)
            + 0.5 * float(omega_lambda)
        )

    def _distance_helper(z, *params):
        z = np.asarray(z, dtype=float)
        return (1.0 + z) * 100.0

    def _hz_helper(z, *params):
        z = np.asarray(z, dtype=float)
        return np.full(z.shape, 70.0, dtype=float)

    return SimpleNamespace(
        MODEL_NAME="ShortChainModel",
        MODEL_DESCRIPTION="Synthetic plugin for autocorrelation guard tests.",
        MODEL_ABSTRACT="",
        PARAMETER_NAMES=("omega_m", "omega_lambda"),
        PARAMETER_LATEX_NAMES=(r"\Omega_m", r"\Omega_\Lambda"),
        PARAMETER_UNITS=("", ""),
        INITIAL_GUESSES=(0.3, 0.7),
        PARAMETER_BOUNDS=((0.0, 1.0), (0.0, 1.5)),
        FIXED_PARAMS={},
        PARAMETER_PRIORS=(
            {"type": "uniform", "lower": 0.0, "upper": 1.0},
            {"type": "uniform", "lower": 0.0, "upper": 1.5},
        ),
        PARAMETER_PRIOR_OBJECTS=(None, None),
        PARAMETER_TRANSFORMS=None,
        valid_for_distance_metrics=True,
        valid_for_bao=False,
        valid_for_cmb=False,
        CMB_PARAM_MAP={},
        LIKELIHOOD_CONFIG={},
        MODEL_EQUATIONS_LATEX_SN=(),
        MODEL_EQUATIONS_LATEX_BAO=(),
        MODEL_FILENAME=None,
        extras={},
        distance_modulus_model=_distance_modulus_model,
        get_comoving_distance_Mpc=_distance_helper,
        get_luminosity_distance_Mpc=_distance_helper,
        get_angular_diameter_distance_Mpc=_distance_helper,
        get_Hz_per_Mpc=_hz_helper,
        get_DV_Mpc=_distance_helper,
        get_sound_horizon_rs_Mpc=_distance_helper,
        compute_cmb_spectrum=None,
        compute_cmb_spectrum_from_dict=None,
    )


class TestMCMCEngine(unittest.TestCase):
    """Verify that the MCMC engine produces chains and NetCDF output."""

    def _build_lcdm_plugin(self):
        return _build_model_plugin("cosmo_model_lcdm.yml")

    def _build_cfsc_plugin(self):
        return _build_model_plugin("cosmo_model_cfsc.yml")

    def test_sampler_produces_netcdf(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=5,
            pool_size=1,
            burn_in_steps=12,
        )
        n_params = len(plugin.PARAMETER_NAMES)
        expected = (5, max(4, 2 * n_params), n_params)
        self.assertEqual(res["samples"].shape, expected)
        self.assertEqual(res["log_probability"].shape, expected[:2])
        self.assertTrue(res["success"])
        self.assertTrue(np.isfinite(res["chi2_min"]))
        components = res.get("chi2_components", {})
        total_components = (
            components.get("sne", 0.0)
            + components.get("bao", 0.0)
            + components.get("cmb", 0.0)
        )
        self.assertAlmostEqual(res["chi2_total"], total_components)
        self.assertAlmostEqual(res["chi2_sne"], components.get("sne", 0.0))
        self.assertAlmostEqual(
            res.get("chi2_bao", 0.0), components.get("bao", 0.0)
        )
        self.assertAlmostEqual(
            res.get("chi2_cmb", 0.0), components.get("cmb", 0.0)
        )
        self.assertSetEqual(
            set(res["fitted_cosmological_params"].keys()),
            set(plugin.PARAMETER_NAMES),
        )
        self.assertSetEqual(
            set(res["posterior_mean_params"].keys()),
            set(plugin.PARAMETER_NAMES),
        )
        self.assertIsInstance(res["burn_in_steps"], int)
        self.assertIsInstance(res["production_steps"], int)
        self.assertIsInstance(res["n_walkers"], int)
        self.assertIsInstance(res["pool_workers"], int)
        diagnostics = res["diagnostics"]
        for key in ("rhat", "ess_bulk", "ess_tail"):
            self.assertIn(key, diagnostics)
            self.assertTrue(diagnostics[key])
            self.assertTrue(
                all(
                    math.isfinite(value) for value in diagnostics[key].values()
                )
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "chain.nc")
            chain_io.save_posterior(
                res["samples"],
                plugin.PARAMETER_NAMES,
                path,
                metadata={"model": plugin.MODEL_NAME},
            )
            # Use a context manager so Windows can remove the file when the
            # temporary directory cleans up. Without explicitly closing the
            # dataset the cleanup step fails because the file handle remains
            # open on that platform.
            with xr.open_dataset(path, group="posterior") as ds:
                for name in plugin.PARAMETER_NAMES:
                    self.assertIn(name, ds.data_vars)

    def test_progress_logging_reports_statistics(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        with self.assertLogs(level="INFO") as captured:
            cosmo_engine_mcmc.fit_sne_parameters(
                sne_df,
                plugin,
                n_walkers=4,
                n_steps=6,
                pool_size=1,
                progress_granularity=4,
                burn_in_steps=12,
            )

        joined = "\n".join(captured.output)
        self.assertIn("logP μ=", joined)
        self.assertNotIn("Walker[", joined)
        for name in plugin.PARAMETER_NAMES:
            self.assertIn(f"    {name}:", joined)
        self.assertNotIn("omitted", joined)
        self.assertNotIn("snapshot", joined)

    def test_explicit_pool_size_respected(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=2,
            burn_in_steps=4,
        )
        self.assertEqual(res["pool_workers"], 2)
        self.assertGreaterEqual(res["n_walkers"], res["pool_workers"])

    def test_log_probability_penalty(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_sne_logposterior(
            plugin,
            sne_df,
        )
        bad = np.array([200.0] + list(plugin.INITIAL_GUESSES[1:]))
        lp = posterior(bad)
        self.assertTrue(np.isneginf(lp))

    def test_invalid_walkers_are_reseeded(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        bounds = plugin.PARAMETER_BOUNDS
        lower = np.array(
            [-np.inf if low is None else float(low) for low, _ in bounds]
        )
        upper = np.array(
            [np.inf if high is None else float(high) for _, high in bounds]
        )
        ndim = len(plugin.PARAMETER_NAMES)
        coords = np.vstack(
            [
                np.asarray(plugin.INITIAL_GUESSES, dtype=float),
                np.full(ndim, np.nan),
            ]
        )
        posterior, _, _ = _build_sne_logposterior(
            plugin,
            sne_df,
        )
        log_prob = np.array([posterior(coords[0]), np.nan])
        rng = np.random.default_rng(12345)
        new_coords, new_log_prob = _reseed_invalid_walkers(
            coords,
            log_prob,
            lower=lower,
            upper=upper,
            rng=rng,
            log_probability_fn=lambda pos: posterior(pos),
            reference_position=np.asarray(plugin.INITIAL_GUESSES, dtype=float),
        )
        self.assertTrue(np.all(np.isfinite(new_coords)))
        self.assertTrue(np.all(np.isfinite(new_log_prob)))

    def test_sampler_respects_shared_seed(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        set_random_seed(31415)
        first = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=8,
        )
        set_random_seed(31415)
        second = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=8,
        )
        np.testing.assert_array_equal(first["samples"], second["samples"])
        np.testing.assert_array_equal(
            first["log_probability"], second["log_probability"]
        )
        self.assertTrue(first["success"])  # sanity-check regression inputs
        self.assertTrue(second["success"])  # matches the deterministic run
        set_random_seed(0)

    def test_active_log_probability_expands_parameters(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_sne_logposterior(plugin, sne_df)
        bounds = plugin.PARAMETER_BOUNDS
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logging.getLogger()
        )
        template = np.asarray(plugin.INITIAL_GUESSES, dtype=float)
        active_indices = np.flatnonzero(~fixed_mask)
        adapter = _ActiveLogProbability(posterior, template, active_indices)
        trial = template[active_indices]
        assembled = adapter.assemble_full(trial)
        self.assertTrue(np.allclose(assembled[active_indices], trial))
        self.assertTrue(
            np.allclose(assembled[fixed_mask], template[fixed_mask])
        )
        value = adapter(trial)
        self.assertIsInstance(value, float)
        self.assertTrue(math.isfinite(value) or math.isneginf(value))

        clipped = np.clip(
            trial + 0.1,
            lower[~fixed_mask],
            upper[~fixed_mask],
        )
        assembled_clipped = adapter.assemble_full(clipped)
        self.assertTrue(
            np.allclose(assembled_clipped[active_indices], clipped)
        )

    def test_sampler_runs_with_spawn_pool(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=6,
            pool_size=2,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["pool_workers"], 2)
        self.assertTrue(math.isfinite(result["log_posterior_best"]))

    def test_sampler_handles_fixed_bounds(self):
        plugin = self._build_cfsc_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=30,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        self.assertEqual(chain.shape[2], len(plugin.PARAMETER_NAMES))
        const_idx = plugin.PARAMETER_NAMES.index("c")
        fixed_spread = np.ptp(chain[:, :, const_idx])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)

    def test_likelihood_state_reported(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        state = result["likelihood_state"]
        self.assertIn("components", state["metadata"])
        self.assertIn("sne", state["metadata"]["components"])
        self.assertTrue(math.isfinite(result["log_likelihood_best"]))
        self.assertTrue(math.isfinite(result["log_posterior_best"]))
        self.assertTrue(math.isfinite(result["log_prior_best"]))

    def test_joint_fit_component_chi2_totals(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        initial = np.asarray(plugin.INITIAL_GUESSES, dtype=float)
        z_bao = np.array([0.1])
        dm = plugin.get_comoving_distance_Mpc(z_bao, *initial)
        rs = plugin.get_sound_horizon_rs_Mpc(*initial)
        bao_df = pd.DataFrame(
            {
                "redshift": z_bao,
                "observable_type": ["DM_over_rs"],
                "value": dm / rs,
                "error": [0.05],
            }
        )
        bao_df.attrs["covariance_matrix_inv"] = np.eye(1)

        ells = np.arange(30, 34)
        camb_params = plugin.get_camb_params(initial)
        dl_vals = cosmo_engine_mcmc.compute_cmb_spectrum(
            camb_params,
            ells,
            spectra=("TT",),
        )
        cmb_df = pd.DataFrame({"ell": ells, "Dl_obs": dl_vals})
        cmb_df.attrs["covariance_matrix_inv"] = np.eye(len(ells))

        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_walkers=6,
            n_steps=6,
            pool_size=1,
            burn_in_steps=12,
        )
        components = result.get("chi2_components", {})
        total = sum(components.values())
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["chi2_total"], total, places=6)
        self.assertAlmostEqual(
            result["chi2_bao"], components.get("bao", float("nan"))
        )
        self.assertAlmostEqual(
            result["chi2_cmb"], components.get("cmb", float("nan"))
        )
        self.assertIn(
            "bao", result["likelihood_state"]["metadata"]["components"]
        )
        self.assertIn(
            "cmb", result["likelihood_state"]["metadata"]["components"]
        )

    def test_comoving_distance_vectorized(self):
        plugin = self._build_lcdm_plugin()
        params = plugin.INITIAL_GUESSES
        z_vals = np.array([0.1, 0.2, 0.3])
        arr = plugin.get_comoving_distance_Mpc(z_vals, *params)
        loop = np.array(
            [
                plugin.get_comoving_distance_Mpc(float(z), *params)
                for z in z_vals
            ]
        )
        np.testing.assert_allclose(arr, loop)


class TestStepProgressEmitter(unittest.TestCase):
    """Exercise the idle spinner helper under deterministic timing."""

    def test_idle_tick_repaints_after_single_update(self) -> None:
        """``tick`` repaints repeatedly after a lone walker update."""

        class _DummyBar:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int, int]] = []
                self.uses_live_display = True

            def start_step(self, step_index: int, walker_total: int) -> str:
                self.calls.append(("start", step_index, walker_total))
                return ""

            def update(
                self,
                step_index: int,
                *,
                processed: int,
                total: int,
                step_progress: float | None = None,
                force: bool = False,
            ) -> str:
                self.calls.append(("update", processed, total))
                return f"line-{len(self.calls)}"

        dummy_bar = _DummyBar()
        emitter = StepProgressEmitter(dummy_bar)
        idle_times = [0.0]

        def _fake_timer() -> float:
            return idle_times[0]

        def _update_count() -> int:
            count = 0
            for call in dummy_bar.calls:
                if call[0] == "update":
                    count += 1
            return count

        emitter._timer = _fake_timer  # type: ignore[attr-defined]
        emitter._idle_interval = 0.1  # type: ignore[attr-defined]
        emitter.start(1, 1)
        idle_times[0] = 0.05
        emitter(1, 1)
        initial_updates = _update_count()
        idle_times[0] = 0.14
        emitter.tick()
        self.assertEqual(_update_count(), initial_updates)
        idle_times[0] = 0.16
        emitter.tick()
        idle_times[0] = 0.30
        emitter.tick()
        self.assertGreaterEqual(
            _update_count(),
            initial_updates + 2,
        )
        final_count = _update_count()
        emitter.clear()
        idle_times[0] = 0.50
        emitter.tick()
        self.assertEqual(
            _update_count(),
            final_count,
        )


class TestMCMCHelpers(unittest.TestCase):
    """Exercise helper utilities that remain active without arviz."""

    def test_near_fixed_bounds_are_flagged(self):
        logger = logging.getLogger("test.mcmc.bounds")
        bounds = [(1.0, 1.0 + 5e-10), (0.0, 2.0)]
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
        self.assertTrue(fixed_mask[0])
        self.assertFalse(fixed_mask[1])
        self.assertAlmostEqual(lower[0], 1.0)
        self.assertAlmostEqual(upper[0], 1.0 + 5e-10)

    def test_initialise_walkers_relaxes_condition_number(self):
        initial = np.array([5.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([10.0, 10.0])
        rng = np.random.default_rng(42)

        def logp(_):
            return 0.0

        walkers, logp_vals = _initialise_active_walkers(
            initial,
            lower,
            upper,
            n_walkers=6,
            rng=rng,
            log_probability_fn=logp,
        )
        self.assertTrue(np.all(np.isfinite(logp_vals)))
        cond = _estimate_condition_number(walkers)
        if cond is not None:
            self.assertLessEqual(cond, 1e12)

    def test_sampler_handles_near_fixed_bounds(self):
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
        tight_value = plugin.INITIAL_GUESSES[0]
        plugin.PARAMETER_BOUNDS = list(plugin.PARAMETER_BOUNDS)
        plugin.PARAMETER_BOUNDS[0] = (
            tight_value - 5e-10,
            tight_value + 5e-10,
        )
        plugin.INITIAL_GUESSES = list(plugin.INITIAL_GUESSES)
        plugin.INITIAL_GUESSES[0] = tight_value

        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        result = cosmo_engine_mcmc.fit_sne_parameters(
            sne_df,
            plugin,
            n_walkers=10,
            n_steps=4,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        fixed_spread = np.ptp(chain[:, :, 0])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)


class TestAutocorrelationGuard(unittest.TestCase):
    """Validate that short chains skip autocorrelation diagnostics."""

    def test_short_chain_returns_none_without_runtime_warning(self):
        plugin = _build_short_chain_plugin()
        z_values = np.linspace(0.01, 0.03, 3)
        baseline = np.array([0.3, 0.7])
        mu_model = (
            5.0 * np.log10(1.0 + z_values) + baseline[0] + 0.5 * baseline[1]
        )
        sne_df = pd.DataFrame(
            {"zcmb": z_values, "mu_obs": mu_model, "e_mu_obs": np.full(3, 0.1)}
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = cosmo_engine_mcmc.fit_sne_parameters(
                sne_df,
                plugin,
                n_walkers=4,
                n_steps=3,
                pool_size=1,
                burn_in_steps=1,
                progress_granularity=1,
            )
        runtime_warnings = [
            item
            for item in caught
            if issubclass(item.category, RuntimeWarning)
        ]
        self.assertIsNone(result.get("autocorrelation_time"))
        self.assertFalse(runtime_warnings)


class BatchProgressBarTestCase(unittest.TestCase):
    """Exercise the sampler progress bar without timing metadata."""

    def test_progress_bar_reports_pluralisation_and_width(self) -> None:
        """Progress lines reflect partial updates and honour the width."""

        captured: list[tuple[str, str]] = []

        def _capture(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            captured.append((msg, end))

        bar = BatchProgressBar(
            "Test stage",
            4,
            display=True,
        )
        with (
            mock.patch(
                "copernican_lib.progress.console.write",
                side_effect=_capture,
            ),
        ):
            bar.start_batch(1, 4)
            self.assertTrue(bar.uses_live_display)
            initial_line = bar.start_step(1, walker_total=8)
            self.assertIn("  0%", initial_line)
            spinner_frames = set(BatchProgressBar._SPINNER_FRAMES)
            self.assertTrue(spinner_frames & set(initial_line))
            half_line = bar.update(1, processed=4, total=8)
            self.assertIn("step 1 of 4 steps", half_line)
            self.assertIn("4 steps remaining", half_line)
            self.assertTrue(spinner_frames & set(half_line))
            bar_segment = half_line.lstrip("\r").split(" ", 1)[0]
            self.assertEqual(
                len(bar_segment),
                BatchProgressBar._BAR_WIDTH,
            )
            self.assertIn("█", bar_segment)
            self.assertIn("4/8", half_line)
            walker_segment = (
                half_line.split(";", 1)[1].strip().split(",", 1)[0]
            )
            walker_bar_segment, walker_counts = walker_segment.rsplit(" ", 1)
            self.assertEqual(
                len(walker_bar_segment),
                BatchProgressBar._WALKER_BAR_WIDTH,
            )
            self.assertEqual(walker_counts, "4/8")
            later_line = bar.start_step(2, walker_total=8)
            self.assertIn("step 2 of 4 steps", later_line)
            self.assertIn("3 steps remaining", later_line)
            bar.start_step(3, walker_total=8)
            near_end = bar.update(3, processed=8, total=8)
            self.assertIn("1 step remaining", near_end)
            bar.start_step(4, walker_total=8)
            final_line = bar.update(4, processed=8, total=8)
            self.assertIn("0 steps remaining", final_line)
            bar.finish_batch()

        announcements = [msg for msg, _ in captured if "batch" in msg]
        self.assertTrue(announcements)
        newline_calls = [end for msg, end in captured if msg == ""]
        self.assertIn("\n", newline_calls)
        emitted_lines = [msg for msg, end in captured if end == ""]
        self.assertTrue(emitted_lines)
        # Spinner frames appear in subsequent captured repaint strings.
        spinner_hits = [
            set(msg) & spinner_frames for msg, end in captured if end == ""
        ]
        self.assertTrue(any(hit for hit in spinner_hits))

    def test_progress_bar_emits_partial_block_during_small_updates(
        self,
    ) -> None:
        """Fractional updates render the Unicode sub-block glyphs."""

        bar = BatchProgressBar(
            "Unicode stage",
            10,
            display=True,
        )
        with (
            mock.patch(
                "copernican_lib.progress.console.write",
                side_effect=lambda *args, **kwargs: None,
            ),
        ):
            bar.start_batch(1, 5)
            fractional_line = bar.start_step(1, walker_total=20)
            partial_line = bar.update(1, processed=1, total=20)
            self.assertIsNotNone(fractional_line)
            bar_segment = partial_line.lstrip("\r").split(" ", 1)[0]
            self.assertEqual(
                len(bar_segment),
                BatchProgressBar._BAR_WIDTH,
            )
            partial_set = set(BatchProgressBar._PARTIAL_GLYPHS)
            has_partial = bool(set(bar_segment) & partial_set)
            self.assertTrue(has_partial)
            spinner_frames = set(BatchProgressBar._SPINNER_FRAMES)
            self.assertTrue(set(partial_line) & spinner_frames)

    def test_force_update_rerenders_identical_text(self) -> None:
        """Explicitly forced updates repaint even when text is stable."""

        with mock.patch("copernican_lib.progress.console.write") as patched:
            bar = BatchProgressBar("Forced stage", 2, display=True)
            bar.start_batch(1, 2)
            bar.start_step(1, 4)
            with mock.patch.object(bar, "_next_spinner", return_value="⠋"):
                first_line = bar.update(1, processed=2, total=4)
                self.assertIsNotNone(first_line)
                patched.reset_mock()
                self.assertIsNone(bar.update(1, processed=2, total=4))
                forced_line = bar.update(1, processed=2, total=4, force=True)
            self.assertIsNotNone(forced_line)
            patched.assert_called()

    def test_finish_batch_clears_active_line(self) -> None:
        """Closing a batch wipes the progress line before spacing."""

        with mock.patch("copernican_lib.progress.console.write") as patched:
            bar = BatchProgressBar("Cleanup stage", 1, display=True)
            bar.start_batch(1, 1)
            bar.start_step(1, 4)
            bar.update(1, processed=4, total=4)
            patched.reset_mock()
            bar.finish_batch()
            self.assertGreaterEqual(patched.call_count, 3)
            blank_call = patched.call_args_list[0]
            self.assertTrue(blank_call.args[0].startswith("\r"))
            self.assertTrue(set(blank_call.args[0][1:]) <= {" "})

    def test_finish_batch_clears_line_without_updates(self) -> None:
        """Initial 0% renders are tracked so cleanup blanks the console."""

        with mock.patch("copernican_lib.progress.console.write") as patched:
            bar = BatchProgressBar("Idle stage", 2, display=True)
            bar.start_batch(1, 2)
            patched.reset_mock()
            bar.finish_batch()
            self.assertGreaterEqual(patched.call_count, 3)
            blank_calls = [
                call.args[0] for call in patched.call_args_list if call.args
            ]
            self.assertTrue(
                any(
                    msg.startswith("\r") and set(msg[1:]) <= {" "}
                    for msg in blank_calls
                )
            )


class ProgressIntegrationTestCase(unittest.TestCase):
    """Ensure sampler hooks stream live walker updates."""

    def test_sampler_emits_walker_progress_before_step_finishes(self) -> None:
        """Instrumented sampler writes partial walker counts to the bar."""

        def _log_prob(coord: np.ndarray) -> float:
            vec = np.atleast_1d(coord)
            return float(-0.5 * np.dot(vec, vec))

        sampler = emcee.EnsembleSampler(6, 2, _log_prob)
        bar = BatchProgressBar("Integration stage", 1, display=True)
        emitter = StepProgressEmitter(bar)
        configure_sampler_progress_reporting(sampler, emitter)
        initial = np.random.default_rng(42).standard_normal((6, 2))
        iterator = sampler.sample(initial, iterations=1, progress=False)

        with mock.patch("copernican_lib.progress.console.write") as patched:
            bar.start_batch(1, 1)
            emitter.start(1, sampler.nwalkers)
            next(iterator)
            emitter.clear()
            bar.update(
                1,
                processed=sampler.nwalkers,
                total=sampler.nwalkers,
                step_progress=1.0,
            )
            bar.finish_batch()

        walker_prefix = f"/{sampler.nwalkers}"
        walker_updates = [
            call.args[0]
            for call in patched.call_args_list
            if call.args and walker_prefix in call.args[0]
        ]
        self.assertGreaterEqual(len(walker_updates), 2)
        self.assertTrue(
            any(f"1{walker_prefix}" in msg for msg in walker_updates)
        )

    def test_stage_cleanup_wipes_progress_when_sampler_errors(self) -> None:
        """Failing samplers still trigger a final console clear."""

        class _FailingSampler:
            def __init__(self) -> None:
                self.nwalkers = 4
                self._moves = [moves.StretchMove()]

            def sample(
                self, *args, **kwargs
            ):  # pragma: no cover - generator stub
                raise RuntimeError("sampler failure")

            def get_last_sample(self):  # pragma: no cover - unused guard
                raise AssertionError("should not be called")

        sampler = _FailingSampler()
        initial_state = np.zeros((sampler.nwalkers, 2))

        with mock.patch("copernican_lib.progress.console.write") as patched:
            with self.assertRaises(RuntimeError):
                cosmo_engine_mcmc._run_stage_with_progress(
                    sampler,
                    initial_state,
                    3,
                    stage_name="error-prone",
                    logger=logging.getLogger("copernican.tests"),
                    progress_granularity=2,
                    summary_callback=None,
                    progress_label="Failing stage",
                    display_progress=True,
                )

        blank_calls = [
            call.args[0] for call in patched.call_args_list if call.args
        ]
        self.assertTrue(
            any(
                msg.startswith("\r") and set(msg[1:]) <= {" "}
                for msg in blank_calls
            )
        )


class ConfigureSamplerProgressReportingTestCase(unittest.TestCase):
    """Ensure sampler move collections attach progress notifiers."""

    def test_weight_first_pair_wraps_stretch_move(self) -> None:
        """Tuples storing ``(weight, move)`` gain reporting wrappers."""

        sampler = SimpleNamespace(_moves=[(0.75, moves.StretchMove())])
        notifier = object()

        configure_sampler_progress_reporting(sampler, notifier)

        weight, move_obj = sampler._moves[0]
        self.assertEqual(weight, 0.75)
        self.assertIsInstance(move_obj, progress_helpers._ReportingStretchMove)
        self.assertIs(getattr(move_obj, "_progress_notifier"), notifier)

    def test_move_first_pair_wraps_stretch_move(self) -> None:
        """Tuples storing ``(move, weight)`` gain reporting wrappers."""

        base_move = moves.StretchMove(a=3.5)
        sampler = SimpleNamespace(_moves=[(base_move, 0.5)])
        notifier = object()

        configure_sampler_progress_reporting(sampler, notifier)

        move_obj, weight = sampler._moves[0]
        self.assertEqual(weight, 0.5)
        self.assertIsInstance(move_obj, progress_helpers._ReportingStretchMove)
        self.assertIs(getattr(move_obj, "_progress_notifier"), notifier)
        self.assertEqual(getattr(move_obj, "a"), getattr(base_move, "a"))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
