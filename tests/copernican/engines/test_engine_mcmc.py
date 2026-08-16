"""Behavior and integration tests for copernican.engines.engine_mcmc."""

import logging
import math
import os
import tempfile
import unittest
import warnings
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from unittest import mock

import numpy
import pandas
import xarray as xarray_dataset

from copernican.engines import engine_mcmc as module
from copernican.engines.engine_mcmc import (
    _ActiveLogProbability,
    _build_joint_logposterior,
    _classify_parameter_bounds,
    _estimate_condition_number,
    _initialise_active_walkers,
    _preflight_initial_model_point,
    _reseed_invalid_walkers,
    _resolve_mcmc_pool_processes,
)
from copernican.lib import chain_io
from copernican.lib import engine_adapter as engine_plugin_validation
from copernican.lib import model_coder, model_spec_validator
from copernican.lib.likelihoods.cmb.native_errors import (
    NativeInitialPointError,
)
from copernican.lib.progress import BatchProgressBar
from copernican.lib.utils import set_random_seed


def _build_model_plugin(
    yaml_filename: str,
    *,
    compact_native: bool = False,
    fixed_native: bool = False,
):
    """Return a validated plugin with optional bounded native test controls."""

    models_dir = Path(__file__).resolve().parents[3] / "copernican" / "models"
    yaml_path = models_dir / yaml_filename
    with tempfile.TemporaryDirectory() as cache_dir:
        cache_path = model_spec_validator.validate_and_cache_model(
            yaml_path,
            cache_dir,
        )
        func_dict, parsed = model_coder.generate_callables(cache_path)
    if compact_native:
        # The joint-likelihood test exercises native execution repeatedly.
        # Keep that fixture explicitly native while bounding its declared
        # accuracy tier so the test measures likelihood behavior, not a
        # production full-range spectrum on every proposal.
        numerical = parsed["cmb"]["numerical"]
        numerical.update(
            {
                "ell_max": 40,
                "k_min": 1.0e-3,
                "k_max": 0.12,
                "k_sample_count": 8,
                "eta_sample_count": 16,
                "source_grid_multiplier": 1,
                "photon_hierarchy_l_max": 4,
                "neutrino_hierarchy_l_max": 2,
            }
        )
        perturbations = parsed["cmb"]["perturbations"]
        perturbations["accuracy_controls"].pop("accuracy_tier", None)
        perturbations["accuracy_controls"]["scalar_reference_ells"] = [
            2,
            40,
        ]
        perturbations["numerics"].update(
            {
                "ell_max": 40,
                "k_min": 1.0e-3,
                "k_max": 0.12,
                "k_sample_count": 8,
                "eta_sample_count": 16,
                "source_grid_multiplier": 1,
                "photon_hierarchy_l_max": 4,
                "neutrino_hierarchy_l_max": 2,
            }
        )
        momentum_grids = perturbations["numerics"].get("momentum_grids", {})
        if "massive_neutrino_default" in momentum_grids:
            momentum_grids["massive_neutrino_default"].update(
                {
                    "count": 2,
                    "q_min": 0.1,
                    "q_max": 12.0,
                }
            )
    plugin = engine_plugin_validation.build_plugin(parsed, func_dict)
    if fixed_native:
        plugin.PARAMETER_BOUNDS = tuple(
            (float(value), float(value)) for value in plugin.INITIAL_GUESSES
        )
    return plugin


def _build_short_chain_plugin():
    """Return a lightweight plugin for the autocorrelation guard test."""

    def _distance_modulus_model(z, omega_m, omega_lambda):
        z = numpy.asarray(z, dtype=float)
        return (
            5.0 * numpy.log10(1.0 + z)
            + float(omega_m)
            + 0.5 * float(omega_lambda)
        )

    def _distance_helper(z, *params):
        z = numpy.asarray(z, dtype=float)
        return (1.0 + z) * 100.0

    def _hz_helper(z, *params):
        z = numpy.asarray(z, dtype=float)
        return numpy.full(z.shape, 70.0, dtype=float)

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
        CMB_CONTRACT={},
        CMB_PARAM_MAP={},
        CMB_PERTURBATION_CONTRACT={},
        CMB_PERTURBATION_DATA=None,
        CMB_NATIVE_RUNTIME=None,
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
        compute_cmb_spectrum_from_contract=None,
    )


class TestCosmoEngineMcmc(unittest.TestCase):
    """Exercise the reusable helpers and engine behavior."""

    def test_engine_metadata(self) -> None:
        self.assertEqual(module.ENGINE_KIND, "mcmc")
        self.assertEqual(module.ENGINE_LABEL, "Ensemble MCMC sampler")
        self.assertEqual(module.ENGINE_VERSION, "7.6.20")
        self.assertTrue(module.ENGINE_SETTINGS)
        self.assertTrue(module.ENGINE_PROGRESS_CHUNKS)

    def test_active_log_probability_rebuilds_full_vector(self) -> None:
        def posterior(arr):
            return float(numpy.sum(arr))

        adapter = _ActiveLogProbability(
            posterior,
            numpy.array([1.0, 2.0, 3.0]),
            numpy.array([0, 2]),
        )
        full = adapter.assemble_full(numpy.array([4.0, 5.0]))
        self.assertTrue(numpy.allclose(full, numpy.array([4.0, 2.0, 5.0])))
        self.assertEqual(adapter(numpy.array([4.0, 5.0])), 11.0)

    def test_sampler_produces_netcdf(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=5,
            pool_size=1,
            burn_in_steps=12,
        )
        n_params = len(plugin.PARAMETER_NAMES)
        expected = (5, res["n_walkers"], n_params)
        self.assertEqual(res["samples"].shape, expected)
        self.assertEqual(res["log_probability"].shape, expected[:2])
        self.assertTrue(res["success"])
        self.assertTrue(numpy.isfinite(res["chi2_min"]))
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
        fixed_name = "Tcmb_K"
        total_draws = float(res["production_steps"] * res["n_walkers"])
        self.assertEqual(diagnostics["rhat"][fixed_name], 1.0)
        self.assertEqual(diagnostics["ess_bulk"][fixed_name], total_draws)
        self.assertEqual(diagnostics["ess_tail"][fixed_name], total_draws)

    def test_legacy_fit_alias_warns_and_runs(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = module.fit_sne_parameters(
                sne_df,
                plugin,
                n_walkers=4,
                n_steps=4,
                pool_size=1,
                burn_in_steps=2,
                display_progress=False,
            )
        self.assertTrue(res["success"])
        self.assertTrue(
            any(
                "fit_sne_parameters is deprecated" in str(warning.message)
                for warning in caught
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
            try:
                open_kwargs = {"group": "posterior"}
                dataset = xarray_dataset.open_dataset(path, **open_kwargs)
                expects_group = True
            except ValueError:
                open_kwargs = {}
                dataset = xarray_dataset.open_dataset(path, **open_kwargs)
                expects_group = False

            with dataset as posterior_dataset:
                for name in plugin.PARAMETER_NAMES:
                    self.assertIn(name, posterior_dataset.data_vars)
                self.assertEqual(
                    posterior_dataset.attrs.get("model"), plugin.MODEL_NAME
                )
                if not expects_group:
                    self.assertEqual(
                        posterior_dataset.attrs.get("posterior_group"), "/"
                    )

    def test_progress_logging_reports_statistics(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        with self.assertLogs(level="INFO") as captured:
            module.fit_cosmology_parameters(
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

    def test_helper_fits_use_classified_fixed_bounds(self) -> None:
        lower, upper, fixed_mask = _classify_parameter_bounds(
            [(0.5, 0.5), (0.0, 1.0), (None, None)],
            logger=logging.getLogger("test.mcmc.bounds"),
        )
        self.assertTrue(fixed_mask[0])
        self.assertFalse(fixed_mask[1])
        self.assertFalse(fixed_mask[2])
        self.assertAlmostEqual(lower[0], 0.5)
        self.assertAlmostEqual(upper[0], 0.5)

    def test_condition_number_estimator_handles_well_conditioned(self) -> None:
        matrix = numpy.array([[1.0, 0.0], [0.0, 2.0]])
        self.assertLess(_estimate_condition_number(matrix), 3.0)

    def test_joint_logposterior_builds_with_active_coordinates(self) -> None:
        plugin = _build_short_chain_plugin()
        posterior, joint_like, labels = _build_joint_logposterior(
            plugin,
            pandas.DataFrame(
                {"zcmb": [0.1], "mu_obs": [40.0], "e_mu_obs": [0.1]}
            ),
            None,
            None,
        )
        self.assertTrue(labels)
        self.assertTrue(callable(posterior))
        self.assertTrue(callable(joint_like))

    def test_initialise_and_reseed_walkers(self) -> None:
        lower = numpy.array([0.0, 0.0])
        upper = numpy.array([1.0, 1.0])
        rng = numpy.random.default_rng(42)

        def logp(_):
            return 0.0

        initial, logp_vals = _initialise_active_walkers(
            numpy.array([0.3, 0.7]),
            lower,
            upper,
            n_walkers=4,
            rng=rng,
            log_probability_fn=logp,
        )
        self.assertEqual(initial.shape[1], 2)
        self.assertTrue(numpy.all(numpy.isfinite(logp_vals)))
        reseeded, reseeded_logp = _reseed_invalid_walkers(
            initial,
            numpy.array([True, False, True, False]),
            lower=lower,
            upper=upper,
            rng=rng,
            log_probability_fn=logp,
            reference_position=numpy.array([0.3, 0.7]),
        )
        self.assertEqual(reseeded.shape, initial.shape)
        self.assertTrue(numpy.isfinite(reseeded).all())
        self.assertTrue(numpy.isfinite(reseeded_logp).all())

    def test_initial_point_preflight_rejects_before_walker_creation(self):
        """A non-finite nominal point must stop before proposals exist."""

        with self.assertRaises(NativeInitialPointError):
            _preflight_initial_model_point(
                lambda _parameters: float("-inf"),
                (0.3, 0.7),
            )

        plugin = _build_short_chain_plugin()
        sne_df = pandas.DataFrame(
            {"zcmb": [0.1], "mu_obs": [40.0], "e_mu_obs": [0.1]}
        )
        with (
            mock.patch.object(
                module,
                "_build_joint_logposterior",
                return_value=(
                    lambda _parameters: float("-inf"),
                    mock.Mock(),
                    mock.Mock(),
                ),
            ),
            mock.patch.object(module, "_initialise_active_walkers") as walkers,
            self.assertRaises(NativeInitialPointError),
        ):
            module.fit_cosmology_parameters(
                sne_df,
                plugin,
                n_walkers=4,
                n_steps=1,
                pool_size=1,
                burn_in_steps=1,
            )
        walkers.assert_not_called()

    def test_worker_initializer_prepares_runtime_once(self) -> None:
        """Spawn workers should install and prepare one callable bundle."""

        prepare_runtime = mock.Mock()
        posterior = mock.Mock(return_value=-2.5)
        posterior.like = mock.Mock(prepare_worker_runtime=prepare_runtime)
        adapter = _ActiveLogProbability(
            posterior,
            numpy.asarray((1.0, 2.0)),
            numpy.asarray((1,), dtype=int),
        )
        adapter.prepare_worker_runtime()
        prepare_runtime.assert_called_once_with()
        prepare_runtime.reset_mock()

        previous = module._WORKER_LOG_PROBABILITY
        try:
            module._initialize_mcmc_worker(adapter)
            first = module._worker_log_probability(numpy.asarray((3.0,)))
            second = module._worker_log_probability(numpy.asarray((4.0,)))
        finally:
            module._WORKER_LOG_PROBABILITY = previous

        prepare_runtime.assert_called_once_with()
        self.assertEqual(first, -2.5)
        self.assertEqual(second, -2.5)
        self.assertEqual(posterior.call_count, 2)

    @mock.patch("copernican.engines.engine_mcmc.BatchProgressBar")
    def test_progress_bar_reports_updates(self, bar_cls) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        bar_instance = mock.MagicMock()
        bar_cls.return_value = bar_instance
        module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=6,
            pool_size=1,
            progress_granularity=4,
            burn_in_steps=12,
        )
        self.assertGreaterEqual(bar_cls.call_count, 1)
        _, kwargs = bar_cls.call_args_list[0]
        self.assertIn("display", kwargs)
        self.assertGreaterEqual(bar_instance.start_batch.call_count, 1)
        self.assertGreaterEqual(bar_instance.update.call_count, 1)
        self.assertGreaterEqual(bar_instance.finish_batch.call_count, 1)

    def test_explicit_pool_size_respected(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        res = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=2,
            burn_in_steps=4,
        )
        self.assertEqual(res["pool_workers"], 2)
        self.assertGreaterEqual(res["n_walkers"], res["pool_workers"])

    def test_pool_size_is_capped_to_avoid_cpu_oversubscription(self) -> None:
        """A requested pool must leave one CPU for its parent process."""

        with mock.patch.object(
            module.multiprocessing_module,
            "cpu_count",
            return_value=3,
        ):
            worker_count = _resolve_mcmc_pool_processes(
                requested_pool=8,
                n_walkers=32,
            )

        self.assertEqual(worker_count, 2)

    def test_log_probability_penalty(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_joint_logposterior(
            plugin,
            sne_df,
        )
        bad = numpy.array([200.0] + list(plugin.INITIAL_GUESSES[1:]))
        log_posterior = posterior(bad)
        self.assertTrue(numpy.isneginf(log_posterior))

    def test_invalid_walkers_are_reseeded(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        bounds = plugin.PARAMETER_BOUNDS
        lower = numpy.array(
            [-numpy.inf if low is None else float(low) for low, _ in bounds]
        )
        upper = numpy.array(
            [numpy.inf if high is None else float(high) for _, high in bounds]
        )
        ndim = len(plugin.PARAMETER_NAMES)
        coords = numpy.vstack(
            [
                numpy.asarray(plugin.INITIAL_GUESSES, dtype=float),
                numpy.full(ndim, numpy.nan),
            ]
        )
        posterior, _, _ = _build_joint_logposterior(
            plugin,
            sne_df,
        )
        log_prob = numpy.array([posterior(coords[0]), numpy.nan])
        rng = numpy.random.default_rng(12345)
        new_coords, new_log_prob = _reseed_invalid_walkers(
            coords,
            log_prob,
            lower=lower,
            upper=upper,
            rng=rng,
            log_probability_fn=lambda pos: posterior(pos),
            reference_position=numpy.asarray(
                plugin.INITIAL_GUESSES, dtype=float
            ),
        )
        self.assertTrue(numpy.all(numpy.isfinite(new_coords)))
        self.assertTrue(numpy.all(numpy.isfinite(new_log_prob)))

    def test_sampler_respects_shared_seed(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        set_random_seed(31415)
        first = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=8,
        )
        set_random_seed(31415)
        second = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=4,
            n_steps=4,
            pool_size=1,
            burn_in_steps=8,
        )
        numpy.testing.assert_array_equal(first["samples"], second["samples"])
        numpy.testing.assert_array_equal(
            first["log_probability"], second["log_probability"]
        )
        self.assertTrue(first["success"])
        self.assertTrue(second["success"])
        set_random_seed(0)

    def test_active_log_probability_expands_parameters(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01],
                "mu_obs": [40.0],
                "e_mu_obs": [0.1],
            }
        )
        posterior, _, _ = _build_joint_logposterior(plugin, sne_df)
        bounds = plugin.PARAMETER_BOUNDS
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logging.getLogger()
        )
        template = numpy.asarray(plugin.INITIAL_GUESSES, dtype=float)
        active_indices = numpy.flatnonzero(~fixed_mask)
        adapter = _ActiveLogProbability(posterior, template, active_indices)
        trial = template[active_indices]
        assembled = adapter.assemble_full(trial)
        self.assertTrue(numpy.allclose(assembled[active_indices], trial))
        self.assertTrue(
            numpy.allclose(assembled[fixed_mask], template[fixed_mask])
        )
        value = adapter(trial)
        self.assertIsInstance(value, float)
        self.assertTrue(math.isfinite(value) or math.isneginf(value))

        clipped = numpy.clip(
            trial + 0.1,
            lower[~fixed_mask],
            upper[~fixed_mask],
        )
        assembled_clipped = adapter.assemble_full(clipped)
        self.assertTrue(
            numpy.allclose(assembled_clipped[active_indices], clipped)
        )

    def test_sampler_runs_with_spawn_pool(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = module.fit_cosmology_parameters(
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

    def test_sampler_handles_fixed_bounds(self) -> None:
        plugin = _build_model_plugin("model_wcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=30,
            n_steps=10,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        self.assertEqual(chain.shape[2], len(plugin.PARAMETER_NAMES))
        const_idx = plugin.PARAMETER_NAMES.index("c_light")
        fixed_spread = numpy.ptp(chain[:, :, const_idx])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)

    def test_likelihood_state_reported(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = module.fit_cosmology_parameters(
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

    def test_joint_fit_component_chi2_totals(self) -> None:
        plugin = _build_model_plugin(
            "model_lcdm.yml",
            compact_native=True,
            fixed_native=True,
        )
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        initial = numpy.asarray(plugin.INITIAL_GUESSES, dtype=float)
        z_bao = numpy.array([0.1])
        distance_modulus = plugin.get_comoving_distance_Mpc(z_bao, *initial)
        sound_horizon = plugin.get_sound_horizon_rs_Mpc(*initial)
        bao_df = pandas.DataFrame(
            {
                "redshift": z_bao,
                "observable_type": ["DM_over_rs"],
                "value": distance_modulus / sound_horizon,
                "error": [0.05],
            }
        )
        bao_df.attrs["covariance_matrix_inv"] = numpy.eye(1)

        ells = numpy.arange(30, 34)
        cmb_contract = plugin.get_cmb_contract(initial)
        perturbation_contract = plugin.get_cmb_perturbation_contract(initial)
        structured_contract = dict(cmb_contract)
        structured_contract["perturbations"] = perturbation_contract
        dl_vals = module.compute_cmb_spectrum(
            structured_contract,
            ells,
            spectra=("TT",),
        )
        self.assertTrue(numpy.all(numpy.isfinite(dl_vals)))
        cmb_df = pandas.DataFrame({"ell": ells, "Dl_obs": dl_vals})
        cmb_df.attrs["covariance_matrix_inv"] = numpy.eye(len(ells))

        started = perf_counter()
        result = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            bao_data_df=bao_df,
            cmb_data_df=cmb_df,
            n_walkers=4,
            n_steps=2,
            pool_size=1,
            burn_in_steps=2,
        )
        self.assertLess(perf_counter() - started, 60.0)
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

    def test_comoving_distance_vectorized(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        params = plugin.INITIAL_GUESSES
        z_vals = numpy.array([0.1, 0.2, 0.3])
        arr = plugin.get_comoving_distance_Mpc(z_vals, *params)
        loop = numpy.array(
            [
                plugin.get_comoving_distance_Mpc(float(z), *params)
                for z in z_vals
            ]
        )
        numpy.testing.assert_allclose(arr, loop)

    def test_near_fixed_bounds_are_flagged(self) -> None:
        logger = logging.getLogger("test.mcmc.bounds")
        bounds = [(1.0, 1.0 + 5e-10), (0.0, 2.0)]
        lower, upper, fixed_mask = _classify_parameter_bounds(
            bounds, logger=logger
        )
        self.assertTrue(fixed_mask[0])
        self.assertFalse(fixed_mask[1])
        self.assertAlmostEqual(lower[0], 1.0)
        self.assertAlmostEqual(upper[0], 1.0 + 5e-10)

    def test_initialise_walkers_relaxes_condition_number(self) -> None:
        initial = numpy.array([5.0, 5.0])
        lower = numpy.array([0.0, 0.0])
        upper = numpy.array([10.0, 10.0])
        rng = numpy.random.default_rng(42)

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
        self.assertTrue(numpy.all(numpy.isfinite(logp_vals)))
        cond = _estimate_condition_number(walkers)
        if cond is not None:
            self.assertLessEqual(cond, 1e12)

    def test_sampler_handles_near_fixed_bounds(self) -> None:
        plugin = _build_model_plugin("model_lcdm.yml")
        tight_value = plugin.INITIAL_GUESSES[0]
        plugin.PARAMETER_BOUNDS = list(plugin.PARAMETER_BOUNDS)
        plugin.PARAMETER_BOUNDS[0] = (
            tight_value - 5e-10,
            tight_value + 5e-10,
        )
        plugin.INITIAL_GUESSES = list(plugin.INITIAL_GUESSES)
        plugin.INITIAL_GUESSES[0] = tight_value

        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )

        result = module.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_walkers=10,
            n_steps=10,
            pool_size=1,
            burn_in_steps=12,
        )
        self.assertTrue(result["success"])
        chain = result["samples"]
        fixed_spread = numpy.ptp(chain[:, :, 0])
        self.assertAlmostEqual(fixed_spread, 0.0, places=10)

    def test_short_chain_returns_none_without_runtime_warning(self) -> None:
        plugin = _build_short_chain_plugin()
        z_values = numpy.linspace(0.01, 0.03, 3)
        baseline = numpy.array([0.3, 0.7])
        mu_model = (
            5.0 * numpy.log10(1.0 + z_values) + baseline[0] + 0.5 * baseline[1]
        )
        sne_df = pandas.DataFrame(
            {
                "zcmb": z_values,
                "mu_obs": mu_model,
                "e_mu_obs": numpy.full(3, 0.1),
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            result = module.fit_cosmology_parameters(
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

    def test_logs_counter_updates_and_notifies_listener(self) -> None:
        events: list[dict[str, object]] = []
        messages: list[str] = []

        def listener(record: dict[str, object]) -> None:
            events.append(record)

        def recorder(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            messages.append(msg)

        progress_bar = BatchProgressBar(
            "Stage",
            total_steps=5,
            display=True,
            progress_listener=listener,
        )
        with mock.patch(
            "copernican.lib.console_output.write",
            side_effect=recorder,
        ):
            progress_bar.start_batch(1, 5)
            updated = progress_bar.update(1, processed=1, total=5)
            progress_bar.finish_batch()

        self.assertIsNotNone(updated)
        self.assertTrue(any("batch 1" in msg for msg in messages))
        self.assertTrue(any("completed" in msg for msg in messages))
        self.assertEqual(
            [record["event"] for record in events],
            ["batch_start", "progress_update", "batch_finish"],
        )

    def test_listener_receives_events_even_when_percent_stalls(self) -> None:
        events: list[dict[str, object]] = []
        progress_bar = BatchProgressBar(
            "Stage",
            total_steps=5,
            display=False,
            progress_listener=lambda record: events.append(record),
        )
        with mock.patch("copernican.lib.console_output.write") as patched:
            progress_bar.start_batch(1, 5)
            progress_bar.update(1, processed=0, total=5)
            progress_bar.update(1, processed=0, total=5)
        self.assertFalse(patched.called)
        self.assertEqual(
            [record["event"] for record in events],
            ["batch_start", "progress_update", "progress_update"],
        )

    def test_finish_batch_emits_completion_event(self) -> None:
        final_event: dict[str, object] | None = None
        captured: list[str] = []

        def listener(record: dict[str, object]) -> None:
            nonlocal final_event
            if record["event"] == "batch_finish":
                final_event = record

        def recorder(
            msg: str = "", *, end: str = "\n", error: bool = False
        ) -> None:
            captured.append(msg)

        progress_bar = BatchProgressBar(
            "Stage",
            total_steps=1,
            display=True,
            progress_listener=listener,
        )
        with mock.patch(
            "copernican.lib.console_output.write",
            side_effect=recorder,
        ):
            progress_bar.start_batch(1, 1)
            progress_bar.finish_batch()

        self.assertTrue(any("complete" in msg for msg in captured))
        self.assertIsNotNone(final_event)
        self.assertEqual(final_event["event"], "batch_finish")


if __name__ == "__main__":
    unittest.main()
