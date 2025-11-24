# Last Updated: 2025-11-24
"""Integration tests for the nested sampling engine."""

import os
import tempfile
import unittest
import warnings
from types import SimpleNamespace
from unittest import mock

import pandas as pd
import xarray as xr

from copernican_lib import (
    chain_io,
    engine_plugin_validation,
    model_coder,
    model_spec_validator,
    run_manifest,
)
from engines import cosmo_engine_nested


def _build_model_plugin(yaml_filename: str):
    """Return a validated plugin for the supplied YAML file."""

    models_dir = os.path.join(os.path.dirname(__file__), "..", "..", "models")
    yaml_path = os.path.join(models_dir, yaml_filename)
    cache_dir = os.path.join(models_dir, "cache")
    cache_path = model_spec_validator.validate_and_cache_model(
        yaml_path, cache_dir
    )
    func_dict, parsed = model_coder.generate_callables(cache_path)
    return engine_plugin_validation.build_plugin(parsed, func_dict)


class TestNestedEngine(unittest.TestCase):
    """Verify that the nested sampler returns expected diagnostics."""

    def _build_lcdm_plugin(self):
        return _build_model_plugin("cosmo_model_lcdm.yml")

    def test_fit_produces_weighted_samples(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        result = cosmo_engine_nested.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_live_points=32,
            max_iterations=120,
            evidence_tolerance=5e-4,
            enlargement_fraction=1.4,
        )
        self.assertTrue(result["success"])
        samples = result["samples"]
        self.assertEqual(samples.ndim, 3)
        self.assertEqual(samples.shape[2], len(plugin.PARAMETER_NAMES))
        self.assertGreater(samples.shape[0], 0)
        self.assertEqual(result["n_live_points"], 32)
        self.assertEqual(result["max_iterations"], 120)
        self.assertAlmostEqual(result["evidence_tolerance"], 5e-4)
        self.assertAlmostEqual(result["enlargement_fraction"], 1.4)
        diagnostics = result.get("diagnostics", {})
        self.assertIn("log_evidence", diagnostics)
        self.assertIn("iterations_completed", diagnostics)
        self.assertLessEqual(diagnostics["iterations_completed"], 120)
        chi2_components = result.get("chi2_components", {})
        total = sum(chi2_components.values())
        self.assertAlmostEqual(result["chi2_total"], total)
        mean_params = result.get("posterior_mean_params", {})
        self.assertSetEqual(
            set(mean_params.keys()), set(plugin.PARAMETER_NAMES)
        )

    def test_chain_serialisation_to_netcdf(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        result = cosmo_engine_nested.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_live_points=24,
            max_iterations=80,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "nested.nc")
            chain_io.save_posterior(
                result["samples"],
                plugin.PARAMETER_NAMES,
                path,
                metadata={"model": plugin.MODEL_NAME},
            )
            try:
                ds = xr.open_dataset(path, group="posterior")
            except ValueError:
                ds = xr.open_dataset(path)
            with ds:
                for name in plugin.PARAMETER_NAMES:
                    self.assertIn(name, ds.data_vars)

    def test_legacy_alias_warns_and_runs(self):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = cosmo_engine_nested.fit_sne_parameters(
                sne_df,
                plugin,
                n_live_points=24,
                max_iterations=60,
                evidence_tolerance=1e-2,
                enlargement_fraction=1.2,
                display_progress=False,
            )
        self.assertTrue(result["success"])
        self.assertTrue(
            any(
                "fit_sne_parameters is deprecated" in str(warning.message)
                for warning in caught
            )
        )

    def test_manifest_integration_records_engine(self):
        plugin = SimpleNamespace(
            MODEL_NAME="Demo",
            MODEL_FILENAME="demo.py",
            PARAMETER_NAMES=("Ωm",),
            PARAMETER_PRIORS=(
                {"type": "uniform", "lower": 0.0, "upper": 1.0},
            ),
        )
        manifest = run_manifest.build_manifest(
            models=[(plugin, "1.0")],
            engine_module=cosmo_engine_nested,
            datasets=[],
        )
        engine_entry = manifest.get("engine", {})
        self.assertEqual(
            engine_entry.get("name"),
            getattr(
                cosmo_engine_nested, "__name__", "engines.cosmo_engine_nested"
            ),
        )
        self.assertEqual(
            engine_entry.get("version"), cosmo_engine_nested.ENGINE_VERSION
        )

    @mock.patch("engines.cosmo_engine_nested.BatchProgressBar")
    def test_progress_bar_initialises_and_updates(self, bar_cls):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        bar_instance = mock.MagicMock()
        bar_cls.return_value = bar_instance

        cosmo_engine_nested.fit_cosmology_parameters(
            sne_df,
            plugin,
            n_live_points=16,
            max_iterations=12,
            display_progress=False,
        )

        bar_cls.assert_called_once()
        _, kwargs = bar_cls.call_args
        self.assertIn("display", kwargs)
        self.assertFalse(kwargs["display"])
        self.assertEqual(kwargs["subunit_labels"], ("iteration", "iterations"))
        bar_instance.start_batch.assert_called_once_with(1, 1)
        self.assertGreaterEqual(bar_instance.update.call_count, 1)
        for call in bar_instance.update.call_args_list:
            args, call_kwargs = call
            self.assertEqual(args[0], 1)
            self.assertIn("step_progress", call_kwargs)
            self.assertLessEqual(call_kwargs["step_progress"], 1.0)
            self.assertGreaterEqual(call_kwargs["step_progress"], 0.0)
            self.assertEqual(call_kwargs["total"], 12)
        bar_instance.finish_batch.assert_called()

    @mock.patch("engines.cosmo_engine_nested.BatchProgressBar")
    def test_progress_bar_finishes_on_exception(self, bar_cls):
        plugin = self._build_lcdm_plugin()
        sne_df = pd.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        bar_instance = mock.MagicMock()
        bar_cls.return_value = bar_instance

        with mock.patch(
            "engines.cosmo_engine_nested._replacement_sample",
            side_effect=RuntimeError("replacement failure"),
        ):
            with self.assertRaises(RuntimeError):
                cosmo_engine_nested.fit_cosmology_parameters(
                    sne_df,
                    plugin,
                    n_live_points=8,
                    max_iterations=5,
                )

        bar_instance.finish_batch.assert_called()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
