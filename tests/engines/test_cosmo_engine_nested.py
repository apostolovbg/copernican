"""Behavior tests for copernican.engines.cosmo_engine_nested."""

import os
import tempfile
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas
import xarray as xarray_dataset

from copernican.engines import cosmo_engine_nested as module
from copernican.lib import chain_io
from copernican.lib import engine_adapter as engine_plugin_validation
from copernican.lib import model_coder, model_spec_validator, run_manifest


class _DummyJointLike:
    """Minimal joint-like object for adapter behavior checks."""

    def __init__(self) -> None:
        self.calls: list[tuple[float, ...]] = []

    def loglike(self, params):
        self.calls.append(tuple(params))
        return sum(params) + 0.5


def _build_model_plugin(yaml_filename: str):
    """Return a validated plugin for the supplied YAML file."""

    models_dir = Path(__file__).resolve().parents[2] / "copernican" / "models"
    yaml_path = models_dir / yaml_filename
    cache_dir = models_dir / "cache"
    cache_path = model_spec_validator.validate_and_cache_model(
        yaml_path, cache_dir
    )
    func_dict, parsed = model_coder.generate_callables(cache_path)
    return engine_plugin_validation.build_plugin(parsed, func_dict)


class TestCosmoEngineNested(unittest.TestCase):
    """Exercise the reusable helpers and the nested engine workflow."""

    def test_engine_metadata(self) -> None:
        self.assertEqual(module.ENGINE_KIND, "nested")
        self.assertEqual(module.ENGINE_LABEL, "Nested sampling engine")
        self.assertEqual(module.ENGINE_VERSION, "1.1.0")
        self.assertTrue(module.ENGINE_SETTINGS)
        self.assertTrue(module.ENGINE_PROGRESS_CHUNKS)

    def test_logsumexp_pair_handles_finite_and_infinite_inputs(self) -> None:
        self.assertAlmostEqual(
            module._logsumexp_pair(0.0, 0.0),
            0.6931471805599453,
        )
        self.assertEqual(
            module._logsumexp_pair(float("-inf"), 3.0),
            3.0,
        )

    def test_joint_log_likelihood_delegates_to_inner_like(self) -> None:
        joint_like = _DummyJointLike()
        adapter = module._JointLogLikelihood(
            joint_like,
            parameter_bounds=[(0.0, 1.0)],
            parameter_transforms=["transform"],
        )

        self.assertEqual(adapter.parameter_bounds, [(0.0, 1.0)])
        self.assertEqual(adapter.parameter_transforms, ["transform"])
        self.assertEqual(adapter([1.0, 2.0]), 3.5)
        self.assertEqual(joint_like.calls, [(1.0, 2.0)])

    def test_fit_produces_weighted_samples(self) -> None:
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        result = module.fit_cosmology_parameters(
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

    def test_chain_serialisation_to_netcdf(self) -> None:
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
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
                dataset_reader = xarray_dataset.open_dataset(
                    path, group="posterior"
                )
            except ValueError:
                dataset_reader = xarray_dataset.open_dataset(path)
            with dataset_reader:
                for name in plugin.PARAMETER_NAMES:
                    self.assertIn(name, dataset_reader.data_vars)

    def test_legacy_alias_warns_and_runs(self) -> None:
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02],
                "mu_obs": [40.0, 41.0],
                "e_mu_obs": [0.1, 0.1],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = module.fit_sne_parameters(
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

    def test_manifest_integration_records_engine(self) -> None:
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
            engine_module=module,
            datasets=[],
        )
        engine_entry = manifest.get("engine", {})
        self.assertEqual(
            engine_entry.get("name"),
            getattr(
                module, "__name__", "copernican.engines.cosmo_engine_nested"
            ),
        )
        self.assertEqual(engine_entry.get("version"), module.ENGINE_VERSION)

    @mock.patch("copernican.engines.cosmo_engine_nested.BatchProgressBar")
    def test_progress_bar_initialises_and_updates(self, bar_cls) -> None:
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
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

    @mock.patch("copernican.engines.cosmo_engine_nested.BatchProgressBar")
    def test_progress_bar_finishes_on_exception(self, bar_cls) -> None:
        plugin = _build_model_plugin("cosmo_model_lcdm.yml")
        sne_df = pandas.DataFrame(
            {
                "zcmb": [0.01, 0.02, 0.03],
                "mu_obs": [40.0, 41.0, 42.0],
                "e_mu_obs": [0.1, 0.1, 0.1],
            }
        )
        bar_instance = mock.MagicMock()
        bar_cls.return_value = bar_instance

        with mock.patch(
            "copernican.engines.cosmo_engine_nested._replacement_sample",
            side_effect=RuntimeError("replacement failure"),
        ):
            with self.assertRaises(RuntimeError):
                module.fit_cosmology_parameters(
                    sne_df,
                    plugin,
                    n_live_points=8,
                    max_iterations=5,
                )

        bar_instance.finish_batch.assert_called()


if __name__ == "__main__":  # pragma: no cover - manual invocation
    unittest.main()
