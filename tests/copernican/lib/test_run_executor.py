"""Tests for the run executor."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml

from copernican.lib import run_executor
from copernican.lib.likelihoods.cmb.errors import CMBError, ConvergenceError
from tests.copernican.lib.test_run_pipeline import (
    bounded_pipeline_inputs,
    patch_unrelated_pipeline_plots,
)
from tests.project.lib.cmb_solver_fixture import (
    SyntheticCmbSolver,
    explicit_cmb_observations,
)


class FakeDataset:
    """Minimal dataset object with a stable length."""

    def __len__(self):
        return 5


class TestRunExecutor(unittest.TestCase):
    """Exercise manifest execution and persistence behavior."""

    def test_manifest_runs_real_pipeline_and_separates_failure_provenance(
        self,
    ):
        """Pipeline completion and early numerical failure stay distinct."""

        for fail in (False, True):
            with self.subTest(fail=fail):
                control, reference, sne, comparison, plan = (
                    bounded_pipeline_inputs()
                )
                manifest = self._base_manifest()
                manifest["selection"]["models"] = [
                    "model_lcdm",
                    "model_ref_planck2018",
                ]
                manifest["selection"]["comparison"]["test"] = {
                    "name": "model_ref_planck2018",
                    "filename": reference.MODEL_FILENAME,
                }
                manifest["datasets"] = {
                    "sne/small": {
                        "name": "Small SNe",
                        "type": "sne",
                        "version": "1",
                    },
                    "cmb/small": {
                        "name": "Small CMB",
                        "type": "cmb",
                        "version": "1",
                    },
                }
                manifest["configuration"]["run_settings"].update(plan)
                manifest["configuration"]["run_settings"][
                    "display_progress"
                ] = False
                solver = SyntheticCmbSolver(
                    spectrum_offsets={"TE": -1205, "EE": -1190}
                )
                with (
                    tempfile.TemporaryDirectory() as directory,
                    contextlib.ExitStack() as stack,
                ):
                    patch_unrelated_pipeline_plots(stack)
                    stack.enter_context(
                        mock.patch.object(
                            run_executor.run_pipeline.plotter,
                            "plot_cmb_spectrum",
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            run_executor.dataset_registry,
                            "load_sne_data",
                            return_value=sne,
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            run_executor.dataset_registry,
                            "load_cmb_data",
                            return_value=explicit_cmb_observations(),
                        )
                    )
                    plugins = {
                        control.MODEL_FILENAME: control,
                        reference.MODEL_FILENAME: reference,
                    }
                    stack.enter_context(
                        mock.patch.object(
                            run_executor,
                            "_build_plugin_from_path",
                            side_effect=lambda path: plugins[path.name],
                        )
                    )
                    stack.enter_context(
                        mock.patch.object(
                            run_executor,
                            "resolve_cmb_solver",
                            return_value=solver,
                        )
                    )
                    run_executor._PLUGIN_CACHE.clear()
                    if fail:
                        stack.enter_context(
                            mock.patch.object(
                                solver,
                                "evaluate",
                                side_effect=ConvergenceError(
                                    "manufactured incomplete computation"
                                ),
                            )
                        )
                        with self.assertRaises(CMBError):
                            run_executor.execute_run_from_manifest(
                                manifest,
                                script_dir=Path.cwd(),
                                output_root=Path(directory),
                                run_start_ts="20261018_000001",
                            )
                    else:
                        run_executor.execute_run_from_manifest(
                            manifest,
                            script_dir=Path.cwd(),
                            output_root=Path(directory),
                            run_start_ts="20261018_000001",
                        )
                    (manifest_path,) = Path(directory).glob(
                        "run_manifest_*.yml"
                    )
                    stored = yaml.safe_load(manifest_path.read_text())
                    self.assertEqual(
                        stored["selection"]["comparison"],
                        comparison.as_manifest(),
                    )
                    if fail:
                        self.assertNotIn(
                            "sampling", stored.get("provenance", {})
                        )
                        self.assertFalse(
                            list(
                                Path(directory).glob("parameter-summary*.json")
                            )
                        )
                    else:
                        self.assertEqual(
                            stored["provenance"]["cmb_solver"]["solver_id"],
                            solver.solver_id,
                        )
                        self.assertEqual(
                            set(stored["provenance"]["sampling"]),
                            {"control", "test"},
                        )
                        self.assertTrue(
                            list(
                                Path(directory).glob("parameter-summary*.json")
                            )
                        )
                        self.assertTrue(
                            list(Path(directory).glob("cmb-data*.csv"))
                        )

    def setUp(self) -> None:
        self.console_write_patch = mock.patch.object(
            run_executor.console_output,
            "write",
            lambda *_: None,
        )
        self.console_write_patch.start()
        self.addCleanup(self.console_write_patch.stop)

    def test_execute_run_from_manifest_symbol_is_exported(self) -> None:
        self.assertTrue(callable(run_executor.execute_run_from_manifest))
        self.assertEqual(
            run_executor.execute_run_from_manifest.__module__,
            "copernican.lib.run_executor",
        )

    def _base_manifest(self, seed: int = 123):
        return {
            "seed": seed,
            "selection": {
                "models": ["model_lcdm", "model_wcdm"],
                "comparison": {
                    "control": {
                        "name": "model_lcdm",
                        "filename": "model_lcdm.yml",
                    },
                    "test": {
                        "name": "model_wcdm",
                        "filename": "model_wcdm.yml",
                    },
                },
                "sampler": {
                    "name": "copernican.samplers.sampler_mcmc",
                    "version": "7.6.20",
                },
            },
            "datasets": {
                "sne/pantheon": {
                    "name": "Pantheon",
                    "type": "sne",
                    "version": "1.0",
                    "path": tempfile.gettempdir(),
                },
                "bao/bossdr12": {
                    "name": "BOSS DR12",
                    "type": "bao",
                    "version": "1.0",
                    "path": tempfile.gettempdir(),
                },
            },
            "configuration": {"run_settings": {"sampler_kind": "mcmc"}},
        }

    def _enter_common_patches(self, stack: contextlib.ExitStack) -> None:
        stack.enter_context(
            mock.patch.object(
                run_executor.dataset_registry,
                "load_sne_data",
                lambda **kwargs: FakeDataset(),
            )
        )
        stack.enter_context(
            mock.patch.object(
                run_executor.dataset_registry,
                "load_bao_data",
                lambda **kwargs: FakeDataset(),
            )
        )
        stack.enter_context(
            mock.patch.object(
                run_executor,
                "_build_plugin_from_path",
                lambda path: SimpleNamespace(
                    MODEL_NAME=path.stem,
                    MODEL_FILENAME=path.name,
                ),
            )
        )

    def test_execute_run_from_manifest_loads_datasets(self) -> None:
        manifest = self._base_manifest()
        loaded: list[bool] = []
        progress_records = []
        pipeline_calls = []

        def fake_loader():
            loaded.append(True)
            return FakeDataset()

        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.ExitStack() as stack:
                self._enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        run_executor.dataset_registry,
                        "load_sne_data",
                        lambda **kwargs: fake_loader(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_executor.dataset_registry,
                        "load_bao_data",
                        lambda **kwargs: fake_loader(),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_executor.run_pipeline,
                        "execute_run_pipeline",
                        lambda **kwargs: pipeline_calls.append(kwargs),
                    )
                )
                run_executor._PLUGIN_CACHE.clear()
                run_executor.execute_run_from_manifest(
                    manifest,
                    script_dir=Path("."),
                    output_root=Path(tmpdir),
                    progress_callback=lambda record: progress_records.append(
                        record
                    ),
                )

        self.assertEqual(len(loaded), 2)
        self.assertTrue(progress_records)
        self.assertEqual(
            progress_records[0]["status"], "manifest_execution_started"
        )
        self.assertTrue(pipeline_calls)
        sampling_plan = pipeline_calls[0]["sampling_plan"]
        self.assertEqual(sampling_plan["sampler_kind"], "mcmc")
        self.assertTrue(pipeline_calls[0]["display_progress"])

    def test_execute_run_from_manifest_accepts_external_yaml_model(
        self,
    ) -> None:
        manifest = self._base_manifest()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "external_model.yaml"
            model_path.write_text(
                "model_name: ExternalModel\nparameters: []\n",
                encoding="utf-8",
            )
            manifest["selection"]["models"][1] = "external_model"
            manifest["selection"]["comparison"]["test"] = {
                "name": "external_model",
                "filename": str(model_path),
            }
            pipeline_calls = []
            with contextlib.ExitStack() as stack:
                self._enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        run_executor.run_pipeline,
                        "execute_run_pipeline",
                        lambda **kwargs: pipeline_calls.append(kwargs),
                    )
                )
                run_executor._PLUGIN_CACHE.clear()
                run_executor.execute_run_from_manifest(
                    manifest,
                    script_dir=Path("."),
                    output_root=Path(tmpdir),
                )

        self.assertTrue(pipeline_calls)
        test_model = pipeline_calls[0]["test_model_plugin"]
        self.assertEqual(test_model.MODEL_FILENAME, "external_model.yaml")
        control_model = pipeline_calls[0]["control_model_plugin"]
        self.assertEqual(control_model.MODEL_FILENAME, "model_lcdm.yml")
        comparison = pipeline_calls[0]["comparison"]
        self.assertEqual(comparison.control_model.name, "model_lcdm")
        self.assertEqual(comparison.test_model.name, "external_model")
        self.assertEqual(
            comparison.test_model.filename,
            "external_model.yaml",
        )

    def test_execute_run_from_manifest_persists_manifest(self) -> None:
        manifest = self._base_manifest(seed=999)
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.ExitStack() as stack:
                self._enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        run_executor.run_pipeline,
                        "execute_run_pipeline",
                        lambda **kwargs: ({}, {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_executor.utils,
                        "get_timestamp",
                        lambda: "20250101_000000",
                    )
                )
                run_executor._PLUGIN_CACHE.clear()
                run_executor.execute_run_from_manifest(
                    manifest,
                    script_dir=Path("."),
                    output_root=Path(tmpdir),
                )

            manifest_path = Path(tmpdir) / "run_manifest_20250101_000000.yml"
            self.assertTrue(manifest_path.exists())
            content = manifest_path.read_text(encoding="utf-8")
            self.assertIn("seed: 999", content)

    def test_execute_run_from_manifest_sets_seed(self) -> None:
        manifest = self._base_manifest()
        seed_calls: list[int] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            with contextlib.ExitStack() as stack:
                self._enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        run_executor.run_pipeline,
                        "execute_run_pipeline",
                        lambda **kwargs: ({}, {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_executor.utils,
                        "set_random_seed",
                        lambda value: seed_calls.append(value),
                    )
                )
                run_executor._PLUGIN_CACHE.clear()
                run_executor.execute_run_from_manifest(
                    manifest,
                    script_dir=Path("."),
                    output_root=Path(tmpdir),
                )

        self.assertEqual(seed_calls, [123])

    def test_manifest_run_has_one_canonical_external_log(self) -> None:
        """The executor must retain one absolute output and event stream."""

        manifest = self._base_manifest()
        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir).resolve()
            with contextlib.ExitStack() as stack:
                self._enter_common_patches(stack)
                stack.enter_context(
                    mock.patch.object(
                        run_executor.run_pipeline,
                        "execute_run_pipeline",
                        lambda **kwargs: ({}, {}),
                    )
                )
                stack.enter_context(
                    mock.patch.object(
                        run_executor.console_output,
                        "write",
                        lambda message, **_kwargs: (
                            run_executor.log_mod.get_logger().info(message)
                        ),
                    )
                )
                run_executor._PLUGIN_CACHE.clear()
                run_executor.execute_run_from_manifest(
                    manifest,
                    script_dir=Path("/repository/source"),
                    output_root=output_root,
                    run_start_ts="20260815_120000",
                )

            logs = list(output_root.glob("copernican-run_*.txt"))
            self.assertEqual(len(logs), 1)
            content = logs[0].read_text(encoding="utf-8")
            output_message = f"Output directory: {output_root}"
            self.assertEqual(content.count(output_message), 1)
            self.assertNotIn("Output directory: .", content)
            self.assertEqual(content.count("Run confirmed with manifest:"), 1)
            self.assertEqual(
                content.count("Dataset sne/pantheon (sne): Pantheon"), 1
            )


if __name__ == "__main__":
    unittest.main()
