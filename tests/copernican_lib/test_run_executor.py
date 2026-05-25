"""Tests for the run executor."""

import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from copernican_lib import run_executor


class FakeDataset:
    """Minimal dataset object with a stable length."""

    def __len__(self):
        return 5


class TestRunExecutor(unittest.TestCase):
    """Exercise manifest execution and persistence behavior."""

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
            "copernican_lib.run_executor",
        )

    def _base_manifest(self, seed: int = 123):
        return {
            "seed": seed,
            "selection": {
                "models": ["LambdaCDM"],
                "engine": {
                    "name": "engines.cosmo_engine_mcmc",
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
            "configuration": {"run_settings": {"engine_kind": "mcmc"}},
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
        self.assertEqual(sampling_plan["engine_kind"], "mcmc")
        self.assertTrue(pipeline_calls[0]["display_progress"])

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


if __name__ == "__main__":
    unittest.main()
