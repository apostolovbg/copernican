"""Smoke and CLI utility tests for the top-level `copernican.py` entrypoint."""

import importlib
import os
import runpy
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy
import xarray as xarray_dataset
import yaml

os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")

import copernican  # noqa: E402


def _create_run_dir(tmp_path, name, chi2_total, rows, h0_value):
    run_dir = tmp_path / name
    run_dir.mkdir()
    manifest = {
        "datasets": {"union3_2025": {"name": "Union sample", "type": "sne"}}
    }
    (run_dir / "run_manifest_20250101.yml").write_text(
        yaml.safe_dump(manifest), encoding="utf-8"
    )
    summary = {
        "LambdaCDM": {
            "parameters": {"H_0": h0_value},
            "errors_1sigma": {"H_0": 0.4},
            "covariance_matrix": {"param_names": ["H_0"], "matrix": [[0.16]]},
            "sampling": {"production_steps": 200},
        }
    }
    (run_dir / "parameter-summary_20250101.yml").write_text(
        yaml.safe_dump(summary), encoding="utf-8"
    )
    log_lines = "\n".join(
        [
            "2025-12-08 01:09:21,563 - INFO - --- ΛCDM Fit Report ---",
            f"2025-12-08 01:09:21,564 - INFO -   χ²_Total = {chi2_total}",
            "2025-12-08 01:09:21,565 - INFO -   χ²_BAO = 4.50",
            (
                "2025-12-08 01:09:21,566 - INFO - LambdaCDM BAO: "
                "r_s = 146.12 Mpc, χ²_BAO = 4.50"
            ),
            (
                "2025-12-08 01:09:21,566 - INFO - Loaded dataset "
                f"union3_2025: {rows} entries"
            ),
            "2025-12-08 01:09:21,568 - INFO - Evaluation complete.",
        ]
    )
    (run_dir / "copernican-run_20250101.txt").write_text(
        log_lines + "\n", encoding="utf-8"
    )
    return run_dir


def _add_posterior_file(run_dir):
    dataset = xarray_dataset.Dataset(
        {"H0": (("draw",), numpy.linspace(65, 70, 10))}
    )
    posterior_path = run_dir / "posterior-0001.nc"
    dataset.to_netcdf(posterior_path)
    return posterior_path


class TestEntrypoint(unittest.TestCase):
    """Exercise the top-level script without importing it as a module."""

    def test_entrypoint_exits_cleanly(self) -> None:
        script = Path(__file__).resolve().parents[1] / "copernican.py"
        with self.assertRaises(SystemExit) as caught:
            runpy.run_path(script, run_name="__main__")
        self.assertEqual(caught.exception.code, 1)


class TestCliUtilities(unittest.TestCase):
    """Exercise the CLI-facing helper commands."""

    def test_entrypoint_symbols_are_present(self):
        self.assertTrue(callable(copernican.exit_clean))
        self.assertTrue(callable(copernican.launch_gui))
        self.assertTrue(hasattr(copernican, "LaunchRequest"))
        self.assertTrue(callable(copernican.main_workflow))
        self.assertTrue(callable(copernican.main))

    @mock.patch.object(copernican.console, "write")
    def test_exit_clean_raises_requested_system_exit(self, write):
        with self.assertRaises(SystemExit) as caught:
            copernican.exit_clean(7)
        self.assertEqual(caught.exception.code, 7)
        write.assert_called_once_with("")

    @mock.patch("copernican_lib.gui.CopernicanGUI")
    @mock.patch.object(
        copernican.orchestration, "describe_orchestration_services"
    )
    @mock.patch.object(copernican, "_ensure_program_logging")
    @mock.patch.object(copernican.console, "write")
    def test_launch_gui_runs_gui_shell(
        self,
        write,
        ensure_program_logging,
        describe_orchestration_services,
        gui_class,
    ):
        logger = mock.Mock()
        ensure_program_logging.return_value = logger
        describe_orchestration_services.return_value = SimpleNamespace(
            config_validation=SimpleNamespace(
                name="Validation",
                module="validation.module",
                entrypoints=("validate",),
                rationale="validate",
            ),
            manifest_generation=SimpleNamespace(
                name="Manifest",
                module="manifest.module",
                entrypoints=("build",),
                rationale="build",
            ),
            run_control=SimpleNamespace(
                name="Run",
                module="run.module",
                entrypoints=("run",),
                rationale="run",
            ),
        )
        gui = gui_class.return_value

        copernican.launch_gui()

        gui_class.assert_called_once_with(render=True)
        gui.show_home.assert_called_once_with()
        gui.run.assert_called_once_with()
        self.assertGreaterEqual(write.call_count, 1)

    @mock.patch.object(copernican, "_build_gui_progress_callback")
    @mock.patch.object(copernican, "_ensure_program_logging")
    @mock.patch.object(copernican.cli_dependencies, "load_third_party_modules")
    @mock.patch.object(copernican.cli_dependencies, "get_runtime_options")
    @mock.patch("copernican_lib.run_executor.execute_run_from_manifest")
    @mock.patch("copernican_lib.run_manifest.load_manifest")
    def test_main_workflow_executes_manifest_pipeline(
        self,
        load_manifest,
        execute_run_from_manifest,
        get_runtime_options,
        load_third_party_modules,
        ensure_program_logging,
        build_gui_progress_callback,
    ):
        load_manifest.return_value = {"name": "example"}
        get_runtime_options.return_value = SimpleNamespace(
            run_tests=False,
            strict_warnings=False,
        )
        load_third_party_modules.return_value = (None, None, None)
        ensure_program_logging.return_value = mock.Mock()
        build_gui_progress_callback.return_value = None
        launch_dir = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: launch_dir.exists() and launch_dir.rmdir())
        with mock.patch.object(
            copernican,
            "_launch_args",
            SimpleNamespace(output_dir=launch_dir),
        ):
            copernican.main_workflow(manifest_path=launch_dir / "manifest.yml")
        load_manifest.assert_called_once()
        execute_run_from_manifest.assert_called_once()

    @mock.patch.object(copernican, "_announce_program_start")
    @mock.patch.object(copernican, "_handle_auxiliary_requests")
    @mock.patch.object(copernican, "_ensure_program_logging")
    @mock.patch.object(copernican, "_parse_launch_args")
    @mock.patch.object(copernican, "_run_cli_launch", return_value=0)
    def test_main_routes_to_cli_workflow(
        self,
        run_cli_launch,
        parse_launch_args,
        ensure_program_logging,
        handle_auxiliary_requests,
        announce_program_start,
    ):
        parse_launch_args.return_value = SimpleNamespace(
            mode=copernican.orchestration.LaunchMode.CLI,
            manifest_path=Path("manifest.yml"),
            detach_gui=False,
        )
        ensure_program_logging.return_value = mock.Mock()
        handle_auxiliary_requests.return_value = (False, 0)
        result = copernican.main([])
        self.assertEqual(result, 0)
        run_cli_launch.assert_called_once()
        announce_program_start.assert_called_once()

    def test_catalogue_summary_reports_counts(self):
        summary = copernican._gather_catalogue_summary()
        self.assertGreater(summary["dataset_count"], 0)
        self.assertTrue(summary["type_counter"])

    def test_model_engine_summary_reports_counts(self):
        stats = copernican._gather_model_engine_summary()
        self.assertGreater(stats["model_count"], 0)
        self.assertGreater(stats["engine_count"], 0)

    def test_manifest_discovery_sorts_by_mtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "copernican-run_20240101_000000"
            second = Path(tmpdir) / "copernican-run_20240102_000000"
            first.mkdir()
            second.mkdir()
            manifest_first = first / "run_manifest_20240101.yml"
            manifest_second = second / "run_manifest_20240102.yml"
            manifest_first.write_text("seed: 0\n", encoding="utf-8")
            manifest_second.write_text("seed: 1\n", encoding="utf-8")
            os.utime(manifest_first, (time.time() - 100, time.time() - 100))
            os.utime(manifest_second, (time.time(), time.time()))
            records = copernican._discover_manifest_records(Path(tmpdir))
            self.assertEqual(
                [directory.name for directory, _ in records],
                [second.name, first.name],
            )

    def test_cli_revalidate_dataset_reports_missing(self):
        self.assertFalse(copernican._cli_revalidate_dataset("missing-dataset"))

    def test_cli_revalidate_dataset_known_dataset(self):
        self.assertTrue(copernican._cli_revalidate_dataset("planck_2018_lite"))

    def test_analysis_summary_cli_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _create_run_dir(
                Path(tmpdir), "analysis-summary", 360.11, 5, 67.2
            )
            output_dir = Path(tmpdir) / "analysis-summary-output"
            summary_exported = copernican._run_analysis_summary_cli(
                run_dir,
                output_dir,
                ("yml",),
            )
            self.assertTrue(summary_exported)
            summary_files = list(output_dir.glob("analysis-summary_*.yml"))
            self.assertTrue(summary_files)

    def test_analysis_compare_cli_exports(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = _create_run_dir(Path(tmpdir), "base", 360.11, 5, 67.2)
            alt_dir = _create_run_dir(Path(tmpdir), "alt", 362.22, 4, 67.8)
            output_dir = Path(tmpdir) / "comparison-output"
            comparison_exported = copernican._run_analysis_compare_cli(
                base_dir,
                alt_dir,
                output_dir,
                ("yml",),
            )
            self.assertTrue(comparison_exported)
            comparison_files = list(
                output_dir.glob("analysis-comparison_*.yml")
            )
            self.assertTrue(comparison_files)

    def test_analysis_posterior_cli_creates_plot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = _create_run_dir(
                Path(tmpdir), "posterior", 360.11, 5, 67.2
            )
            _add_posterior_file(run_dir)
            output_file = Path(tmpdir) / "posterior.png"
            posterior_exported = copernican._run_analysis_posterior_cli(
                run_dir,
                None,
                output_file,
            )
            self.assertTrue(posterior_exported)
            self.assertTrue(output_file.exists())
            corner_files = list(Path(tmpdir).glob("corner-plot-*.png"))
            hist_files = list(Path(tmpdir).glob("parameter-histograms-*.png"))
            self.assertTrue(corner_files)
            self.assertTrue(hist_files)


class LaunchArgParsingTestCase(unittest.TestCase):
    """Ensure launcher arguments map cleanly onto orchestration modes."""

    def setUp(self) -> None:
        self.venv_path = str(Path(__file__).resolve().parents[1] / ".venv")
        self.settings_path = (
            Path(tempfile.gettempdir())
            / f"copernican_settings_test_{uuid.uuid4().hex}.yml"
        )
        self.env_patch = mock.patch.dict(
            os.environ,
            {
                "VIRTUAL_ENV": self.venv_path,
                "COPERNICAN_SETTINGS_PATH": str(self.settings_path),
            },
            clear=True,
        )
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.copernican = importlib.reload(
            importlib.import_module("copernican")
        )

    def tearDown(self) -> None:
        if self.settings_path.exists():
            self.settings_path.unlink(missing_ok=True)

    def test_default_mode_prefers_cli(self) -> None:
        args = self.copernican._parse_launch_args([])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.CLI
        )

    def test_gui_flag_switches_mode(self) -> None:
        args = self.copernican._parse_launch_args(["--gui"])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.GUI
        )

    def test_no_gui_flag_forces_cli(self) -> None:
        args = self.copernican._parse_launch_args(["--no-gui"])
        self.assertEqual(
            args.mode, self.copernican.orchestration.LaunchMode.CLI
        )

    def test_manifest_and_output_dir_paths_resolve(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = Path(tmpdir) / "run" / "manifest.yml"
            output_dir = Path(tmpdir) / "output"
            args = self.copernican._parse_launch_args(
                [
                    "--manifest",
                    str(manifest_path),
                    "--output-dir",
                    str(output_dir),
                ]
            )
        self.assertEqual(args.manifest_path, manifest_path.resolve())
        self.assertEqual(args.output_dir, output_dir.resolve())

    def test_detach_flag_respects_environment_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"VIRTUAL_ENV": self.venv_path, "COPERNICAN_DETACH_GUI": "0"},
            clear=True,
        ):
            args = self.copernican._parse_launch_args(["--gui"])
        self.assertFalse(args.detach_gui)


if __name__ == "__main__":
    unittest.main()
