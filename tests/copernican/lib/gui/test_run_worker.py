"""Tests for the GUI run worker that delegates to the manifest CLI."""

import builtins
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican.lib.gui import run_worker


class TestGuiRunWorker(unittest.TestCase):
    """Exercise manifest-path forwarding into the CLI entrypoint."""

    def test_worker_main_passes_manifest_path(self) -> None:
        os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")
        import copernican

        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "run_manifest.yml"
            manifest.write_text("seed: 0\n", encoding="utf-8")
            config = {
                "manifest_path": str(manifest),
                "output_dir": str(Path(tmpdir) / "output"),
                "run_start_ts": "20260815_120000",
                "log_prefix": "copernican-run",
                "log_name": "copernican-run_20260815_120000.txt",
            }
            config_path = Path(tmpdir) / "worker_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            called: list[list[str]] = []

            def stub_main(argv):
                called.append(argv)
                return 0

            with (
                mock.patch.object(copernican, "main", stub_main),
                mock.patch.object(
                    run_worker.log_mod, "setup_logging"
                ) as setup_logging,
                mock.patch.dict(os.environ, {}, clear=False),
            ):
                exit_code = run_worker.main([str(config_path)])
                self.assertEqual(
                    os.environ["COPERNICAN_RUN_START_TS"],
                    config["run_start_ts"],
                )
                self.assertEqual(
                    os.environ["COPERNICAN_RUN_LOG_PREFIX"],
                    config["log_prefix"],
                )
                self.assertEqual(
                    os.environ["COPERNICAN_GUI_EVENT_STREAM"], "1"
                )

            self.assertEqual(exit_code, 0)
            self.assertEqual(
                called,
                [
                    [
                        "--manifest",
                        str(manifest),
                        "--output-dir",
                        config["output_dir"],
                    ]
                ],
            )
            setup_logging.assert_called_once_with(
                log_dir=config["output_dir"],
                base_dir=str(Path(run_worker.__file__).resolve().parents[3]),
                log_tag=config["log_name"],
            )

    def test_worker_rejects_inconsistent_log_identity(self) -> None:
        """The child must not invent a second timestamp or log name."""

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "worker_config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest_path": str(Path(tmpdir) / "manifest.yml"),
                        "output_dir": tmpdir,
                        "run_start_ts": "20260815_120000",
                        "log_prefix": "copernican-run",
                        "log_name": "copernican-run_other.txt",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(run_worker.main([str(config_path)]), 1)

    def test_worker_owns_one_canonical_log(self) -> None:
        """A worker event must reach one file once at its original level."""

        os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")
        import copernican

        original_print = builtins.print
        original_input = builtins.input
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "run_manifest.yml"
            manifest.write_text("seed: 0\n", encoding="utf-8")
            output_dir = Path(tmpdir) / "output"
            config = {
                "manifest_path": str(manifest),
                "output_dir": str(output_dir),
                "run_start_ts": "20260815_120000",
                "log_prefix": "copernican-run",
                "log_name": "copernican-run_20260815_120000.txt",
            }
            config_path = Path(tmpdir) / "worker_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")

            def stub_main(_argv):
                logging.getLogger().warning("worker warning")
                run_worker.log_mod.setup_logging(
                    log_dir=config["output_dir"],
                    base_dir=str(
                        Path(run_worker.__file__).resolve().parents[3]
                    ),
                    log_tag=config["log_name"],
                )
                return 0

            try:
                with (
                    mock.patch.object(copernican, "main", stub_main),
                    mock.patch.dict(os.environ, {}, clear=False),
                ):
                    self.assertEqual(run_worker.main([str(config_path)]), 0)
                run_worker.log_mod._close_handlers(logging.getLogger())
                logs = list(output_dir.glob("copernican-run_*.txt"))
                self.assertEqual(len(logs), 1)
                content = logs[0].read_text(encoding="utf-8")
                self.assertEqual(content.count("worker warning"), 1)
                self.assertIn(" - WARNING - worker warning", content)
                self.assertEqual(
                    content.count("Logging initialized with UTC"), 1
                )
            finally:
                run_worker.log_mod._close_handlers(logging.getLogger())
                builtins.print = original_print
                builtins.input = original_input


if __name__ == "__main__":
    unittest.main()
