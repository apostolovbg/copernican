"""Tests for the GUI run worker that now delegates to the manifest CLI."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from copernican_lib.gui import run_worker


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
            }
            config_path = Path(tmpdir) / "worker_config.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            called: list[list[str]] = []

            def stub_main(argv):
                called.append(argv)
                return 0

            with mock.patch.object(copernican, "main", stub_main):
                exit_code = run_worker.main([str(config_path)])

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


if __name__ == "__main__":
    unittest.main()
