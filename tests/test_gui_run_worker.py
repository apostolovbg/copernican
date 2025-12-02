"""Tests for the GUI run worker that now delegates to the manifest CLI."""

import json
import os

from copernican_lib.gui import run_worker


def test_worker_main_passes_manifest_path(tmp_path, monkeypatch):
    os.environ.setdefault("COPERNICAN_ALLOW_DIRECT", "1")
    import copernican

    manifest = tmp_path / "run_manifest.yml"
    manifest.write_text("seed: 0\n")
    config = {
        "manifest_path": str(manifest),
        "output_dir": str(tmp_path / "output"),
    }
    config_path = tmp_path / "worker_config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    called = []

    def stub_main(argv):
        called.append(argv)
        return 0

    monkeypatch.setattr(copernican, "main", stub_main)
    exit_code = run_worker.main([str(config_path)])
    assert exit_code == 0
    assert called == [
        ["--manifest", str(manifest), "--output-dir", config["output_dir"]]
    ]
