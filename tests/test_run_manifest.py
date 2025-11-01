"""Tests for the run manifest helper.

**Last Updated:** 2025-11-01
"""

import os
import tempfile
from types import SimpleNamespace

import yaml

from copernican_lib import run_manifest, utils
from copernican_lib.version import get_version


def _dummy_plugin():
    return SimpleNamespace(
        MODEL_NAME="DummyModel",
        MODEL_FILENAME="dummy.yml",
        PARAMETER_NAMES=["p1"],
        PARAMETER_PRIORS=[{"type": "uniform", "lower": 0, "upper": 1}],
        valid_for_cmb=True,
        CMB_PARAM_MAP={"H0": "p1", "ombh2": 0.022, "omch2": 0.12, "Neff": 3.044},
    )


def test_manifest_contains_required_fields():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.txt")
        with open(data_path, "w", encoding="utf-8") as fh:
            fh.write("hello world\n")
        engine = SimpleNamespace(__name__="engine", ENGINE_VERSION="0.0")
        file_hashes = {"data.txt": utils.compute_sha256(data_path)}
        utils.set_random_seed(123)
        manifest = run_manifest.build_manifest(
            models=[(_dummy_plugin(), "1.0")],
            engine_module=engine,
            datasets=[
                {
                    "id": "ds",
                    "name": "Dummy dataset",
                    "version": "2025.10",
                    "path": tmpdir,
                    "hashes": file_hashes,
                    "independence": "Assumed independent test input",
                }
            ],
        )
        path = run_manifest.save_manifest(manifest, tmpdir)
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["copernican"]["version"] == get_version()
        assert loaded["engine"]["name"] == "engine"
        assert loaded["seed"] == 123
        assert "ds" in loaded["datasets"]
        ds_entry = loaded["datasets"]["ds"]
        assert ds_entry["name"] == "Dummy dataset"
        assert ds_entry["version"] == "2025.10"
        assert ds_entry["path"] == tmpdir
        assert ds_entry["independence"] == [
            "Assumed independent test input",
        ]
        hashes = ds_entry["hashes"]
        assert "data.txt" in hashes
        assert hashes["data.txt"] == file_hashes["data.txt"]
        assert len(loaded["git"]["commit"]) == 40
        assert "dirty" in loaded["git"]
        assert "camb" in loaded
        camb_entry = loaded["camb"]
        assert "version" in camb_entry
        assert camb_entry["models"][0]["model"] == "DummyModel"
