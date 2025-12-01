"""Tests for the run manifest helper.

"""

import os
import tempfile
from pathlib import Path
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
        CMB_PARAM_MAP={
            "H0": "p1",
            "ombh2": 0.022,
            "omch2": 0.12,
            "Neff": 3.044,
        },
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
        assert loaded["status"]["state"] == "pending"
        assert loaded["status"]["outputs"] == "unprepared"
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
        assert loaded["selection"]["models"] == ["DummyModel"]
        assert loaded["selection"]["engine"]["name"] == "engine"
        assert loaded["selection"]["datasets"] == ["ds"]
        assert len(loaded["git"]["commit"]) == 40
        assert "dirty" in loaded["git"]
        assert "camb" in loaded
        camb_entry = loaded["camb"]
        assert "version" in camb_entry
        assert camb_entry["models"][0]["model"] == "DummyModel"


def test_manifest_import_export_cycle() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        utils.set_random_seed(1)
        manifest = run_manifest.build_manifest(
            models=[(_dummy_plugin(), "1.0")],
            engine_module=SimpleNamespace(
                __name__="engine", ENGINE_VERSION="0.1"
            ),
            datasets=[
                {
                    "id": "ds",
                    "name": "Dummy dataset",
                    "version": "2025.10",
                    "path": tmpdir,
                    "hashes": {},
                    "independence": "Independent",
                }
            ],
        )
        saved_path = run_manifest.save_manifest(manifest, tmpdir)
        loaded_manifest = run_manifest.load_manifest(saved_path)
        assert loaded_manifest["engine"]["name"] == "engine"
        aborted = run_manifest.annotate_outcome(
            loaded_manifest,
            state="aborted",
            outputs="archived",
            reason="Test abort",
        )
        assert aborted["status"]["state"] == "aborted"
        assert aborted["status"]["outputs"] == "archived"
        assert aborted["status"]["reason"] == "Test abort"


def test_manifest_custom_target_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        target = Path(tmpdir) / "custom" / "manifest.yml"
        manifest = {
            "copernican": {"version": get_version()},
            "status": {"state": "pending"},
        }
        path = run_manifest.save_manifest(
            manifest,
            tmpdir,
            target_path=target,
        )
        assert Path(path) == target
        assert target.is_file()
        loaded = run_manifest.load_manifest(path)
        assert loaded["copernican"]["version"] == get_version()
