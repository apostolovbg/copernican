"""Tests for the run manifest helper."""

import os
import tempfile
import unittest
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


class TestRunManifest(unittest.TestCase):
    """Exercise manifest creation, persistence, and lifecycle helpers."""

    def test_manifest_contains_required_fields(self) -> None:
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
            self.assertEqual(loaded["copernican"]["version"], get_version())
            self.assertEqual(loaded["engine"]["name"], "engine")
            self.assertEqual(loaded["seed"], 123)
            self.assertEqual(loaded["status"]["state"], "pending")
            self.assertEqual(loaded["status"]["outputs"], "unprepared")
            self.assertIn("ds", loaded["datasets"])
            ds_entry = loaded["datasets"]["ds"]
            self.assertEqual(ds_entry["name"], "Dummy dataset")
            self.assertEqual(ds_entry["version"], "2025.10")
            self.assertEqual(ds_entry["path"], tmpdir)
            self.assertEqual(
                ds_entry["independence"],
                ["Assumed independent test input"],
            )
            hashes = ds_entry["hashes"]
            self.assertIn("data.txt", hashes)
            self.assertEqual(hashes["data.txt"], file_hashes["data.txt"])
            self.assertEqual(loaded["selection"]["models"], ["DummyModel"])
            self.assertEqual(loaded["selection"]["engine"]["name"], "engine")
            self.assertEqual(loaded["selection"]["datasets"], ["ds"])
            self.assertEqual(len(loaded["git"]["commit"]), 40)
            self.assertIn("dirty", loaded["git"])
            self.assertIn("camb", loaded)
            camb_entry = loaded["camb"]
            self.assertIn("version", camb_entry)
            self.assertEqual(camb_entry["models"][0]["model"], "DummyModel")

    def test_manifest_import_export_cycle(self) -> None:
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
            self.assertEqual(loaded_manifest["engine"]["name"], "engine")
            aborted = run_manifest.annotate_outcome(
                loaded_manifest,
                state="aborted",
                outputs="archived",
                reason="Test abort",
            )
            self.assertEqual(aborted["status"]["state"], "aborted")
            self.assertEqual(aborted["status"]["outputs"], "archived")
            self.assertEqual(aborted["status"]["reason"], "Test abort")

    def test_manifest_custom_target_path(self) -> None:
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
            self.assertEqual(Path(path), target)
            self.assertTrue(target.is_file())
            loaded = run_manifest.load_manifest(path)
            self.assertEqual(loaded["copernican"]["version"], get_version())


if __name__ == "__main__":
    unittest.main()
