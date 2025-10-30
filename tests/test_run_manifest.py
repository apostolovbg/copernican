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
            datasets=[("ds", tmpdir, file_hashes)],
        )
        path = run_manifest.save_manifest(manifest, tmpdir)
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        assert loaded["copernican"]["version"] == get_version()
        assert loaded["engine"]["name"] == "engine"
        assert loaded["seed"] == 123
        assert "ds" in loaded["datasets"]
        hashes = loaded["datasets"]["ds"]["hashes"]
        assert "data.txt" in hashes
        assert hashes["data.txt"] == file_hashes["data.txt"]
        assert len(loaded["git"]["commit"]) == 40
        assert "dirty" in loaded["git"]
