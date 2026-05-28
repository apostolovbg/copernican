"""Tests for the run manifest helper."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from copernican_lib import run_manifest, utils
from copernican_lib.perturbation_contract import PerturbationContractIR
from copernican_lib.version import get_version


def _dummy_plugin():
    return SimpleNamespace(
        MODEL_NAME="DummyModel",
        MODEL_FILENAME="dummy.yml",
        PARAMETER_NAMES=["p1"],
        PARAMETER_PRIORS=[{"type": "uniform", "lower": 0, "upper": 1}],
        valid_for_cmb=True,
        CMB_CONTRACT={
            "backend": "camb",
            "param_map": {
                "H0": "p1",
                "ombh2": 0.022,
                "omch2": 0.12,
                "Neff": 3.044,
            },
            "grids": {},
            "values": {},
            "calls": [],
        },
        CMB_PARAM_MAP={
            "H0": "p1",
            "ombh2": 0.022,
            "omch2": 0.12,
            "Neff": 3.044,
        },
        CMB_PERTURBATION_CONTRACT={
            "contract_version": 1,
            "standard": True,
            "gauge": "unspecified",
            "variables": {},
            "derived": {},
            "equations": {},
            "closures": {},
            "sources": {},
            "validity": {
                "regimes": ["standard_camb"],
                "notes": "Uses the backend standard perturbation machinery.",
            },
            "backend_mapping": {
                "camb": {
                    "uses_standard_perturbations": True,
                }
            },
            "notes": (
                "This model declares that its CMB perturbations are "
                "represented by the selected backend's standard "
                "perturbation system."
            ),
        },
        CMB_PERTURBATION_STANDARD=True,
        CMB_PERTURBATION_IR=PerturbationContractIR(
            model_name="DummyModel",
            backend="camb",
            contract_version=1,
            standard=True,
            gauge="unspecified",
            variables={
                "delta_x": object(),
            },
            derived={
                "Phi_tau": object(),
            },
            equations={
                "continuity_x": object(),
            },
            closures={
                "no_anisotropic_stress": object(),
            },
            sources={
                "poisson": object(),
            },
            validity=SimpleNamespace(
                regimes=("standard_camb",),
                notes="Uses standard backend.",
            ),
            backend_mapping={
                "camb": SimpleNamespace(
                    uses_standard_perturbations=True,
                    native_solver_required=None,
                    solver=None,
                    implemented=None,
                )
            },
            dependency_graph_summary=SimpleNamespace(
                independent_variables_used=("tau",),
                model_parameters_used=("p1",),
                background_references_used=("H0",),
            ),
            manifest_summary={},
        ),
    )


class TestRunManifest(unittest.TestCase):
    """Exercise manifest creation, persistence, and lifecycle helpers."""

    def test_manifest_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.txt")
            with open(data_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("hello world\n")
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
            with open(path, "r", encoding="utf-8") as file_handle:
                loaded = yaml.safe_load(file_handle)
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
            model_entry = camb_entry["models"][0]
            self.assertEqual(model_entry["model"], "DummyModel")
            self.assertEqual(model_entry["backend"], "camb")
            self.assertEqual(
                model_entry["param_map_keys"],
                ["H0", "Neff", "ombh2", "omch2"],
            )
            self.assertEqual(model_entry["call_methods"], [])
            self.assertEqual(model_entry["grids"], {})
            self.assertEqual(model_entry["value_names"], [])
            self.assertEqual(model_entry["perturbation_contract_version"], 1)
            self.assertTrue(model_entry["perturbation_standard"])
            self.assertEqual(model_entry["perturbation_gauge"], "unspecified")
            self.assertEqual(
                model_entry["perturbation_variable_names"],
                ["delta_x"],
            )
            self.assertEqual(
                model_entry["perturbation_derived_names"],
                ["Phi_tau"],
            )
            self.assertEqual(
                model_entry["perturbation_equation_names"],
                ["continuity_x"],
            )
            self.assertEqual(
                model_entry["perturbation_closure_names"],
                ["no_anisotropic_stress"],
            )
            self.assertEqual(
                model_entry["perturbation_source_names"],
                ["poisson"],
            )
            self.assertEqual(
                model_entry["perturbation_independent_variables_used"],
                ["tau"],
            )
            self.assertEqual(
                model_entry["perturbation_model_parameters_used"],
                ["p1"],
            )
            self.assertEqual(
                model_entry["perturbation_background_references_used"],
                ["H0"],
            )
            self.assertEqual(model_entry["perturbation_backend"], "camb")
            self.assertIsNone(model_entry["perturbation_backend_solver"])
            self.assertIsNone(model_entry["perturbation_backend_implemented"])
            self.assertTrue(
                model_entry["perturbation_backend_uses_standard_perturbations"]
            )
            self.assertIsNone(
                model_entry["perturbation_backend_native_solver_required"]
            )
            self.assertIn(
                "camb", model_entry["perturbation_backend_mapping_summary"]
            )

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


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the manifest helper API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(run_manifest.build_manifest))
        self.assertTrue(callable(run_manifest.save_manifest))
        self.assertTrue(callable(run_manifest.load_manifest))
        self.assertTrue(callable(run_manifest.annotate_outcome))


if __name__ == "__main__":
    unittest.main()
