"""Tests for the run manifest helper."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from copernican.lib import run_manifest, utils
from copernican.lib.perturbation_contract import PerturbationContractData
from copernican.version import get_version


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
            "model_parameters": {
                "Tcmb_K": 2.7255,
            },
            "background": {
                "derived": {
                    "h": "H0 / 100.0",
                    "baryon_density_fraction": "ombh2 / (h * h)",
                    "Omega_b0": "baryon_density_fraction",
                    "photon_fraction_today": (
                        "2.469e-5 * ((Tcmb_K / 2.7255) ** 4) / (h * h)"
                    ),
                    "Omega_gamma0": "photon_fraction_today",
                    "H": "H0",
                },
                "reionization": {
                    "calibration": {
                        "symbol": "reionization_log10_amplitude",
                        "target_optical_depth": "tau",
                        "lower": -24.0,
                        "upper": 32.0,
                    },
                    "quantities": {
                        "hydrogen_ionization_rate": "1.0e-20",
                    },
                },
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
            "contract_version": 2,
            "standard": True,
            "gauge": "unspecified",
            "variables": {},
            "derived": {},
            "equations": {},
            "constraints": {},
            "closures": {},
            "sources": {},
            "observables": {},
            "initial_conditions": {},
            "boundary_conditions": {},
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
        CMB_PERTURBATION_DATA=PerturbationContractData(
            model_name="DummyModel",
            backend="camb",
            contract_version=2,
            standard=True,
            gauge="unspecified",
            variables={
                "delta_x": object(),
            },
            derived={
                "density_drive": object(),
            },
            equations={
                "continuity_x": object(),
            },
            constraints={
                "poisson_phi": object(),
            },
            closures={
                "psi_equals_phi": object(),
            },
            sources={
                "monopole_source": object(),
            },
            observables={
                "temperature": object(),
                "TT": object(),
            },
            initial_conditions={
                "delta_seed": object(),
            },
            boundary_conditions={},
            numerics={},
            validity=SimpleNamespace(
                regimes=("standard_camb",),
                notes="Uses standard backend.",
            ),
            backend_mapping={
                "camb": SimpleNamespace(
                    uses_standard_perturbations=True,
                    native_solver_required=None,
                    implemented=None,
                )
            },
            dependency_graph_summary=SimpleNamespace(
                independent_variables_used=("tau",),
                model_parameters_used=("p1",),
                background_references_used=("H0",),
                evaluation_order=("equation:continuity_x",),
            ),
            manifest_summary={
                "observable_names": ("temperature", "TT"),
                "execution_route": {
                    "route_id": "backend_standard_perturbations",
                    "prediction_engine": "camb",
                    "transfer_function_path": "camb.standard",
                    "solver": "camb_standard",
                    "route_ready_for_execution": True,
                    "uses_backend_standard_perturbations": True,
                    "uses_native_declared_graph": False,
                    "uses_camb_prediction": True,
                    "uses_camb_standard_perturbations": True,
                    "backend_mapping_implemented": None,
                    "backend_mapping_native_solver_required": None,
                    "backend_mapping_uses_standard_perturbations": True,
                },
                "transfer_component_contracts": {
                    "temperature": {
                        "projection": "line_of_sight_temperature",
                        "kernel": "temperature_mixed_window",
                        "source_term_roles": ("monopole",),
                        "source_term_names": {
                            "monopole": "monopole_source",
                        },
                        "required_projection_roles": (),
                    },
                },
                "angular_power_spectrum_targets": {
                    "TT": {
                        "primary": "temperature",
                        "secondary": "temperature",
                    },
                },
            },
        ),
    )


def _dummy_nonstandard_plugin():
    """Return a native declared-graph plugin fixture."""

    plugin = _dummy_plugin()
    plugin.CMB_PERTURBATION_STANDARD = False
    plugin.CMB_PERTURBATION_CONTRACT["standard"] = False
    plugin.CMB_PERTURBATION_CONTRACT["backend_mapping"]["camb"] = {
        "native_solver_required": True,
        "implemented": True,
    }
    plugin.CMB_PERTURBATION_DATA = PerturbationContractData(
        model_name="DummyModel",
        backend="camb",
        contract_version=2,
        standard=False,
        gauge="conformal_newtonian",
        variables={"delta_x": object()},
        derived={"density_drive": object()},
        equations={"continuity_x": object()},
        constraints={"poisson_phi": object()},
        closures={"psi_equals_phi": object()},
        sources={"monopole_source": object()},
        observables={"temperature": object(), "TT": object()},
        initial_conditions={"delta_seed": object()},
        boundary_conditions={},
        numerics={"ode_rtol": 1.0e-5},
        validity=SimpleNamespace(
            regimes=("linear",),
            notes="Uses native declared graph.",
        ),
        backend_mapping={
            "camb": SimpleNamespace(
                uses_standard_perturbations=None,
                native_solver_required=True,
                implemented=True,
            )
        },
        dependency_graph_summary=SimpleNamespace(
            independent_variables_used=("tau",),
            model_parameters_used=("p1",),
            background_references_used=("H0",),
            evaluation_order=("equation:continuity_x",),
        ),
        manifest_summary={
            "observable_names": ("temperature", "TT"),
            "execution_route": {
                "route_id": "native_declared_graph",
                "prediction_engine": "copernican_native_declared_graph",
                "transfer_function_path": (
                    "copernican.lib.likelihoods.cmb.custom"
                ),
                "solver": "declared_math_graph",
                "route_ready_for_execution": True,
                "uses_backend_standard_perturbations": False,
                "uses_native_declared_graph": True,
                "uses_camb_prediction": False,
                "uses_camb_standard_perturbations": False,
                "backend_mapping_implemented": True,
                "backend_mapping_native_solver_required": True,
                "backend_mapping_uses_standard_perturbations": None,
            },
            "transfer_component_contracts": {
                "temperature": {
                    "projection": "line_of_sight_temperature",
                    "kernel": "temperature_mixed_window",
                    "source_term_roles": ("monopole",),
                    "source_term_names": {
                        "monopole": "monopole_source",
                    },
                    "required_projection_roles": (),
                },
            },
            "angular_power_spectrum_targets": {
                "TT": {
                    "primary": "temperature",
                    "secondary": "temperature",
                },
            },
        },
    )
    return plugin


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
            self.assertEqual(model_entry["perturbation_contract_version"], 2)
            self.assertTrue(model_entry["perturbation_standard"])
            self.assertEqual(model_entry["perturbation_gauge"], "unspecified")
            self.assertEqual(
                model_entry["perturbation_variable_names"],
                ["delta_x"],
            )
            self.assertEqual(
                model_entry["perturbation_derived_names"],
                ["density_drive"],
            )
            self.assertEqual(
                model_entry["perturbation_equation_names"],
                ["continuity_x"],
            )
            self.assertEqual(
                model_entry["perturbation_constraint_names"],
                ["poisson_phi"],
            )
            self.assertEqual(
                model_entry["perturbation_closure_names"],
                ["psi_equals_phi"],
            )
            self.assertEqual(
                model_entry["perturbation_source_names"],
                ["monopole_source"],
            )
            self.assertEqual(
                model_entry["perturbation_observable_names"],
                ["TT", "temperature"],
            )
            self.assertEqual(
                model_entry["perturbation_initial_condition_names"],
                ["delta_seed"],
            )
            self.assertEqual(model_entry["perturbation_equation_count"], 1)
            self.assertEqual(model_entry["perturbation_constraint_count"], 1)
            self.assertEqual(model_entry["perturbation_closure_count"], 1)
            self.assertEqual(model_entry["perturbation_source_count"], 1)
            self.assertEqual(model_entry["perturbation_observable_count"], 2)
            self.assertEqual(
                model_entry["perturbation_initial_condition_count"],
                1,
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
            self.assertEqual(
                model_entry["perturbation_evaluation_order"],
                ["equation:continuity_x"],
            )
            self.assertEqual(model_entry["perturbation_backend"], "camb")
            self.assertIsNone(model_entry["perturbation_backend_implemented"])
            self.assertTrue(
                model_entry["perturbation_backend_uses_standard_perturbations"]
            )
            self.assertIsNone(
                model_entry["perturbation_backend_native_solver_required"]
            )
            self.assertEqual(
                model_entry["custom_cmb_execution_route"]["route_id"],
                "backend_standard_perturbations",
            )
            self.assertTrue(
                model_entry["custom_cmb_execution_route"][
                    "uses_camb_prediction"
                ]
            )
            self.assertTrue(
                model_entry["custom_cmb_execution_route"][
                    "uses_camb_standard_perturbations"
                ]
            )
            self.assertEqual(model_entry["custom_cmb_constraint_count"], 1)
            self.assertEqual(model_entry["custom_cmb_observable_count"], 2)
            self.assertEqual(
                model_entry["custom_cmb_observable_names"],
                ["temperature", "TT"],
            )
            self.assertIn(
                "background_derived_names",
                model_entry["custom_cmb_background_manifest_summary"],
            )
            self.assertIn(
                "Omega_b0",
                model_entry["custom_cmb_background_manifest_summary"][
                    "background_derived_names"
                ],
            )
            self.assertIn(
                "photon_fraction_today",
                model_entry["custom_cmb_background_manifest_summary"][
                    "background_quantity_role_names"
                ]["density"],
            )
            self.assertEqual(
                model_entry["custom_cmb_background_manifest_summary"][
                    "reionization_calibration"
                ]["symbol"],
                "reionization_log10_amplitude",
            )
            self.assertEqual(
                model_entry["custom_cmb_background_manifest_summary"][
                    "recombination_runtime"
                ]["hydrogen_model"],
                "peebles_case_b_ode",
            )
            self.assertIn(
                "camb", model_entry["perturbation_backend_mapping_summary"]
            )
            self.assertEqual(
                model_entry["custom_cmb_graph_manifest_summary"][
                    "transfer_component_contracts"
                ]["temperature"]["kernel"],
                "temperature_mixed_window",
            )
            self.assertEqual(
                model_entry["custom_cmb_graph_manifest_summary"][
                    "angular_power_spectrum_targets"
                ]["TT"]["primary"],
                "temperature",
            )
            self.assertEqual(
                model_entry["custom_cmb_runtime_manifest_summary"][
                    "execution_route"
                ]["solver"],
                "camb_standard",
            )
            self.assertIn(
                "reionization_calibration",
                model_entry["custom_cmb_runtime_manifest_summary"],
            )

    def test_manifest_records_native_execution_route(self) -> None:
        """Native declared runs should record a non-CAMB prediction route."""

        manifest = run_manifest.build_manifest(
            models=[(_dummy_nonstandard_plugin(), "1.0")],
            engine_module=SimpleNamespace(
                __name__="engine", ENGINE_VERSION="0.0"
            ),
            datasets=[],
        )

        model_entry = manifest["camb"]["models"][0]
        self.assertEqual(
            model_entry["custom_cmb_execution_route"]["route_id"],
            "native_declared_graph",
        )
        self.assertTrue(
            model_entry["custom_cmb_execution_route"][
                "uses_native_declared_graph"
            ]
        )
        self.assertFalse(
            model_entry["custom_cmb_execution_route"]["uses_camb_prediction"]
        )
        self.assertEqual(
            model_entry["custom_cmb_runtime_manifest_summary"][
                "execution_route"
            ]["transfer_function_path"],
            "copernican.lib.likelihoods.cmb.custom",
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
