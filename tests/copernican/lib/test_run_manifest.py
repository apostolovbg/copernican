"""Tests for the run manifest helper."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import yaml

from copernican.lib import run_manifest, utils
from copernican.lib.cmb_identity import CCMBS_ID
from copernican.version import get_version


def _dummy_native_runtime():
    """Return one native runtime summary fixture for manifest tests."""

    return SimpleNamespace(
        runtime_signature="native-cmb-runtime:dummy",
        compile_diagnostics=SimpleNamespace(
            runtime_signature="native-cmb-runtime:dummy",
            compiler="copernican.lib.model_coder.compile_native_cmb_runtime",
            compiled_upstream=True,
            hot_path_recompilation_allowed=False,
            parameter_names=("p1",),
            background_reference_names=("H0",),
        ),
    )


def _dummy_plugin():
    return SimpleNamespace(
        MODEL_NAME="DummyModel",
        MODEL_FILENAME="dummy.yml",
        PARAMETER_NAMES=["p1"],
        PARAMETER_PRIORS=[{"type": "uniform", "lower": 0, "upper": 1}],
        valid_for_cmb=True,
        CMB_CONTRACT={
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
                "recombination": {
                    "quantities": {
                        "hydrogen_temperature_K": "3000.0",
                        "hydrogen_alpha_B": "1.0e-19",
                        "beta_continuum": "1.0e-18",
                        "peebles_c": "0.8",
                    },
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
            "gauge": "conformal_newtonian",
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
                "regimes": ["linear"],
                "notes": "Uses the native declared graph.",
            },
            "notes": "Native declared graph manifest fixture.",
        },
        CMB_NATIVE_RUNTIME=_dummy_native_runtime(),
        CMB_PERTURBATION_DATA=SimpleNamespace(
            model_name="DummyModel",
            contract_version=2,
            gauge="conformal_newtonian",
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
                regimes=("linear",),
                notes="Uses native declared graph.",
            ),
            interactions={"photon_baryon_drag": object()},
            conservation_rules={"density_balance": object()},
            projection_extensions={"signal_derivative_alias": object()},
            dependency_graph_summary=SimpleNamespace(
                independent_variables_used=("tau",),
                model_parameters_used=("p1",),
                background_references_used=("H0",),
                evaluation_order=("equation:continuity_x",),
            ),
            manifest_summary={
                "observable_names": ("temperature", "TT"),
                "execution_route": {
                    "solver_id": CCMBS_ID,
                    "solver_label": (
                        "CCMBS — Copernican Cosmic Microwave Background Solver"
                    ),
                    "runtime_module": (
                        "copernican.lib.likelihoods.cmb."
                        "copernican_cmb_solver"
                    ),
                    "ready": True,
                },
                "transfer_component_contracts": {
                    "temperature": {
                        "declared_projection": "line_of_sight_temperature",
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


def _dummy_model_records():
    """Return the required control/test pair for manifest tests."""

    return [(_dummy_plugin(), "1.0"), (_dummy_plugin(), "1.0")]


class TestRunManifest(unittest.TestCase):
    """Exercise manifest creation, persistence, and lifecycle helpers."""

    def test_manifest_contains_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.txt")
            with open(data_path, "w", encoding="utf-8") as file_handle:
                file_handle.write("hello world\n")
            sampler = SimpleNamespace(
                __name__="sampler", SAMPLER_VERSION="0.0"
            )
            file_hashes = {"data.txt": utils.compute_sha256(data_path)}
            utils.set_random_seed(123)
            manifest = run_manifest.build_manifest(
                models=_dummy_model_records(),
                sampler_module=sampler,
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
            self.assertEqual(loaded["sampler"]["name"], "sampler")
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
            self.assertEqual(
                loaded["selection"]["models"],
                ["DummyModel", "DummyModel"],
            )
            self.assertEqual(loaded["selection"]["sampler"]["name"], "sampler")
            self.assertEqual(loaded["selection"]["datasets"], ["ds"])
            self.assertEqual(len(loaded["git"]["commit"]), 40)
            self.assertIn("dirty", loaded["git"])
            self.assertIn("cmb", loaded)
            cmb_entry = loaded["cmb"]
            self.assertEqual(
                cmb_entry["execution_solver"],
                CCMBS_ID,
            )
            model_entry = cmb_entry["models"][0]
            self.assertEqual(model_entry["model"], "DummyModel")
            self.assertEqual(
                model_entry["execution_solver"],
                CCMBS_ID,
            )
            self.assertEqual(
                model_entry["param_map_keys"],
                ["H0", "Neff", "ombh2", "omch2"],
            )
            self.assertEqual(model_entry["call_methods"], [])
            self.assertEqual(model_entry["grids"], {})
            self.assertEqual(model_entry["value_names"], [])
            self.assertEqual(model_entry["perturbation_contract_version"], 2)
            self.assertEqual(
                model_entry["perturbation_gauge"],
                "conformal_newtonian",
            )
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
                model_entry["perturbation_interaction_names"],
                ["photon_baryon_drag"],
            )
            self.assertEqual(
                model_entry["perturbation_conservation_rule_names"],
                ["density_balance"],
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
            self.assertEqual(
                model_entry["perturbation_projection_extension_names"],
                ["signal_derivative_alias"],
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
            self.assertEqual(
                model_entry["native_cmb_execution"]["solver_id"],
                CCMBS_ID,
            )
            self.assertNotIn("backend", model_entry)
            self.assertNotIn("perturbation_standard", model_entry)
            self.assertIn(
                "background_derived_names",
                model_entry["native_cmb_background_manifest_summary"],
            )
            self.assertIn(
                "Omega_b0",
                model_entry["native_cmb_background_manifest_summary"][
                    "background_derived_names"
                ],
            )
            self.assertEqual(
                model_entry["native_cmb_background_manifest_summary"][
                    "background_recombination_quantity_names"
                ],
                [
                    "beta_continuum",
                    "hydrogen_alpha_B",
                    "hydrogen_temperature_K",
                    "peebles_c",
                ],
            )
            self.assertIn(
                "photon_fraction_today",
                model_entry["native_cmb_background_manifest_summary"][
                    "background_quantity_role_names"
                ]["density"],
            )
            self.assertEqual(
                model_entry["native_cmb_background_manifest_summary"][
                    "reionization_calibration"
                ]["symbol"],
                "reionization_log10_amplitude",
            )
            self.assertEqual(
                model_entry["native_cmb_background_manifest_summary"][
                    "recombination_runtime"
                ]["declared_quantity_names"],
                [
                    "beta_continuum",
                    "hydrogen_alpha_B",
                    "hydrogen_temperature_K",
                    "peebles_c",
                ],
            )
            self.assertEqual(
                model_entry["native_cmb_graph_manifest_summary"][
                    "transfer_component_contracts"
                ]["temperature"]["declared_projection"],
                "line_of_sight_temperature",
            )
            self.assertEqual(
                model_entry["native_cmb_graph_manifest_summary"][
                    "angular_power_spectrum_targets"
                ]["TT"]["primary"],
                "temperature",
            )
            self.assertEqual(
                model_entry["native_cmb_runtime_manifest_summary"][
                    "execution_route"
                ]["solver_id"],
                CCMBS_ID,
            )
            self.assertEqual(
                model_entry["native_cmb_runtime_manifest_summary"][
                    "runtime_signature"
                ],
                "native-cmb-runtime:dummy",
            )
            self.assertIn("native_cmb_numerical_envelope", model_entry)
            self.assertIn(
                "numerical_envelope",
                model_entry["native_cmb_runtime_manifest_summary"],
            )
            self.assertIsNone(
                model_entry["native_cmb_numerical_envelope"]["accuracy_tier"]
            )
            self.assertIn(
                "reionization_calibration",
                model_entry["native_cmb_runtime_manifest_summary"],
            )

    def test_manifest_records_native_execution_route(self) -> None:
        """Native declared runs should record the sole prediction route."""

        manifest = run_manifest.build_manifest(
            models=_dummy_model_records(),
            sampler_module=SimpleNamespace(
                __name__="sampler", SAMPLER_VERSION="0.0"
            ),
            datasets=[],
        )

        model_entry = manifest["cmb"]["models"][0]
        self.assertEqual(
            model_entry["native_cmb_execution"]["solver_id"],
            CCMBS_ID,
        )
        self.assertEqual(
            model_entry["perturbation_interaction_names"],
            ["photon_baryon_drag"],
        )
        self.assertEqual(
            model_entry["perturbation_conservation_rule_names"],
            ["density_balance"],
        )
        self.assertEqual(
            model_entry["perturbation_projection_extension_names"],
            ["signal_derivative_alias"],
        )
        self.assertNotIn(
            "uses_camb_prediction",
            model_entry["native_cmb_execution"],
        )
        self.assertEqual(
            model_entry["native_cmb_runtime_manifest_summary"][
                "execution_route"
            ]["runtime_module"],
            "copernican.lib.likelihoods.cmb.copernican_cmb_solver",
        )
        self.assertEqual(
            model_entry["native_cmb_runtime_manifest_summary"][
                "compile_diagnostics"
            ]["compiler"],
            "copernican.lib.model_coder.compile_native_cmb_runtime",
        )

    def test_manifest_import_export_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            utils.set_random_seed(1)
            manifest = run_manifest.build_manifest(
                models=_dummy_model_records(),
                sampler_module=SimpleNamespace(
                    __name__="sampler", SAMPLER_VERSION="0.1"
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
            self.assertEqual(loaded_manifest["sampler"]["name"], "sampler")
            aborted = run_manifest.annotate_outcome(
                loaded_manifest,
                state="aborted",
                outputs="archived",
                reason="Test abort",
            )
            self.assertEqual(aborted["status"]["state"], "aborted")
            self.assertEqual(aborted["status"]["outputs"], "archived")
            self.assertEqual(aborted["status"]["reason"], "Test abort")

    def test_manifest_rejects_a_single_model_record(self) -> None:
        with self.assertRaisesRegex(ValueError, "control model"):
            run_manifest.build_manifest(
                models=[(_dummy_plugin(), "1.0")],
                sampler_module=SimpleNamespace(__name__="sampler"),
                datasets=[],
            )

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
