"""Tests for the explicit scientific acceptance handoff runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.project.lib import scientific_acceptance


class ScientificAcceptanceHandoffTestCase(unittest.TestCase):
    """Keep the handoff wiring durable without running the scientific solve."""

    def test_named_point_failure_preserves_actual_observed_arrays(self):
        """A failed numerical call retains its partial evidence honestly."""

        import numpy

        from copernican.lib.likelihoods.cmb import cmb, evidence_assets
        from copernican.lib.likelihoods.cmb.runtime import evidence

        runtime = {"model_name": "failure_fixture"}
        physical_inputs = scientific_acceptance.completion_physical_inputs(
            runtime
        )
        request = {
            "ells": [2],
            "spectra": ["TT"],
            "numerical_overrides": {},
            "workload": "full_spectrum",
        }
        case_id = "fixture:initial"
        payload = {"state": numpy.arange(4.0)}

        def failing_solve(*args, **kwargs):
            evidence.publish_completed_cmb_products(
                "native_refinement", payload
            )
            raise ValueError("independent numerical rejection")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = evidence_assets.write_json_artifact(
                root, "reference.json", {"not_a_scientific_reference": True}
            )
            contract = {
                "contract_sha256": "fixture-contract",
                "cases": {
                    case_id: {
                        "reference_kind": "camb",
                        "ells": [2],
                        "surfaces": ["TT"],
                        "reference_identity": "fixture",
                        "reference_sha256": reference["sha256"],
                    }
                },
                "bindings": {
                    case_id: {
                        "reference_artifact": reference,
                        "model_filename": "fixture",
                        "point": "initial",
                        "physical_inputs": physical_inputs,
                        "request": request,
                        "declaration_sha256": "fixture-declaration",
                        "numerical_plan": {"policy": "ordinary_automatic"},
                    }
                },
            }
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "completion_point_contract",
                    return_value=runtime,
                ),
                mock.patch.object(
                    cmb,
                    "compute_cmb_spectrum_from_contract",
                    side_effect=failing_solve,
                ),
            ):
                result = scientific_acceptance.run_completion_point(
                    root,
                    case_id,
                    contract=contract,
                    model_records=[
                        SimpleNamespace(
                            model_filename="fixture", plugin=object()
                        )
                    ],
                )
            self.assertFalse(result["accepted"])
            self.assertFalse(result["completion_accepted"])
            self.assertIn(
                "independent numerical rejection", result["failure"]["message"]
            )
            self.assertEqual(len(result["observations"]), 1)
            retained = evidence_assets.read_array_products(
                root, result["observations"][0]["artifact"], materialize=True
            )
            numpy.testing.assert_array_equal(
                retained["state"], payload["state"]
            )
            producer = next(root.glob("cases/*/*/producer.json"))
            self.assertEqual(
                json.loads(producer.read_text())["observations"], []
            )
            self.assertEqual(
                json.loads(producer.read_text())["source_identity"],
                result["source_identity"],
            )

    def test_physical_binding_keeps_math_and_excludes_compiled_handles(
        self,
    ):
        runtime = {
            "model_name": "independent_declaration",
            "background": {"H": "H0/a"},
            "perturbations": {"equations": ["x'=k*x"]},
            "background_runtime": object(),
            "perturbation_data": object(),
            "compile_diagnostics": object(),
            "_engine_numerical_plan": {"nodes": 1024},
        }
        inputs = scientific_acceptance.completion_physical_inputs(runtime)
        self.assertEqual(inputs["background"], runtime["background"])
        self.assertEqual(inputs["perturbations"], runtime["perturbations"])
        self.assertNotIn("background_runtime", inputs)
        self.assertNotIn("_engine_numerical_plan", inputs)
        self.assertEqual(json.loads(json.dumps(inputs)), inputs)

    def test_completion_reference_retains_declared_temperature(self):
        """Reference input resolution follows the actual physical contract."""

        from tests.copernican.lib.likelihoods.cmb import test_cmb
        from tests.project.lib import camb_reference

        raw = test_cmb._declared_scalar_hierarchy_contract()
        raw["model_parameters"]["Tcmb_K"] = 3.0
        contract = test_cmb._prepare_declared_contract(raw)
        reference = scientific_acceptance.completion_reference_contract(
            contract
        )
        self.assertEqual(reference["param_map"]["TCMB"], 3.0)
        self.assertEqual(camb_reference._make_camb_params(reference).TCMB, 3.0)

    def test_canonical_default_temperature_keeps_resolved_physics(self):
        from tests.copernican.lib.likelihoods.cmb import test_cmb
        from tests.project.lib import camb_reference

        runtime = test_cmb._prepare_declared_contract(
            test_cmb._declared_scalar_hierarchy_contract()
        )
        explicit = scientific_acceptance.completion_reference_contract(runtime)
        canonical = scientific_acceptance.completion_reference_contract(
            runtime, canonical_defaults=True
        )
        self.assertEqual(explicit["param_map"]["TCMB"], 2.7255)
        self.assertNotIn("TCMB", canonical["param_map"])
        self.assertEqual(
            str(camb_reference._make_camb_params(explicit)),
            str(camb_reference._make_camb_params(canonical)),
        )
        runtime["model_parameters"]["Tcmb_K"] = 3.0
        canonical = scientific_acceptance.completion_reference_contract(
            runtime, canonical_defaults=True
        )
        self.assertEqual(canonical["param_map"]["TCMB"], 3.0)

    def test_completion_preflight_persists_failures_without_solver_execution(
        self,
    ):
        from copernican.lib.likelihoods.cmb import cmb, evidence_assets

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "materialize_completion_contract",
                    side_effect=ValueError(
                        "Frozen independent reference differs"
                    ),
                ),
                mock.patch.object(
                    cmb,
                    "compute_cmb_spectrum_from_contract",
                    side_effect=AssertionError("preflight must not solve"),
                ) as solve,
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    ["--completion-preflight", "--evidence-root", directory]
                )
            report = json.loads(
                (root / "completion-preflight.json").read_text()
            )
            self.assertEqual(status, 1)
            self.assertFalse(report["preflight_ready"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["scientific_acceptance"])
            self.assertIn("Frozen independent", report["failure"]["message"])
            retained = evidence_assets.read_json_artifact(
                root, report["attempt"]
            )
            self.assertEqual(retained["failure"], report["failure"])
            solve.assert_not_called()

    def test_preflight_readiness_does_not_claim_scientific_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scientific_acceptance._retain(
                root / "completion-contract.json", {}
            )
            with mock.patch.object(
                scientific_acceptance,
                "materialize_completion_contract",
                return_value={
                    "contract_sha256": "contract",
                    "cases": {"case": {}},
                },
            ):
                report = scientific_acceptance.run_completion_preflight(root)
            self.assertTrue(report["preflight_ready"])
            self.assertFalse(report["accepted"])
            self.assertFalse(report["scientific_acceptance"])

    def test_development_cli_reports_cost_without_rejecting_passing_tests(
        self,
    ):
        """A passing test retains cost telemetry and CLI success."""

        selection = (
            "tests.copernican.lib.test_cmb_output.CMBOutputTestCase."
            "test_public_units_follow_temperature_and_potential_dimensions",
        )
        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.dict(
                    scientific_acceptance.DEVELOPMENT_WORKLOADS,
                    {"focused": selection},
                ),
                mock.patch.dict(
                    scientific_acceptance.DEVELOPMENT_COST_TARGETS,
                    {"focused": 0.0},
                ),
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    [
                        "--development-checks",
                        "focused",
                        "--evidence-root",
                        directory,
                    ]
                )
            report = json.loads(
                (Path(directory) / "development-focused.json").read_text()
            )
            self.assertEqual(status, 0)
            self.assertTrue(report["passed"])
            self.assertFalse(report["within_cost_target"])
            self.assertGreater(report["elapsed_seconds"], 0.0)
            self.assertEqual(report["tests_run"], 1)
            self.assertFalse(report["scientific_acceptance"])

    def test_development_runner_measures_tests_without_scientific_solves(
        self,
    ):
        """A measured unit-test workload retains per-test cost and logs."""

        selection = (
            "tests.copernican.lib.test_cmb_output.CMBOutputTestCase."
            "test_public_units_follow_temperature_and_potential_dimensions",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "development.json"
            with mock.patch.dict(
                scientific_acceptance.DEVELOPMENT_WORKLOADS,
                {"focused": selection},
            ):
                report = scientific_acceptance.run_development_checks(
                    "focused", destination
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["tests_run"], 1)
            self.assertEqual(report["tests"][0]["test"], selection[0])
            self.assertGreater(report["elapsed_seconds"], 0)
            self.assertTrue((destination.parent / report["test_log"]).exists())
            self.assertFalse(report["scientific_acceptance"])
            self.assertIn("numpy", report["environment"])

    def test_development_runner_retains_an_actual_failed_negative_control(
        self,
    ):
        """An isolated faulty probe produces a failed development report."""

        class FaultyProbe(unittest.TestCase):
            def runTest(self):
                self.assertEqual(1, 2, "manufactured negative control")

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(
                    unittest.defaultTestLoader,
                    "loadTestsFromNames",
                    return_value=unittest.TestSuite((FaultyProbe(),)),
                ),
                mock.patch("builtins.print"),
            ):
                status = scientific_acceptance.main(
                    [
                        "--development-checks",
                        "focused",
                        "--evidence-root",
                        directory,
                    ]
                )
            report = json.loads(
                (Path(directory) / "development-focused.json").read_text()
            )
            self.assertEqual(status, 1)
            self.assertFalse(report["passed"])
            self.assertEqual(report["failures"], 1)
            self.assertEqual(report["tests"][0]["outcome"], "failed")

    def test_history_regeneration_command_cannot_launch_scientific_acceptance(
        self,
    ):
        with mock.patch.object(
            scientific_acceptance.camb_reference,
            "write_lcdm_history_reference_fixture",
            return_value=Path.cwd() / "reference.json",
        ) as write:
            with mock.patch.object(
                scientific_acceptance,
                "run_completion_baseline",
                side_effect=AssertionError("No baseline execution"),
            ):
                with mock.patch("builtins.print"):
                    self.assertEqual(
                        scientific_acceptance.main(
                            ["--freeze-development-reference"]
                        ),
                        0,
                    )
        write.assert_called_once_with()

    def test_handoff_reloads_the_report_written_by_the_runner(self):
        """The explicit handoff must compare writer and verifier outputs."""

        plugin = SimpleNamespace(
            INITIAL_GUESSES=(75.0,),
            get_cmb_declared_runtime=mock.Mock(
                return_value={"param_map": {"H0": 75.0}}
            ),
        )
        record = SimpleNamespace(
            model_filename="model_lcdm.yml",
            ready=True,
            plugin=plugin,
            failure=None,
            status="ready",
        )
        report = {
            "accepted": True,
            "quantitative_decision": {"status": "accepted"},
            "report_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as output_directory:
            destination = Path(output_directory) / "fixed-lcdm.json"
            with (
                mock.patch.object(
                    scientific_acceptance,
                    "discover_cmb_model_records",
                    return_value=(record,),
                ),
                mock.patch.object(
                    scientific_acceptance.camb_reference,
                    "build_fixed_lcdm_camb_parity_row",
                    return_value={"reference": "camb"},
                ),
                mock.patch.object(
                    scientific_acceptance,
                    "run_fixed_lcdm_cmb_parity",
                    return_value=report,
                ) as run,
                mock.patch.object(
                    scientific_acceptance,
                    "read_cmb_parity_matrix_report",
                    return_value=report,
                ) as read,
            ):
                result = (
                    scientific_acceptance.run_fixed_lcdm_scientific_acceptance(
                        destination
                    )
                )

        self.assertEqual(result, report)
        run.assert_called_once()
        read.assert_called_once_with(destination)
        plugin.get_cmb_declared_runtime.assert_called_once_with((75.0,))

    def test_baseline_command_keeps_rejection_explicit(self):
        with mock.patch.object(
            scientific_acceptance,
            "run_completion_baseline",
            return_value={"accepted": False},
        ) as run:
            with mock.patch("builtins.print"):
                result = scientific_acceptance.main(["--baseline"])
        self.assertEqual(result, 1)
        self.assertIn("ccmbs-acceptance", str(run.call_args.args[0]))

    def test_checkout_provenance_records_current_uncommitted_content(self):
        root = Path(__file__).resolve().parents[3]
        recorded = scientific_acceptance.checkout_provenance(root)
        self.assertEqual(len(recorded["source_revision"]), 40)
        self.assertEqual(
            recorded["sha256"],
            scientific_acceptance.diagnostics._canonical_sha256(
                {
                    name: recorded[name]
                    for name in ("source_revision", "uncommitted")
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
