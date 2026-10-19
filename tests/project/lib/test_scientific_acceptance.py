"""Tests for the explicit scientific acceptance handoff runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests.project.lib import scientific_acceptance


class ScientificAcceptanceHandoffTestCase(unittest.TestCase):
    """Keep the handoff wiring durable without running the scientific solve."""

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
            with mock.patch.object(
                unittest.defaultTestLoader,
                "loadTestsFromNames",
                return_value=unittest.TestSuite((FaultyProbe(),)),
            ):
                report = scientific_acceptance.run_development_checks(
                    "focused", Path(directory) / "negative.json"
                )
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


if __name__ == "__main__":
    unittest.main()
