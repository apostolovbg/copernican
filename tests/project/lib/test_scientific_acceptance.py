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


if __name__ == "__main__":
    unittest.main()
