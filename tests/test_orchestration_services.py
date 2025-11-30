"""Tests for the GUI-safe orchestration service descriptors."""

import unittest
from pathlib import Path

from copernican_lib import orchestration

class OrchestrationServiceMapTestCase(unittest.TestCase):
    """Validate service map contents and controller defaults."""

    def test_service_map_lists_expected_modules(self) -> None:
        service_map = orchestration.describe_orchestration_services()
        self.assertEqual(
            service_map.config_validation.module,
            "copernican_lib.model_spec_validator",
        )
        self.assertIn(
            "validate_and_cache_model",
            service_map.config_validation.entrypoints,
        )
        self.assertEqual(
            service_map.manifest_generation.module,
            "copernican_lib.run_manifest",
        )
        self.assertIn(
            "build_manifest", service_map.manifest_generation.entrypoints
        )
        self.assertEqual(
            service_map.run_control.module,
            "copernican_lib.result_writer",
        )
        self.assertIn("save_summary", service_map.run_control.entrypoints)

    def test_in_process_controller_defaults(self) -> None:
        controller = orchestration.InProcessRunController(lambda _: "token")
        handle = controller.request_run(
            orchestration.RunRequest(
                config=Path("dummy.yml"), datasets=[], engine="engine"
            )
        )
        self.assertEqual(handle.token, "token")
        self.assertEqual(handle.mode, orchestration.LaunchMode.GUI)
        self.assertEqual(tuple(controller.stream_status(handle)), ())
        self.assertEqual(tuple(controller.stream_logs(handle)), ())
        with self.assertRaises(RuntimeError):
            controller.cancel(handle)
        with self.assertRaises(RuntimeError):
            controller.pause(handle)
        with self.assertRaises(RuntimeError):
            controller.resume(handle)
