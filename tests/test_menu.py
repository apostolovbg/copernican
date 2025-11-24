# Last Updated: 2025-11-24
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
"""Tests for menu interaction helpers in ``copernican.py``."""

import collections.abc as collections_abc
import importlib
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

if not hasattr(collections_abc, "Buffer"):

    class _DummyBufferProtocol:  # pragma: no cover - shim for Python 3.11
        """Lightweight stand-in so NumPy imports succeed during tests."""

        pass

    collections_abc.Buffer = _DummyBufferProtocol

with mock.patch("sys.version_info", (3, 12, 0)):
    with mock.patch.dict(
        os.environ,
        {"VIRTUAL_ENV": str(Path(__file__).resolve().parents[1] / ".venv")},
    ):
        copernican = importlib.import_module("copernican")
import copernican_lib.dataset_registry


class SplashScreenTestCase(unittest.TestCase):
    """Ensure the splash screen renders and pauses as expected."""

    def test_show_splash_screen_waits_briefly(self) -> None:
        """The helper should print the banner once and sleep for one second."""

        captured: list[str] = []

        def _record(message: str, *, error: bool = False) -> None:
            """Collect console output while mirroring the console signature."""

            prefix = "ERROR: " if error else ""
            captured.append(f"{prefix}{message}")

        with (
            mock.patch("copernican.console.write", _record),
            mock.patch("copernican.time.sleep") as sleep_mock,
        ):
            copernican.show_splash_screen()

        self.assertTrue(
            any("C O P E R N I C A N" in line for line in captured),
            "Splash banner text was not written to the console.",
        )
        sleep_mock.assert_called_once_with(1)


class MenuRunTestsTestCase(unittest.TestCase):
    """Verify the menu invokes ``python -m unittest`` discovery."""

    @mock.patch("subprocess.run")
    def test_run_startup_tests_invokes_unittest_discover(self, run_mock):
        """Ensure the helper spawns the expected discovery command."""
        run_mock.return_value.returncode = 0
        result = copernican.run_startup_tests()
        self.assertTrue(result)
        run_mock.assert_called_once()
        cmd = run_mock.call_args[0][0]
        self.assertEqual(cmd[:3], [sys.executable, "-m", "unittest"])
        self.assertEqual(cmd[3], "discover")
        self.assertIn("-v", cmd)


class SelectSourceDisplayTestCase(unittest.TestCase):
    """Ensure selection prompts show names and return identifiers."""

    @mock.patch(
        "copernican_lib.dataset_registry.console.ask", return_value="1"
    )
    def test_select_source_shows_name(self, _ask_mock):
        registry = {
            "dummy_id": {
                "dataset_name": "Dummy Dataset",
                "description": "demo",
                "data_dir": None,
                "function": lambda *_: None,
            }
        }
        captured = []
        with mock.patch(
            "copernican_lib.dataset_registry.console.write",
            lambda msg: captured.append(msg),
        ):
            result = copernican_lib.dataset_registry._select_source(
                registry, "SNe"
            )
        self.assertEqual(result, "dummy_id")
        out = "".join(captured)
        self.assertIn("Dummy Dataset", out)
        self.assertNotIn("dummy_id", out)


class DependencyPromptTestCase(unittest.TestCase):
    """Test dependency installer confirmation and CI override."""

    @mock.patch("copernican.Path")
    def test_installs_after_confirmation(self, path_mock):
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican._gather_required_packages", return_value=["demo"]
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch("copernican.console.ask", return_value="y") as ask_mock,
            mock.patch("subprocess.run") as run_mock,
            mock.patch("importlib.import_module"),
        ):
            copernican.check_dependencies()
            ask_mock.assert_called_once()
            run_mock.assert_called_once_with(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )

    @mock.patch("copernican.Path")
    def test_auto_confirm_skips_prompt(self, path_mock):
        path_mock.return_value.resolve.return_value.name = ".venv"
        with (
            mock.patch(
                "copernican._gather_required_packages", return_value=["demo"]
            ),
            mock.patch("importlib.util.find_spec", return_value=None),
            mock.patch("copernican.console.ask") as ask_mock,
            mock.patch("subprocess.run") as run_mock,
            mock.patch("importlib.import_module"),
        ):
            copernican.check_dependencies(auto_confirm=True)
            ask_mock.assert_not_called()
            run_mock.assert_called_once_with(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "requirements.lock",
                ],
                check=True,
            )


class SamplerConfigurationPromptTestCase(unittest.TestCase):
    """Exercise the sampler configuration questionnaire."""

    def setUp(self) -> None:
        def fit_cosmology_parameters(
            n_steps: int = 400,
            n_walkers: int = 64,
            burn_in_steps: int | None = None,
            **_kwargs,
        ) -> dict[str, bool]:
            return {"success": True}

        self.engine = SimpleNamespace(
            fit_cosmology_parameters=fit_cosmology_parameters,
            fit_sne_parameters=fit_cosmology_parameters,
            _FIXED_BOUNDS_RTOL=1e-9,
            _FIXED_BOUNDS_ATOL=1e-12,
            ENGINE_KIND="mcmc",
        )
        self.lcdm_plugin = SimpleNamespace(
            PARAMETER_BOUNDS=[(0.0, 1.0)] * 3,
            PARAMETER_NAMES=["Ωm", "ΩΛ", "H0"],
            MODEL_NAME="ΛCDM",
        )
        self.alt_plugin = SimpleNamespace(
            PARAMETER_BOUNDS=[(0.0, 1.0)] * 4,
            PARAMETER_NAMES=["w0", "wa", "Ωk", "Neff"],
            MODEL_NAME="AltModel",
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    @mock.patch("copernican.os.cpu_count", return_value=16)
    def test_defaults_selected_via_enter(
        self, _cpu_mock, ask_mock, _write_mock
    ) -> None:
        """Pressing Enter chooses the recommended sampler plan."""

        ask_mock.side_effect = [""]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "mcmc",
                "n_steps": 400,
                "burn_in_steps": 100,
                "n_walkers": 64,
                "pool_size": 16,
            },
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    @mock.patch("copernican.os.cpu_count", return_value=12)
    def test_custom_plan_collects_values(
        self, _cpu_mock, ask_mock, _write_mock
    ) -> None:
        """Users can enter custom sampler values via the questionnaire."""

        ask_mock.side_effect = ["2", "600", "", "80", "4", ""]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "mcmc",
                "n_steps": 600,
                "burn_in_steps": 120,
                "n_walkers": 80,
                "pool_size": 4,
            },
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    @mock.patch("copernican.os.cpu_count", return_value=10)
    def test_back_option_returns_to_summary(
        self, _cpu_mock, ask_mock, _write_mock
    ) -> None:
        """Selecting back restarts the menu before launching."""

        ask_mock.side_effect = ["2", "", "", "", "", "b", ""]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "mcmc",
                "n_steps": 400,
                "burn_in_steps": 100,
                "n_walkers": 64,
                "pool_size": 10,
            },
        )


class NestedSamplerConfigurationPromptTestCase(unittest.TestCase):
    """Exercise the nested-sampling configuration questionnaire."""

    def setUp(self) -> None:
        def fit_cosmology_parameters(
            *,
            n_live_points: int = 128,
            max_iterations: int = 5000,
            evidence_tolerance: float = 1e-3,
            enlargement_fraction: float = 1.5,
            **_kwargs,
        ) -> dict[str, bool]:
            return {"success": True}

        self.engine = SimpleNamespace(
            fit_cosmology_parameters=fit_cosmology_parameters,
            fit_sne_parameters=fit_cosmology_parameters,
            ENGINE_KIND="nested",
        )
        self.lcdm_plugin = SimpleNamespace(
            PARAMETER_BOUNDS=[(0.0, 1.0)] * 3,
            PARAMETER_NAMES=["Ωm", "ΩΛ", "H0"],
            MODEL_NAME="ΛCDM",
        )
        self.alt_plugin = SimpleNamespace(
            PARAMETER_BOUNDS=[(0.0, 1.0)] * 4,
            PARAMETER_NAMES=["w0", "wa", "Ωk", "Neff"],
            MODEL_NAME="AltModel",
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_nested_defaults_selected(self, ask_mock, _write_mock) -> None:
        """Pressing Enter should accept the recommended nested plan."""

        ask_mock.side_effect = [""]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "nested",
                "n_live_points": 128,
                "max_iterations": 5000,
                "evidence_tolerance": 1e-3,
                "enlargement_fraction": 1.5,
            },
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_nested_custom_plan(self, ask_mock, _write_mock) -> None:
        """Operators can customise nested-sampling parameters."""

        ask_mock.side_effect = [
            "2",
            "256",
            "10000",
            "5e-4",
            "2.0",
            "",
        ]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "nested",
                "n_live_points": 256,
                "max_iterations": 10000,
                "evidence_tolerance": 5e-4,
                "enlargement_fraction": 2.0,
            },
        )

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_nested_back_option(self, ask_mock, _write_mock) -> None:
        """Backtracking returns to the defaults summary before exiting."""

        ask_mock.side_effect = ["2", "", "", "", "", "b", ""]
        plan = copernican.prompt_sampling_configuration(
            self.engine,
            self.lcdm_plugin,
            self.alt_plugin,
            None,
            None,
            None,
        )
        self.assertEqual(
            plan,
            {
                "engine_kind": "nested",
                "n_live_points": 128,
                "max_iterations": 5000,
                "evidence_tolerance": 1e-3,
                "enlargement_fraction": 1.5,
            },
        )


class PostRunMenuTestCase(unittest.TestCase):
    """Validate the post-run navigation menu."""

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_default_selection_runs_again(self, ask_mock, _write_mock) -> None:
        """Pressing Enter launches another evaluation."""

        ask_mock.side_effect = [""]
        result = copernican.prompt_post_run_action()
        self.assertTrue(result)

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_cancel_option_exits(self, ask_mock, _write_mock) -> None:
        """Choosing C exits the workflow."""

        ask_mock.side_effect = ["c"]
        result = copernican.prompt_post_run_action()
        self.assertFalse(result)

    @mock.patch("copernican.console.write")
    @mock.patch("copernican.console.ask")
    def test_invalid_then_valid_choice(self, ask_mock, write_mock) -> None:
        """The menu repeats until a valid answer is provided."""

        ask_mock.side_effect = ["maybe", "1"]
        result = copernican.prompt_post_run_action()
        self.assertTrue(result)
        write_calls = [
            args[0] for args, _ in write_mock.call_args_list if args
        ]
        self.assertIn("Please choose 1 or C.", write_calls)


if __name__ == "__main__":
    unittest.main()
