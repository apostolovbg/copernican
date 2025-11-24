# Last Updated: 2025-11-24
"""Tests for CLI menu helpers."""

import os
import unittest
from unittest import mock

from copernican_lib.cli import menus


class SplashScreenTestCase(unittest.TestCase):
    """Ensure the splash screen renders and pauses as expected."""

    def test_show_splash_screen_waits_briefly(self) -> None:
        captured: list[str] = []

        def _record(message: str, *, error: bool = False) -> None:
            prefix = "ERROR: " if error else ""
            captured.append(f"{prefix}{message}")

        with (
            mock.patch("copernican_lib.cli.menus.console.write", _record),
            mock.patch("copernican_lib.cli.menus.time.sleep") as sleep_mock,
        ):
            menus.show_splash_screen("10.1.2")

        self.assertTrue(
            any("C O P E R N I C A N" in line for line in captured),
            "Splash banner text was not written to the console.",
        )
        sleep_mock.assert_called_once_with(1)


class SelectSeedTestCase(unittest.TestCase):
    """Exercise the seed selection prompt."""

    def test_environment_seed_short_circuits_prompt(self) -> None:
        with mock.patch.dict(os.environ, {"COPERNICAN_SEED": "7"}, clear=True):
            with mock.patch(
                "copernican_lib.cli.menus.utils.set_random_seed"
            ) as seed_mock:
                seed = menus.select_seed()
        self.assertEqual(seed, 7)
        seed_mock.assert_called_once_with(7)

    def test_default_seed_selection(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            with (
                mock.patch(
                    "copernican_lib.cli.menus.console.ask", return_value=""
                ),
                mock.patch(
                    "copernican_lib.cli.menus.utils.set_random_seed"
                ) as seed_mock,
            ):
                seed = menus.select_seed()
        self.assertEqual(seed, 0)
        seed_mock.assert_called_once_with(0)


class FailureReasonTestCase(unittest.TestCase):
    """Confirm Stage 1 retry prompts summarise errors."""

    def test_prompt_stage1_retry_accepts_restart(self) -> None:
        with mock.patch(
            "copernican_lib.cli.menus.console.ask", return_value="1"
        ) as ask_mock:
            decision = menus.prompt_stage1_retry(["Example reason"])
        self.assertTrue(decision)
        ask_mock.assert_called()

    def test_prompt_stage1_retry_accepts_exit(self) -> None:
        with mock.patch(
            "copernican_lib.cli.menus.console.ask", return_value="c"
        ) as ask_mock:
            decision = menus.prompt_stage1_retry(["Example reason"])
        self.assertFalse(decision)
        ask_mock.assert_called()


if __name__ == "__main__":  # pragma: no cover - manual execution hook
    unittest.main()
