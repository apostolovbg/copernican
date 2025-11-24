# Last Updated: 2025-11-24
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.
"""Tests for console I/O helpers in ``copernican_lib.console_output``."""

import unittest
from unittest import mock

from copernican_lib import console_output


class ReadKeypressFallbackTestCase(unittest.TestCase):
    """Verify ``read_keypress`` behaviour when stdin is not interactive."""

    def test_read_keypress_uses_input_when_not_tty(self) -> None:
        """Non-interactive stdin should fall back to the prompt helper."""

        captured: list[tuple[str, str, bool]] = []
        fake_stdin = mock.Mock()
        fake_stdin.isatty.return_value = False

        def _record(
            msg: str = "",
            *,
            end: str = "\n",
            error: bool = False,
        ) -> None:
            captured.append((msg, end, error))

        with (
            mock.patch(
                "copernican_lib.console_output.sys.stdin",
                fake_stdin,
            ),
            mock.patch(
                "copernican_lib.console_output.ask", return_value="Q"
            ) as ask_mock,
            mock.patch(
                "copernican_lib.console_output.write",
                side_effect=_record,
            ),
        ):
            result = console_output.read_keypress(
                {"q", "quit"}, prompt="Select: "
            )

        self.assertEqual(result, "q")
        ask_mock.assert_called_once_with("Select: ")
        self.assertIn(("Select: ", "", False), captured)


if __name__ == "__main__":
    unittest.main()
