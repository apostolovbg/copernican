"""Tests for launcher scripts.

**Last Updated:** 2025-11-29

These checks guard against regressions in the Windows bootstrapper
where the download URL previously collapsed to an empty string on
PowerShell. Ensuring the script defines the URL segments explicitly
and creates the `.python` directory ahead of extraction keeps the
managed interpreter installation robust on new machines.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class StartScriptTestCase(unittest.TestCase):
    """Validate critical safeguards baked into the start scripts."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the launcher contents once for all tests."""

        super().setUpClass()
        cls.start_bat = (REPO_ROOT / "start.bat").read_text(encoding="utf-8")
        cls.start_sh = (REPO_ROOT / "start.sh").read_text(encoding="utf-8")
        cls.start_command = (REPO_ROOT / "start.command").read_text(
            encoding="utf-8"
        )

    def test_windows_launcher_defines_download_url(self) -> None:
        """Ensure the Windows launcher builds the Python URL explicitly."""

        self.assertIn(
            'set "URL_BASE=%BASE%/download/%REL%/"',
            self.start_bat,
        )
        self.assertIn(
            ('set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-' 'pc-windows-msvc-"'),
            self.start_bat,
        )
        self.assertIn(
            'set "URL_FILE=%URL_FILE%install_only.tar.gz"',
            self.start_bat,
        )
        self.assertIn('set "URL=%URL_BASE%%URL_FILE%"', self.start_bat)
        self.assertIn('set "COPERNICAN_PYTHON_URL=%URL%"', self.start_bat)
        self.assertIn(
            'set "COPERNICAN_PYTHON_TAR=python.tar.gz"', self.start_bat
        )
        self.assertIn('set "COPERNICAN_PYDIR=%PYDIR%"', self.start_bat)
        self.assertIn("$url = $env:COPERNICAN_PYTHON_URL;", self.start_bat)
        self.assertIn(
            "if ([string]::IsNullOrWhiteSpace($url))",
            self.start_bat,
        )

    def test_windows_launcher_creates_python_directory(self) -> None:
        """Ensure the script creates `.python` before tar extraction."""

        self.assertIn(
            'if not exist "%PYDIR%" mkdir "%PYDIR%"',
            self.start_bat,
        )

    def test_windows_launcher_limits_python_series(self) -> None:
        """Confirm the launcher purges interpreters outside Python 3.11."""

        self.assertIn(
            "PY_VERSION_CHECK=import sys;print(1 if (3,11)<=sys.version_info<",
            self.start_bat,
        )
        self.assertIn("%PY_VERSION_CHECK%", self.start_bat)

    def test_launchers_expose_environment_menu(self) -> None:
        """Ensure every launcher prints the refreshed menu copy."""

        self.assertIn(
            "echo 5^) Environment and dependency management",
            self.start_bat,
        )
        self.assertIn(
            'echo "5) Environment and dependency management"',
            self.start_sh,
        )
        self.assertIn(
            'echo "5) Environment and dependency management"',
            self.start_command,
        )

    def test_launchers_prompt_with_write_the_number_copy(self) -> None:
        """Verify the new prompt text appears across start scripts."""

        self.assertIn(
            "set /p CHOICE=Write the number of choice:",
            self.start_bat,
        )
        self.assertIn(
            'read -r -p "Write the number of choice: " choice',
            self.start_sh,
        )
        self.assertIn(
            'read -r -p "Write the number of choice: " choice',
            self.start_command,
        )

    def test_launchers_surface_gui_and_cli_options(self) -> None:
        """Ensure launchers expose dedicated GUI and CLI entries."""

        self.assertIn(
            "echo 1^) Start Copernican Suite (GUI)",
            self.start_bat,
        )
        self.assertIn(
            'echo "1) Start Copernican Suite (GUI)"',
            self.start_sh,
        )
        self.assertIn(
            'echo "1) Start Copernican Suite (GUI)"',
            self.start_command,
        )
        self.assertIn(
            "echo 2^) Start Copernican Suite (CLI)",
            self.start_bat,
        )
        self.assertIn(
            'echo "2) Start Copernican Suite (CLI)"',
            self.start_sh,
        )
        self.assertIn(
            'echo "2) Start Copernican Suite (CLI)"',
            self.start_command,
        )


if __name__ == "__main__":  # pragma: no cover - manual helper
    unittest.main()
