# Last Updated: 2025-10-31

"""Tests for launcher scripts.

These checks guard against regressions in the Windows bootstrapper
where the download URL previously collapsed to an empty string on
PowerShell. Ensuring the script defines the URL segments explicitly
and creates the `.python` directory ahead of extraction keeps the
managed interpreter installation robust on new machines.
"""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
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
        self.assertIn(
            'set "DOWNLOAD_URL=%URL_BASE%%URL_FILE%"', self.start_bat
        )
        self.assertIn(
            'set "COPERNICAN_PYTHON_URL=%DOWNLOAD_URL%"', self.start_bat
        )
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
            'set "COPERNICAN_VERSION_PROBE=import sys; print(1 if (3, 11) ',
            self.start_bat,
        )


class LauncherMenuTestCase(unittest.TestCase):
    """Ensure the Unix launchers render the new management menu."""

    def _copy_launcher(self, name: str, target: Path) -> Path:
        """Copy and mark a launcher executable for isolated tests."""

        src = REPO_ROOT / name
        dest = target / name
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        mode = dest.stat().st_mode
        dest.chmod(mode | stat.S_IXUSR)
        return dest

    def _run_menu_capture(self, name: str, setup=None) -> str:
        """Run a launcher in menu-only mode and capture stdout."""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            script = self._copy_launcher(name, tmp_path)
            if setup is not None:
                setup(tmp_path)
            env = os.environ.copy()
            env.pop("VIRTUAL_ENV", None)
            env["COPERNICAN_LAUNCHER_TEST"] = "print-menu"
            result = subprocess.run(
                ["/bin/bash", str(script)],
                check=True,
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
            )
            return result.stdout

    def _write_wrapper(self, target: Path, binary: Path) -> None:
        """Create a small shell wrapper that proxies to the system Python."""

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            f'#!/bin/sh\nexec {binary} "$@"\n',
            encoding="utf-8",
        )
        target.chmod(target.stat().st_mode | stat.S_IXUSR)

    def _create_stub_environment(self, base: Path) -> None:
        """Populate `.python` and `.venv` with wrappers for menu tests."""

        python_exec = (
            subprocess.check_output(["which", "python3"])
            .strip()
            .decode("utf-8")
        )
        python_path = Path(python_exec)
        self._write_wrapper(base / ".python/bin/python3", python_path)
        venv_bin = base / ".venv/bin"
        self._write_wrapper(venv_bin / "python", python_path)
        activate = venv_bin / "activate"
        activate.parent.mkdir(parents=True, exist_ok=True)
        activate.write_text("#!/bin/sh\n", encoding="utf-8")
        activate.chmod(activate.stat().st_mode | stat.S_IXUSR)

    def test_start_sh_reports_install_prompt_when_empty(self) -> None:
        """The Linux launcher offers an install option when clean."""

        output = self._run_menu_capture("start.sh")
        self.assertIn("1) Install dependencies", output)
        self.assertNotIn("Use existing environment", output)

    def test_start_sh_reports_runtime_options_when_ready(self) -> None:
        """The Linux launcher recognises managed dependencies."""

        output = self._run_menu_capture(
            "start.sh", setup=self._create_stub_environment
        )
        self.assertIn("1) Use existing environment", output)
        self.assertIn("Reinstall dependencies", output)

    def test_start_command_shares_menu_logic(self) -> None:
        """The macOS launcher prints the same management menu."""

        output = self._run_menu_capture(
            "start.command", setup=self._create_stub_environment
        )
        self.assertIn("Use existing environment", output)
        self.assertIn("Uninstall dependencies", output)


if __name__ == "__main__":  # pragma: no cover - manual helper
    unittest.main()
