"""Tests for the custom start-script parity policy."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from devcovenant.core.policy_contract import CheckContext
from devcovenant.custom.policies.start_script_parity import (
    start_script_parity as parity_module,
)

StartScriptParityCheck = parity_module.StartScriptParityCheck


def _find_repo_root() -> Path:
    """Return the Copernican repository root."""
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "start.bat").is_file() and (
            candidate / "devcovenant"
        ).is_dir():
            return candidate
    raise RuntimeError("Unable to locate the repository root.")


def _copy_launchers(source_root: Path, target_root: Path) -> None:
    """Copy the launcher trio into one temporary repository root."""
    for launcher_name in ("start.bat", "start.sh", "start.command"):
        shutil.copy2(source_root / launcher_name, target_root / launcher_name)


class StartScriptParityPolicyTest(unittest.TestCase):
    """Exercise the start-script parity checker."""

    @classmethod
    def setUpClass(cls) -> None:
        """Cache the repository root and policy instance once."""
        super().setUpClass()
        cls.repo_root = _find_repo_root()
        cls.check = StartScriptParityCheck()

    def _run(self, root: Path):
        """Run the policy against one repository root."""
        context = CheckContext(repo_root=root)
        return self.check.check(context)

    def test_current_launchers_are_in_sync(self) -> None:
        """The tracked launchers should already satisfy the policy."""
        violations = self._run(self.repo_root)
        self.assertEqual(violations, [])

    def test_missing_shared_menu_text_is_reported(self) -> None:
        """Changing the shared menu text should trigger a violation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _copy_launchers(self.repo_root, root)
            launcher = root / "start.command"
            text = launcher.read_text(encoding="utf-8")
            launcher.write_text(
                text.replace(
                    "Environment and dependency management",
                    "Environment management",
                ),
                encoding="utf-8",
            )
            violations = self._run(root)
            self.assertTrue(violations)
            self.assertIn(
                "Environment and dependency management",
                violations[0].message,
            )
