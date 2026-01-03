"""Ensure DevCovenant runs inside the managed .venv."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class ManagedVenvCheck(PolicyCheck):
    """Verify the active interpreter is the repository's managed virtualenv."""

    policy_id = "managed-venv"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Error when DevCovenant runs outside `<repo>/.venv`."""
        repo_root = context.repo_root.resolve()
        cfg = context.get_policy_config(self.policy_id)
        expected_entries = cfg.get(
            "expected_virtualenvs",
            [".venv"],
        )
        expected_paths = [
            (repo_root / Path(entry)).resolve() for entry in expected_entries
        ]
        if not any(path.exists() for path in expected_paths):
            return []

        if self._in_expected_venv(expected_paths):
            return []

        message = (
            "DevCovenant must run from the managed virtual environment. "
            "Please re-run start.sh/start.command/start.bat so `.venv` is "
            "active before editing code."
        )
        return [
            Violation(
                policy_id=self.policy_id,
                severity="error",
                file_path=expected_paths[0],
                line_number=1,
                message=message,
            )
        ]

    def _in_expected_venv(self, expected_paths: List[Path]) -> bool:
        """Return True when the active interpreter lives inside *expected*."""
        env_path = os.environ.get("VIRTUAL_ENV")
        candidates = []
        if env_path:
            candidates.append(Path(env_path))
        candidates.append(Path(sys.executable).parent)

        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            for directory in expected_paths:
                if directory in resolved.parents or resolved == directory:
                    return True
        return False
