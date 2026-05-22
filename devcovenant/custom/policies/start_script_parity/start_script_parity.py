"""Keep the Copernican launchers in visible parity across platforms."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)

_DEFAULT_LAUNCHERS = ("start.bat", "start.sh", "start.command")
_SHARED_VISIBLE_FRAGMENTS = (
    "Start Copernican Suite (GUI)",
    "Start Copernican Suite (CLI)",
    "Run the unit test suite",
    "Environment and dependency management",
    "Update dependencies in the managed virtual environment",
    "Remove the managed virtual environment",
    "Rebuild the managed virtual environment",
    "Install Copernican Suite",
    "Uninstall Copernican Suite",
    "Enable strict warning mode",
    "Disable strict warning mode",
    "Write the number of choice:",
)
_PER_FILE_FRAGMENTS = {
    "start.bat": (
        "set /p CHOICE=Write the number of choice:",
        "echo 1^) Start Copernican Suite (GUI)",
        "echo 2^) Start Copernican Suite (CLI)",
        "echo 5^) Environment and dependency management",
        "echo 7^) Exit",
    ),
    "start.sh": (
        'read -r -p "Write the number of choice: " choice',
        'echo "1) Start Copernican Suite (GUI)"',
        'echo "2) Start Copernican Suite (CLI)"',
        'echo "5) Environment and dependency management"',
        'echo "7) Exit"',
    ),
    "start.command": (
        'read -r -p "Write the number of choice: " choice',
        'echo "1) Start Copernican Suite (GUI)"',
        'echo "2) Start Copernican Suite (CLI)"',
        'echo "5) Environment and dependency management"',
        'echo "7) Exit"',
    ),
}


def _string_options(
    raw_value: object,
    *,
    default: Sequence[str],
) -> tuple[str, ...]:
    """Return a normalized tuple of launcher names."""
    if raw_value is None:
        values: Sequence[object] = default
    elif isinstance(raw_value, str):
        values = [raw_value]
    elif isinstance(raw_value, (list, tuple, set)):
        values = list(raw_value)
    else:
        values = [raw_value]
    cleaned: list[str] = []
    for entry in values:
        token = str(entry or "").strip()
        if token:
            cleaned.append(token)
    return tuple(cleaned or default)


def _violation(
    *,
    path: Path,
    policy_id: str,
    fragment: str,
    detail: str,
) -> Violation:
    """Return one parity violation for a missing launcher fragment."""
    return Violation(
        policy_id=policy_id,
        severity="error",
        file_path=path,
        message=(
            "Launcher parity drift: "
            f"{path.name} is missing {detail} `{fragment}`."
        ),
        suggestion=(
            "Mirror the visible launcher menu copy and prompt text across "
            "start.bat, start.sh, and start.command."
        ),
    )


class StartScriptParityCheck(PolicyCheck):
    """Validate that the three launcher scripts stay in sync."""

    policy_id = "start-script-parity"
    version = "1.0.0"

    def check(self, context: CheckContext):
        """Check visible menu and prompt parity across the launchers."""
        repo_root = context.repo_root
        launcher_names = _string_options(
            self.get_option("launcher_files", _DEFAULT_LAUNCHERS),
            default=_DEFAULT_LAUNCHERS,
        )
        violations: list[Violation] = []

        for launcher_name in launcher_names:
            path = repo_root / launcher_name
            if not path.is_file():
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(f"Missing launcher script: {launcher_name}."),
                        suggestion=(
                            "Restore the launcher and keep its menu copy in "
                            "lock-step with the other entrypoints."
                        ),
                    )
                )
                continue

            text = path.read_text(encoding="utf-8")
            for fragment in _SHARED_VISIBLE_FRAGMENTS:
                if fragment not in text:
                    violations.append(
                        _violation(
                            path=path,
                            policy_id=self.policy_id,
                            fragment=fragment,
                            detail="the shared launcher text",
                        )
                    )
            for fragment in _PER_FILE_FRAGMENTS.get(path.name, ()):
                if fragment not in text:
                    violations.append(
                        _violation(
                            path=path,
                            policy_id=self.policy_id,
                            fragment=fragment,
                            detail="the platform-specific launcher text",
                        )
                    )

        return violations
