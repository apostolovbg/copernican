"""Keep the Copernican launchers on their managed bootstrap path."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from devcovenant.core.policy_contract import CheckContext, PolicyCheck, Violation

_DEFAULT_LAUNCHERS = ("start.bat", "start.sh", "start.command")
_WINDOWS_REQUIRED_FRAGMENTS = (
    'set "URL_BASE=%BASE%/download/%REL%/"',
    'set "URL_FILE=cpython-%VER%+%REL%-%ARCH%-pc-windows-msvc-"',
    'set "URL_FILE=%URL_FILE%install_only.tar.gz"',
    'set "URL=%URL_BASE%%URL_FILE%"',
    'set "COPERNICAN_PYTHON_URL=%URL%"',
    'set "COPERNICAN_PYTHON_TAR=python.tar.gz"',
    'set "COPERNICAN_PYDIR=%PYDIR%"',
    'set "PY_VERSION_CHECK=import sys;print(1 if (3,11)<=sys.version_info<"',
    'if not exist "%PYDIR%" mkdir "%PYDIR%"',
    'if "%DOWNLOAD_URL%"=="" (',
    'if defined VIRTUAL_ENV',
    'if not "%COPERNICAN_PYOK%"=="1" if exist "%PYDIR%" rmdir /s /q "%PYDIR%"',
    "Package managers may request your password.",
    "Suite never reads or stores it.",
    "Deactivate the active virtual environment before running.",
)
_POSIX_REQUIRED_FRAGMENTS = (
    "set -eu",
    'SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"',
    'EXPECTED_VENV="$(pwd)/.venv"',
    'VENV_PYTHON="$EXPECTED_VENV/bin/python"',
    'PY_DIR="$(pwd)/.python"',
    'TCL_LIBRARY="$(pwd)/.python/lib/tcl8.6"',
    'TK_LIBRARY="$(pwd)/.python/lib/tk8.6"',
    "python_in_311_series()",
    'if [ -n "${VIRTUAL_ENV:-}" ] && [ "$VIRTUAL_ENV" != "$EXPECTED_VENV" ]; then',
    'if [ "${VIRTUAL_ENV:-}" = "$EXPECTED_VENV" ]; then',
    'curl -fL "https://bootstrap.pypa.io/get-pip.py"',
    "A package manager may request your password.",
    "The Copernican Suite never reads or stores it.",
    "Deactivate the active virtual environment before running",
)


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
    """Return one guardrail violation for a missing launcher fragment."""
    return Violation(
        policy_id=policy_id,
        severity="error",
        file_path=path,
        message=(
            "Launcher guardrail missing: "
            f"{path.name} does not contain {detail} `{fragment}`."
        ),
        suggestion=(
            "Keep the launcher on the managed bootstrap path and preserve "
            "the explicit safety checks from the other entrypoints."
        ),
    )


class StartScriptGuardrailsCheck(PolicyCheck):
    """Validate the launchers' security and bootstrap guardrails."""

    policy_id = "start-script-guardrails"
    version = "1.0.0"

    def check(self, context: CheckContext):
        """Check bootstrap, environment, and download guardrails."""
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
                        message=(
                            f"Missing launcher script: {launcher_name}."
                        ),
                        suggestion=(
                            "Restore the launcher and keep its guardrails in "
                            "sync with the other entrypoints."
                        ),
                    )
                )
                continue

            text = path.read_text(encoding="utf-8")
            if path.name == "start.bat":
                required_fragments = _WINDOWS_REQUIRED_FRAGMENTS
            else:
                required_fragments = _POSIX_REQUIRED_FRAGMENTS
            for fragment in required_fragments:
                if fragment not in text:
                    violations.append(
                        _violation(
                            path=path,
                            policy_id=self.policy_id,
                            fragment=fragment,
                            detail="the launcher safety guardrail",
                        )
                    )

        return violations

