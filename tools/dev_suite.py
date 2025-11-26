# Last Updated: 2025-11-25
"""Unified developer automation suite for Copernican.

The script mirrors CI expectations by chaining the formatting, linting,
policy enforcement and test commands that contributors should run before
committing changes. It intentionally mirrors the pre-commit hook ordering
so developers can rely on a single entry point while still invoking the
individual tools when troubleshooting specific failures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence, Tuple

Command = Tuple[str, Sequence[str]]


def _run_step(step: Command) -> None:
    """Run a single tool and exit immediately on failure.

    Each step logs the command line prior to execution so developers can
    copy and rerun the underlying tool directly. Exiting on the first
    failure keeps the workflow predictable and prevents noisy cascades
    when an early formatter or fixer has not yet run.
    """

    name, command = step
    command_display = " ".join(command)
    print(f"[dev-suite] Running {name}: {command_display}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            f"[dev-suite] {name} failed with exit code "
            f"{result.returncode}."
        )
        sys.exit(result.returncode)


def _command_sequence(repo_root: Path) -> Iterable[Command]:
    """Yield the command sequence used by the developer suite.

    The order matches the pre-commit pipeline: formatters run first,
    followed by Ruff's fast checks, DriftGuard enforcement for staged
    changes, and finally the pytest suite. Each command uses explicit
    modules to avoid dependency on shell resolution and to make the
    intent clear to readers and automation alike.
    """

    yield ("Black", ["python", "-m", "black", str(repo_root)])
    yield (
        "isort",
        ["python", "-m", "isort", "--profile", "black", str(repo_root)],
    )
    yield (
        "Ruff",
        ["python", "-m", "ruff", "check", "--fix", str(repo_root)],
    )
    yield (
        "DriftGuard fix",
        [
            "python",
            "-m",
            "driftguard.cli",
            "fix",
            "--scope=staged",
            "--mode=fast",
            "--repo-root",
            str(repo_root),
        ],
    )
    yield (
        "DriftGuard check",
        [
            "python",
            "-m",
            "driftguard.cli",
            "check",
            "--scope=staged",
            "--mode=fast",
            "--repo-root",
            str(repo_root),
        ],
    )
    yield ("pytest", ["python", "-m", "pytest", "-q"])


def main() -> None:
    """Execute the full developer suite and exit on the first failure."""

    repo_root = Path(__file__).resolve().parent.parent
    for step in _command_sequence(repo_root):
        _run_step(step)
    print("[dev-suite] All checks passed.")


if __name__ == "__main__":
    main()
