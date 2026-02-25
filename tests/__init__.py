# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Test package for the Copernican Suite."""

import atexit
import logging
import shutil
from pathlib import Path

# Configure a simple root logger so tests surface informative messages.  The
# ``force`` flag ensures that duplicate handlers are not added when the test
# suite is executed multiple times.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,
)


_repo_root = Path(__file__).resolve().parents[1]
_output_root = _repo_root / "output"
_PREEXISTING_RUN_DIRS = (
    {
        entry.name
        for entry in _output_root.iterdir()
        if entry.is_dir() and entry.name.startswith("copernican-run_")
    }
    if _output_root.is_dir()
    else set()
)
_NEW_CONFIG_DIR = _output_root / "copernican_run_NEW_CONFIG"


def _cleanup_new_config_workspace() -> None:
    """Remove any lingering `copernican_run_NEW_CONFIG` workspace."""

    if _NEW_CONFIG_DIR.is_dir():
        shutil.rmtree(_NEW_CONFIG_DIR, ignore_errors=True)


def _cleanup_test_outputs() -> None:
    """Delete only test-generated `copernican-run_*` folders on exit."""

    if not _output_root.is_dir():
        return
    _cleanup_new_config_workspace()
    for entry in _output_root.iterdir():
        if (
            entry.is_dir()
            and entry.name.startswith("copernican-run_")
            and entry.name not in _PREEXISTING_RUN_DIRS
        ):
            shutil.rmtree(entry, ignore_errors=True)


atexit.register(_cleanup_test_outputs)
_cleanup_test_outputs()
_cleanup_new_config_workspace()

# The presence of this file allows unittest discovery to treat ``tests`` as a
# package.
