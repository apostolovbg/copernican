# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Version helpers for the Copernican Suite.

This module centralises retrieval of the project's version string so every
component reports a consistent value.  The helper first respects the
``COPERNICAN_VERSION`` environment variable, enabling CI pipelines to pin
prerelease identifiers without editing the source.  When that variable is
unset it reads the tracked ``copernican_lib/VERSION`` file, keeping runtime
metadata aligned with the documented version number.  If every lookup fails a
final fallback of ``"0+unknown"`` ensures the logger and plot footers still
display a version-like string.
"""

import os
from importlib import resources
from pathlib import Path
from typing import Optional

PACKAGE_NAME = "copernican-suite"
VERSION_FILENAME = "VERSION"


def _read_version_file() -> Optional[str]:
    """Return the version stored alongside the package if available.

    The helper searches both the installed package resources and the source
    checkout, trimming whitespace from the stored value.  Any failure to
    locate or parse the file silently falls back to the next lookup stage so
    runtime version retrieval never raises unexpectedly.
    """

    candidates = []
    package = __package__ or "copernican_lib"
    try:
        candidates.append(resources.files(package).joinpath(VERSION_FILENAME))
    except Exception:  # pragma: no cover - importlib.resources fallback
        pass
    candidates.append(Path(__file__).with_name(VERSION_FILENAME))

    for candidate in candidates:
        try:
            if hasattr(candidate, "read_text"):
                data = candidate.read_text(encoding="utf-8")
            else:  # pragma: no cover - safety net for unexpected types
                data = Path(candidate).read_text(encoding="utf-8")
        except FileNotFoundError:
            continue
        except Exception:
            continue
        value = data.strip()
        if value:
            return value
    return None


def get_version() -> str:
    """Return the Copernican Suite version string.

    The function first honours the ``COPERNICAN_VERSION`` environment
    variable so CI or development builds can supply custom prerelease
    identifiers. When the variable is unset the helper attempts to read the
    tracked ``copernican_lib/VERSION`` file.
    A final fallback of ``"0+unknown"`` ensures logging and plot footers still
    display a version-like string in degenerate environments.
    """

    env_version = os.environ.get("COPERNICAN_VERSION")
    if env_version:
        return env_version

    file_version = _read_version_file()
    if file_version:
        return file_version
    return "0+unknown"


__all__ = ["get_version"]
