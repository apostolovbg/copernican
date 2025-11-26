# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Rationale: Version retrieval is centralised here because every surface must
# read a single authoritative value to keep manifests, logs and footers
# aligned.
"""Version helpers for the Copernican Suite.

This module centralises retrieval of the project's version string so that
all components report a consistent value. The lookup order honours the
``COPERNICAN_VERSION`` environment variable first so build pipelines can
inject prerelease identifiers. When the variable is unset the helper reads
the ``copernican_lib/VERSION`` file that ships with the source and wheel
distributions. Falling back to a tracked file keeps the runtime version in
sync with ``README.md`` even when a Git tag for the next release has not yet
been created. If the file is missing the function queries
:mod:`importlib.metadata` for the installed package version and, when that
fails, asks :func:`setuptools_scm.get_version` for a Git-derived identifier.
The redundant fallbacks exist because end users lean on version strings for
reproducibility and support even in partially installed environments.
"""

import os
from importlib import resources
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional

try:
    from setuptools_scm import get_version as scm_get_version
except Exception:  # pragma: no cover - dependency missing
    # ``setuptools_scm`` is listed as a runtime dependency, but guard the
    # import so source-only checkouts can still run the suite after installing
    # requirements manually.
    scm_get_version = None  # type: ignore[assignment]

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
        lines = [line.strip() for line in data.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("#"):
                continue
            return line
    return None


def get_version() -> str:
    """Return the Copernican Suite version string.

    The function honours the ``COPERNICAN_VERSION`` environment variable so
    CI or development builds can supply custom prerelease identifiers. If
    the variable is unset the helper attempts to read the tracked
    ``copernican_lib/VERSION`` file. When package metadata is unavailable
    the lookup falls back to :mod:`importlib.metadata` and then to
    :func:`setuptools_scm.get_version`. A final placeholder of
    ``"0+unknown"`` keeps logs and manifests readable even when everything
    else fails, because support requests often rely on the recorded version.
    """

    env_version = os.environ.get("COPERNICAN_VERSION")
    if env_version:
        return env_version

    file_version = _read_version_file()
    if file_version:
        return file_version
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        if scm_get_version is None:
            return "0+unknown"
        try:
            return scm_get_version(root="..", relative_to=__file__)
        except Exception:
            return "0+unknown"


__all__ = ["get_version"]
