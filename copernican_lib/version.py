"""Version helpers for the Copernican Suite.

This module centralises retrieval of the project's version string so that
all components report a consistent value. It first queries
``importlib.metadata`` for the installed package version and, when that
fails, asks :func:`setuptools_scm.get_version` for a Git-derived version. A
fallback of ``"0+unknown"`` ensures that logging and plot footers still show a
version-like identifier even in degenerate cases. Centralising this lookup
avoids scattering hard-coded versions across the codebase.
"""

from importlib.metadata import PackageNotFoundError, version

from setuptools_scm import get_version as scm_get_version

PACKAGE_NAME = "copernican-suite"


def get_version() -> str:
    """Return the Copernican Suite version string.

    The function first tries :mod:`importlib.metadata` to retrieve the
    installed distribution version. When the package is not installed, the
    version is derived from the Git worktree using
    :func:`setuptools_scm.get_version`. If both lookups fail, the placeholder
    ``"0+unknown"`` is returned.
    """

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        try:
            return scm_get_version(root="..", relative_to=__file__)
        except Exception:
            return "0+unknown"


__all__ = ["get_version"]
