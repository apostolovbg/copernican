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

try:
    from setuptools_scm import get_version as scm_get_version
except Exception:  # pragma: no cover - dependency missing
    # ``setuptools_scm`` is listed as a runtime dependency, but guard the
    # import so source-only checkouts can still run the suite after installing
    # requirements manually.
    scm_get_version = None  # type: ignore[assignment]

PACKAGE_NAME = "copernican-suite"


def get_version() -> str:
    """Return the Copernican Suite version string.

    The function first tries :mod:`importlib.metadata` to retrieve the
    installed distribution version. When the package is not installed, the
    version is derived from the Git worktree using
    :func:`setuptools_scm.get_version` if the optional dependency is available.
    If both lookups fail or ``setuptools_scm`` is missing, the placeholder
    ``"0+unknown"`` is returned.
    """

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
