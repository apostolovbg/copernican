"""Version helpers for the Copernican Suite.

This module centralises retrieval of the project's version string so that
all components report a consistent value. It queries ``importlib.metadata``
for the installed package version and falls back to ``"0+unknown"`` when the
metadata is missing (for example when running from an unpackaged source
checkout). The helper avoids scattering hard-coded version numbers across the
codebase and keeps logging and plot footers in sync with tagged releases.
"""

from importlib.metadata import PackageNotFoundError, version

PACKAGE_NAME = "copernican-suite"


def get_version() -> str:
    """Return the installed package version or a fallback.

    The function asks :mod:`importlib.metadata` for the version of the
    ``copernican-suite`` distribution. When that information cannot be found,
    a fallback string of ``"0+unknown"`` is returned so that users still see a
    version-like identifier in logs and plot footers.
    """

    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return "0+unknown"


__all__ = ["get_version"]
