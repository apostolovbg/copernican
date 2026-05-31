"""Version helpers for Copernican.

The version comes from the tracked ``copernican/VERSION`` file so runtime
metadata stays aligned with the package contents.
"""

from importlib import resources
from pathlib import Path
from typing import Optional

VERSION_FILENAME = "VERSION"


def _read_version_file() -> Optional[str]:
    """Return the version stored alongside the package if available."""

    candidates = []
    package = __package__ or "copernican"
    try:
        candidates.append(resources.files(package).joinpath(VERSION_FILENAME))
    except (AttributeError, ImportError, ValueError):
        pass
    candidates.append(Path(__file__).with_name(VERSION_FILENAME))

    for candidate in candidates:
        try:
            if hasattr(candidate, "read_text"):
                version_text = candidate.read_text(encoding="utf-8")
            else:  # pragma: no cover - safety net for unexpected types
                version_text = Path(candidate).read_text(encoding="utf-8")
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            continue
        version_str = version_text.strip()
        if version_str:
            return version_str
    return None


def get_version() -> str:
    """Return the Copernican version string."""

    file_version = _read_version_file()
    if file_version:
        return file_version
    return "0+unknown"


__all__ = ["get_version"]
