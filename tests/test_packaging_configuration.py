"""Verify packaging metadata guards against flat-layout auto discovery.

The regression guards the macOS launcher, which still bootstraps with
setuptools 79.0.1 via ``ensurepip``. That legacy release aborts
installation when projects rely on implicit discovery in a flat repository
layout. The test ensures ``pyproject.toml`` pins the allowed namespaces so
the "Multiple top-level packages discovered" guard never reappears.
"""

from pathlib import Path
import tomllib


def test_pyproject_limits_package_discovery() -> None:
    """Ensure setuptools only sees the intended namespaces during packaging."""
    config_path = Path("pyproject.toml")
    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    finder = config["tool"]["setuptools"]["packages"]["find"]

    include = tuple(finder.get("include", ()))
    exclude = tuple(finder.get("exclude", ()))

    assert include == (
        "copernican_lib",
        "copernican_lib.*",
        "engines",
        "engines.*",
        "models",
    )

    for unwanted in ("archive", "data", "licenses", "tests"):
        assert unwanted in exclude
