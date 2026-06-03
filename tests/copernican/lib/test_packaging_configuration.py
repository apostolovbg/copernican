"""Verify packaging metadata guards against flat-layout auto discovery."""

import tomllib
import unittest
from pathlib import Path


class TestPackagingConfiguration(unittest.TestCase):
    """Exercise the setuptools package-discovery guard."""

    def test_pyproject_limits_package_discovery(self) -> None:
        config_path = Path("pyproject.toml")
        config_text = config_path.read_text(encoding="utf-8")
        config = tomllib.loads(config_text)
        finder = config["tool"]["setuptools"]["packages"]["find"]

        include = tuple(finder.get("include", ()))
        exclude = tuple(finder.get("exclude", ()))

        expected_include = (
            "copernican",
            "copernican.*",
        )
        expected_exclude = ("archive", "data", "licenses", "tests")

        self.assertEqual(include, expected_include)
        self.assertEqual(exclude, expected_exclude)


if __name__ == "__main__":
    unittest.main()
