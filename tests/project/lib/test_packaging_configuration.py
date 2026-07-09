"""Verify packaging metadata guards against flat-layout auto discovery."""

import tomllib
import unittest
from pathlib import Path

from tests.project import filesystem_helpers


class TestPackagingConfiguration(unittest.TestCase):
    """Exercise the setuptools package-discovery guard."""

    def test_pyproject_limits_package_discovery(self) -> None:
        config_path = Path("pyproject.toml")
        config_text = filesystem_helpers.read_text(config_path)
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

    def test_console_script_targets_cli_main(self) -> None:
        config_path = Path("pyproject.toml")
        config = tomllib.loads(filesystem_helpers.read_text(config_path))
        scripts = config["project"]["scripts"]
        package_data = config["tool"]["setuptools"]["package-data"]

        self.assertEqual(scripts["copernican"], "copernican.cli:main")
        self.assertIn("global_settings/**/*", package_data["copernican.lib"])


if __name__ == "__main__":
    unittest.main()
