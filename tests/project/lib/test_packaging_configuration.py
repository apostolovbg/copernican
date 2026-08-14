"""Verify packaging metadata guards against flat-layout auto discovery."""

import tomllib
import unittest
from pathlib import Path

import yaml

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

    def test_camb_is_a_workspace_reference_dependency_only(self) -> None:
        """CAMB should stay out of the production dependency surface."""

        config = tomllib.loads(
            filesystem_helpers.read_text(Path("pyproject.toml"))
        )
        runtime_dependencies = {
            str(requirement).split("==", maxsplit=1)[0].lower()
            for requirement in config["project"]["dependencies"]
        }
        workspace_manifest = filesystem_helpers.read_text(
            Path("requirements.in")
        )
        workspace_lock = filesystem_helpers.read_text(
            Path("requirements.lock")
        )
        package_lock = filesystem_helpers.read_text(
            Path("copernican/runtime-requirements.lock")
        )

        self.assertNotIn("camb", runtime_dependencies)
        self.assertIn("\ncamb==1.6.0\n", workspace_manifest)
        self.assertIn("\ncamb==1.6.0 \\\n", workspace_lock)
        self.assertNotIn("\ncamb==", package_lock)

    def test_reference_licenses_follow_their_dependency_surface(self) -> None:
        """Only the workspace license inventory should contain CAMB."""

        workspace_report = filesystem_helpers.read_text(
            Path("licenses/THIRD_PARTY_LICENSES.md")
        )
        package_report = filesystem_helpers.read_text(
            Path("copernican/lib/licenses/THIRD_PARTY_LICENSES.md")
        )

        self.assertIn("`camb==1.6.0`", workspace_report)
        self.assertTrue(Path("licenses/camb-1.6.0.txt").is_file())
        self.assertNotIn("`camb==", package_report)
        self.assertFalse(
            Path("copernican/lib/licenses/camb-1.6.0.txt").exists()
        )

    def test_reference_code_and_workspace_licenses_are_not_packaged(
        self,
    ) -> None:
        """Package discovery should exclude references and workspace assets."""

        config = tomllib.loads(
            filesystem_helpers.read_text(Path("pyproject.toml"))
        )
        setuptools_config = config["tool"]["setuptools"]
        manifest = filesystem_helpers.read_text(Path("MANIFEST.in"))

        self.assertNotIn("data-files", setuptools_config)
        self.assertIn("prune tests", manifest)
        self.assertNotIn("recursive-include licenses *", manifest)
        self.assertIn("prune licenses", manifest)
        self.assertIn("global-exclude *.py[cod]", manifest)
        self.assertIn("global-exclude */__pycache__/*", manifest)
        excluded_package_data = config["tool"]["setuptools"][
            "exclude-package-data"
        ]["*"]
        self.assertIn("**/*.py[cod]", excluded_package_data)
        self.assertIn("**/__pycache__/*", excluded_package_data)
        reference_helper = Path("tests/project/lib/camb_reference.py")
        self.assertTrue(reference_helper.is_file())
        self.assertFalse(Path("copernican/lib/camb_reference.py").exists())
        self.assertFalse(Path("copernican/lib/camb_contract.py").exists())
        self.assertFalse(
            Path("copernican/lib/likelihoods/cmb/camb_solver.py").exists()
        )

    def test_package_readme_is_synced_from_root(self) -> None:
        """Package documentation must mirror the canonical root README."""

        root_readme = filesystem_helpers.read_text(Path("README.md"))
        package_readme = filesystem_helpers.read_text(
            Path("copernican/README.md")
        )
        profile = yaml.safe_load(
            filesystem_helpers.read_text(
                Path(
                    "devcovenant/custom/profiles/userproject/"
                    "userproject.yaml"
                )
            )
        )
        sync_pairs = profile["policy_overlays"]["package-doc-sync"][
            "sync_pairs"
        ]

        self.assertEqual(package_readme, root_readme)
        self.assertIn("**Doc Type:** repo-readme", package_readme)
        self.assertIn("## Repository Layout", package_readme)
        self.assertIn("## Repository Policy", package_readme)
        self.assertIn(
            "README.md=>copernican/README.md",
            sync_pairs,
        )


if __name__ == "__main__":
    unittest.main()
