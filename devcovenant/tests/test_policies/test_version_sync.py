"""Tests for version_sync policy."""

import tempfile
import unittest
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.version_sync import VersionSyncCheck


class TestVersionSyncPolicy(unittest.TestCase):
    """Test suite for VersionSyncCheck."""

    def _write_pyproject(self, repo_root: Path, version: str) -> Path:
        """Create a minimal pyproject.toml with the requested version."""
        pyproject = repo_root / "pyproject.toml"
        pyproject.write_text(f'[project]\nversion = "{version}"\n')
        return pyproject

    def test_detects_version_mismatch(self):
        """Policy should detect version mismatches."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            # Create VERSION file
            version_dir = repo_root / "copernican_lib"
            version_dir.mkdir()
            version_file = version_dir / "VERSION"
            version_file.write_text("1.0.0\n")

            # Create README with different version
            readme = repo_root / "README.md"
            readme.write_text("**Version:** 2.0.0\n")

            # Create CITATION.cff with matching versions
            citation = repo_root / "CITATION.cff"
            citation.write_text('version: "1.0.0"\nversion: "1.0.0"\n')
            self._write_pyproject(repo_root, "1.0.0")
            self._write_pyproject(repo_root, "1.0.0")

            context = CheckContext(repo_root=repo_root)
            policy = VersionSyncCheck()
            violations = policy.check(context)

            self.assertGreater(len(violations), 0)

    def test_allows_matching_versions(self):
        """Policy should pass when versions match."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            version_dir = repo_root / "copernican_lib"
            version_dir.mkdir()
            version_file = version_dir / "VERSION"
            version_file.write_text("1.0.0\n")

            readme = repo_root / "README.md"
            readme.write_text("**Version:** 1.0.0\n")

            citation = repo_root / "CITATION.cff"
            citation.write_text('version: "1.0.0"\nversion: "1.0.0"\n')

            context = CheckContext(repo_root=repo_root)
            policy = VersionSyncCheck()
            violations = policy.check(context)

            version_errs = [
                v for v in violations if "does not match" in v.message
            ]
            self.assertEqual(len(version_errs), 0)

    def test_detects_pyproject_mismatch(self):
        """Policy should flag mismatched pyproject versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            version_dir = repo_root / "copernican_lib"
            version_dir.mkdir()
            version_file = version_dir / "VERSION"
            version_file.write_text("1.0.0\n")

            readme = repo_root / "README.md"
            readme.write_text("**Version:** 1.0.0\n")

            citation = repo_root / "CITATION.cff"
            citation.write_text('version: "1.0.0"\nversion: "1.0.0"\n')

            self._write_pyproject(repo_root, "2.0.0")

            context = CheckContext(repo_root=repo_root)
            policy = VersionSyncCheck()
            violations = policy.check(context)

            mismatch = [
                v for v in violations if "pyproject.toml version" in v.message
            ]
            self.assertTrue(mismatch)

    def test_flags_hardcoded_runtime_version(self):
        """Policy should reject hard-coded versions in runtime code."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)

            version_dir = repo_root / "copernican_lib"
            version_dir.mkdir()
            version_file = version_dir / "VERSION"
            version_file.write_text("1.0.0\n")

            readme = repo_root / "README.md"
            readme.write_text("**Version:** 1.0.0\n")

            citation = repo_root / "CITATION.cff"
            citation.write_text('version: "1.0.0"\nversion: "1.0.0"\n')

            self._write_pyproject(repo_root, "1.0.0")

            runtime_file = repo_root / "copernican.py"
            runtime_file.write_text('APP_VERSION = "1.0.0"\n')

            context = CheckContext(repo_root=repo_root)
            policy = VersionSyncCheck()
            violations = policy.check(context)

            hardcoded = [
                v
                for v in violations
                if "Hard-coded suite version" in v.message
            ]
            self.assertEqual(len(hardcoded), 1)
            self.assertEqual(hardcoded[0].file_path, runtime_file)
