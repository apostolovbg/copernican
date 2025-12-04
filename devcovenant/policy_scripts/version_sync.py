"""DevCovenant policy: Ensure version sync across metadata files.

This policy ensures the canonical version in copernican_lib/VERSION
matches the metadata declarations in README.md, pyproject.toml and
CITATION.cff, preventing version drift across the project docs and tools.
"""

import re
from pathlib import Path
from typing import List, Optional

from devcovenant.base import CheckContext, PolicyCheck, Violation

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore[assignment]


class VersionSyncCheck(PolicyCheck):
    """Ensure README, pyproject, CITATION and VERSION agree on the version."""

    policy_id = "version-sync"
    version = "1.2.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for version synchronization across metadata files."""
        violations: List[Violation] = []

        version_file = context.repo_root / "copernican_lib" / "VERSION"
        readme_file = context.repo_root / "README.md"
        citation_file = context.repo_root / "CITATION.cff"
        pyproject_file = context.repo_root / "pyproject.toml"

        # Verify required files exist
        required_targets = (
            version_file,
            readme_file,
            citation_file,
            pyproject_file,
        )
        for target in required_targets:
            if not target.exists():
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=target,
                        message="Required metadata file missing",
                    )
                )
                return violations

        # Read canonical version
        try:
            version = version_file.read_text(encoding="utf-8").strip()
        except OSError as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=version_file,
                    message=f"Cannot read VERSION file: {exc}",
                )
            )
            return violations

        # Check README version
        try:
            readme_text = readme_file.read_text(encoding="utf-8")
            readme_match = re.search(
                r"^\s*\*\*Version:\*\*\s*(?P<version>\d+\.\d+\.\d+)",
                readme_text,
                flags=re.MULTILINE,
            )
            if not readme_match:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=readme_file,
                        message="Missing Version header",
                    )
                )
            elif readme_match.group("version") != version:
                readme_version = readme_match.group("version")
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=readme_file,
                        message=(
                            f"Version {readme_version} does not match "
                            f"copernican_lib/VERSION ({version})"
                        ),
                    )
                )
        except OSError as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=readme_file,
                    message=f"Cannot read README.md: {exc}",
                )
            )

        # Check pyproject version
        try:
            pyproject_version = self._read_pyproject_version(pyproject_file)
            if not pyproject_version:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=pyproject_file,
                        message="pyproject.toml lacks project.version",
                    )
                )
            elif pyproject_version != version:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=pyproject_file,
                        message=(
                            f"pyproject.toml version {pyproject_version} "
                            f"does not match copernican_lib/VERSION "
                            f"({version})"
                        ),
                    )
                )
        except (OSError, tomllib.TOMLDecodeError) as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=pyproject_file,
                    message=f"Cannot read pyproject.toml: {exc}",
                )
            )

        # Check CITATION.cff versions
        try:
            citation_text = citation_file.read_text(encoding="utf-8")
            citation_regex = r"version:\s*\"(?P<version>\d+\.\d+\.\d+)\""
            citation_pattern = re.compile(citation_regex)
            citation_matches = citation_pattern.findall(citation_text)

            if len(citation_matches) < 2:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=citation_file,
                        message=(
                            "Must declare project and "
                            "preferred-citation versions"
                        ),
                    )
                )
            else:
                unique_versions = set(citation_matches)
                if len(unique_versions) != 1 or version not in unique_versions:
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=citation_file,
                            message=(
                                f"Versions {unique_versions} out of sync "
                                f"with VERSION ({version})"
                            ),
                        )
                    )
        except OSError as exc:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=citation_file,
                    message=f"Cannot read CITATION.cff: {exc}",
                )
            )

        # Prevent runtime code from hard-coding the suite version.
        runtime_hits: List[Path] = []
        for runtime_file in self._runtime_python_files(context.repo_root):
            try:
                contents = runtime_file.read_text(encoding="utf-8")
            except OSError as exc:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=runtime_file,
                        message=f"Cannot read runtime file: {exc}",
                    )
                )
                continue

            if version in contents:
                runtime_hits.append(runtime_file)

        for runtime_file in runtime_hits:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=runtime_file,
                    message=(
                        f"Hard-coded suite version {version}; "
                        "call copernican_lib.version.get_version() instead"
                    ),
                )
            )

        return violations

    @staticmethod
    def _read_pyproject_version(pyproject_path: Path) -> Optional[str]:
        """Return the project.version from pyproject.toml."""
        raw = pyproject_path.read_text(encoding="utf-8")
        data = tomllib.loads(raw)
        project = data.get("project")
        if not isinstance(project, dict):
            return None
        return project.get("version")

    def _runtime_python_files(self, repo_root: Path) -> List[Path]:
        """Return Python files that represent runtime entrypoints."""
        runtime_files: List[Path] = []

        copernican_entry = repo_root / "copernican.py"
        if copernican_entry.is_file():
            runtime_files.append(copernican_entry)

        for folder_name in ("copernican_lib", "engines"):
            root = repo_root / folder_name
            if not root.exists():
                continue
            runtime_files.extend(root.rglob("*.py"))

        return runtime_files
