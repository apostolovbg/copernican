"""DevCovenant policy: Ensure version sync across VERSION, README, CITATION.

This policy ensures that the canonical version in copernican_lib/VERSION
matches the version declared in README.md and CITATION.cff, preventing
version drift across documentation.
"""

import re
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation


class VersionSyncCheck(PolicyCheck):
    """Ensure README, CITATION and VERSION agree on the recorded version."""

    policy_id = "version-sync"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for version synchronization."""
        violations = []

        version_file = context.repo_root / "copernican_lib" / "VERSION"
        readme_file = context.repo_root / "README.md"
        citation_file = context.repo_root / "CITATION.cff"

        # Check that required files exist
        for target in (version_file, readme_file, citation_file):
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
                r"\*\*Version:\*\*\s*(?P<version>\d+\.\d+\.\d+)",
                readme_text,
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

        return violations
