"""DevCovenant policy: Keep dependency listings and license docs in sync."""

from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

DEPENDENCY_FILES = {"requirements.in", "requirements.lock", "pyproject.toml"}
THIRD_PARTY = Path("THIRD_PARTY_LICENSES.md")
LICENSES_DIR = "licenses"
LICENSE_REPORT_HEADING = "## License Report"


def _extract_license_report(text: str) -> str:
    """Extract the text inside the License Report section."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == LICENSE_REPORT_HEADING.lower():
            start = index
            break

    if start is None:
        return ""

    # Collect lines until the next section header
    section_lines: List[str] = [lines[start]]
    remaining = iter(lines)
    for _ in range(start + 1):
        next(remaining, None)
    for line in remaining:
        stripped = line.strip()
        header_prefix = stripped.startswith("## ")
        header_not_report = not stripped.lower().startswith(
            LICENSE_REPORT_HEADING.lower()
        )
        if header_prefix and header_not_report:
            break
        section_lines.append(line)

    return "\n".join(section_lines)


def _contains_reference(section: str, needle: str) -> bool:
    """Case-insensitive search inside the license report."""
    return needle.lower() in section.lower()


class DependencyLicenseSyncCheck(PolicyCheck):
    """Ensure dependency changes update licenses and the report section."""

    policy_id = "dependency-license-sync"
    version = "1.0.0"

    def check(self, context: CheckContext):
        """Verify dependency changes match the recorded license summary."""
        files = context.changed_files or []
        if not files:
            return []

        changed_dependency_files = {
            path.name for path in files if path.name in DEPENDENCY_FILES
        }
        if not changed_dependency_files:
            return []

        violations = []

        third_party_path = context.repo_root / THIRD_PARTY
        if not any(path.name == THIRD_PARTY.name for path in files):
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        "Dependencies changed without updating "
                        "the license table `THIRD_PARTY_LICENSES.md`."
                    ),
                )
            )

        license_dir_touched = False
        for path in files:
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == LICENSES_DIR:
                license_dir_touched = True
                break

        if not license_dir_touched:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message="License files under licenses/ must be refreshed.",
                )
            )

        if third_party_path.is_file():
            report = _extract_license_report(
                third_party_path.read_text(encoding="utf-8")
            )
        else:
            report = ""

        if not report:
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=(
                        "Add a '## License Report' section to "
                        "`THIRD_PARTY_LICENSES.md` that chronicles dependency "
                        "updates."
                    ),
                )
            )
        else:
            for dep_file in sorted(changed_dependency_files):
                if not _contains_reference(report, dep_file):
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            message=(
                                "The license report must mention changes to "
                                f"`{dep_file}` when dependencies are altered."
                            ),
                        )
                    )

        return violations
