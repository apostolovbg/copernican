"""
Policy: Changelog Coverage

Routes each changed file to the proper changelog based on the metadata-defined
`main_changelog`, `skipped_files` and `collections` options.
"""

import subprocess
from pathlib import Path
from typing import Any, List

from devcovenant.base import CheckContext, PolicyCheck, Violation


def _latest_section(content: str) -> str:
    """Return the newest version section from a changelog."""

    marker = "## Version"
    search_start = 0
    log_marker = "## Log changes here"
    log_index = content.find(log_marker)
    if log_index != -1:
        search_start = log_index
    start = content.find(marker, search_start)
    if start == -1:
        start = content.find(marker)
        if start == -1:
            return content
    next_start = content.find("\n" + marker, start + len(marker))
    if next_start == -1:
        return content[start:]
    return content[start:next_start]


class ChangelogCoverageCheck(PolicyCheck):
    """Verify that modified files land in the appropriate changelog."""

    policy_id = "changelog-coverage"
    version = "2.2.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check if all changed files are documented in the relevant changelog.

        Args:
            context: Check context with repository info

        Returns:
            List of violations (empty if all files are documented)
        """
        violations: List[Violation] = []
        main_changelog_rel = Path(
            self.get_option("main_changelog", "CHANGELOG.md")
        )
        skip_option = self.get_option(
            "skipped_files",
            [
                "CHANGELOG.md",
                "rng_minigames/CHANGELOG.md",
                ".gitignore",
                ".pre-commit-config.yaml",
            ],
        )
        if isinstance(skip_option, str):
            skip_files = {
                entry.strip()
                for entry in skip_option.split(",")
                if entry.strip()
            }
        else:
            skip_files = set(skip_option or [])

        collections = self._resolve_collections(
            self.get_option(
                "collections",
                [
                    "rng_minigames/:rng_minigames/CHANGELOG.md:true",
                ],
            )
        )

        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", "HEAD"],
                cwd=context.repo_root,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [
                f for f in result.stdout.strip().split("\n") if f
            ]
        except Exception:
            return violations

        if not changed_files:
            return violations

        main_files: List[str] = []
        collection_files: List[List[str]] = [[] for _ in collections]

        for file_path in changed_files:
            if file_path in skip_files:
                continue
            assigned = False
            for index, entry in enumerate(collections):
                prefix = entry.get("prefix", "")
                if prefix and file_path.startswith(prefix):
                    collection_files[index].append(file_path)
                    assigned = True
                    break
            if not assigned:
                main_files.append(file_path)

        root_changelog = context.repo_root / main_changelog_rel
        should_read_root = (
            (main_files or any(collection_files))
            and root_changelog.exists()
        )
        root_content = (
            root_changelog.read_text(encoding="utf-8")
            if should_read_root
            else None
        )
        root_section = _latest_section(root_content) if root_content else None

        if main_files:
            if root_content is None:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        message=(
                            f"{main_changelog_rel.as_posix()} does not exist"
                        ),
                        suggestion=(
                            f"Create {main_changelog_rel.as_posix()} and "
                            "document the changes listed in the diff."
                        ),
                        can_auto_fix=False,
                    )
                )
            else:
                missing = [
                    path for path in main_files if path not in root_section
                ]
                if missing:
                    files_str = ", ".join(missing)
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=root_changelog,
                            message=(
                                "The following files are not mentioned in "
                                f"{main_changelog_rel.as_posix()}: "
                                f"{files_str}"
                            ),
                            suggestion=(
                                "Add entries to "
                                f"{main_changelog_rel.as_posix()} "
                                f"documenting changes to: {files_str}"
                            ),
                            can_auto_fix=False,
                        )
                    )

        for index, entry in enumerate(collections):
            files_for_collection = collection_files[index]
            changelog_rel = entry.get("changelog")
            changelog_path = context.repo_root / changelog_rel
            exclusive = entry.get("exclusive", True)

            changelog_content = (
                changelog_path.read_text(encoding="utf-8")
                if files_for_collection and changelog_path.exists()
                else None
            )
            changelog_section = (
                _latest_section(changelog_content)
                if changelog_content
                else None
            )

            if files_for_collection:
                if changelog_content is None:
                    prefix_label = (
                        entry.get("prefix") or "the configured prefix"
                    )
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            message=(
                                f"{changelog_rel.as_posix()} does not exist, "
                                f"but files under {prefix_label} changed"
                            ),
                            suggestion=(
                                f"Create {changelog_rel.as_posix()} and log "
                                "the updates recorded under that prefix."
                            ),
                            can_auto_fix=False,
                        )
                    )
                else:
                    missing_entries = [
                        path
                        for path in files_for_collection
                        if path not in changelog_section
                    ]
                    if missing_entries:
                        files_str = ", ".join(missing_entries)
                        violations.append(
                            Violation(
                                policy_id=self.policy_id,
                                severity="error",
                                file_path=changelog_path,
                                message=(
                                    "The following files are not mentioned in "
                                    f"{changelog_rel.as_posix()}: {files_str}"
                                ),
                                suggestion=(
                                    "Add entries to "
                                    f"{changelog_rel.as_posix()} documenting "
                                    f"changes to: {files_str}"
                                ),
                                can_auto_fix=False,
                            )
                        )

            if exclusive and root_section and files_for_collection:
                forbidden_mentions = [
                    path
                    for path in files_for_collection
                    if path in root_section
                ]
                if forbidden_mentions:
                    files_str = ", ".join(forbidden_mentions)
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=root_changelog,
                            message=(
                                "Files belonging to "
                                f"{changelog_rel.as_posix()} must not be "
                                "logged in the root changelog: "
                                f"{files_str}"
                            ),
                            suggestion=(
                                "Remove those entries from "
                                f"{main_changelog_rel.as_posix()} and log "
                                f"them only in {changelog_rel.as_posix()}."
                            ),
                            can_auto_fix=False,
                        )
                    )

        return violations

    def _resolve_collections(self, raw: Any) -> List[dict]:
        """Normalize metadata-configured collection entries."""
        default = [
            {
                "prefix": "rng_minigames/",
                "changelog": Path("rng_minigames/CHANGELOG.md"),
                "exclusive": True,
            }
        ]
        if raw is None:
            return default
        collections: List[dict] = []
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, str):
            entries = [
                item.strip() for item in raw.split(";") if item.strip()
            ]
        else:
            entries = default
        for entry in entries:
            if isinstance(entry, dict):
                prefix = entry.get("prefix", "")
                changelog = entry.get("changelog")
                if not changelog:
                    continue
                collections.append(
                    {
                        "prefix": prefix or "",
                        "changelog": Path(changelog),
                        "exclusive": entry.get("exclusive", True),
                    }
                )
            elif isinstance(entry, str):
                parts = entry.split(":")
                if len(parts) < 2:
                    continue
                prefix = parts[0]
                changelog = parts[1]
                exclusive = True
                if len(parts) >= 3:
                    exclusive = parts[2].lower() != "false"
                collections.append(
                    {
                        "prefix": prefix,
                        "changelog": Path(changelog),
                        "exclusive": exclusive,
                    }
                )
        return collections or default
