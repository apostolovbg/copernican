"""
Policy: Last Updated

Ensures Last Updated markers appear only in allowlisted docs and stay current
for touched allowlisted files.
"""

from __future__ import annotations

import fnmatch
import re
from datetime import datetime, timezone
from typing import List, Set

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet


class LastUpdatedCheck(PolicyCheck):
    """Validate allowlisted Last Updated markers and recency."""

    policy_id = "last-updated"
    version = "2.0.0"

    LAST_UPDATED_PATTERN = re.compile(
        r"(\*\*Last Updated:\*\*|Last Updated:|# Last Updated)",
        re.IGNORECASE,
    )
    DATE_PATTERN = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
    MANAGED_BLOCK_BEGIN = "<!-- DEVCOV:BEGIN -->"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check Last Updated placement and date freshness."""
        violations: List[Violation] = []
        files_to_check = context.all_files or context.changed_files or []
        touched_files = self._resolve_touched_files(context)

        allowlist = set(
            self._normalize_list(self.get_option("allowed_files", []))
        )
        selectors = SelectorSet.from_policy(self)
        allowed_suffixes = {
            suffix if suffix.startswith(".") else f".{suffix}"
            for suffix in self._normalize_list(
                self.get_option("allowed_suffixes", [])
            )
        }
        allowed_globs = self._normalize_list(
            self.get_option("allowed_globs", [])
        )
        required_files = set(
            self._normalize_list(self.get_option("required_files", []))
        )
        required_globs = self._normalize_list(
            self.get_option("required_globs", [])
        )

        today = datetime.now(timezone.utc).date().isoformat()

        for file_path in files_to_check:
            if not selectors.matches(file_path, context.repo_root):
                continue

            try:
                relative_path = file_path.relative_to(
                    context.repo_root
                ).as_posix()
            except ValueError:
                continue

            is_allowlisted = self._is_allowlisted(
                file_path=file_path,
                relative_path=relative_path,
                allowlist=allowlist,
                allowed_suffixes=allowed_suffixes,
                allowed_globs=allowed_globs,
            )
            required = relative_path in required_files or self._glob_matches(
                relative_path, required_globs
            )

            try:
                text = file_path.read_text(encoding="utf-8")
            except OSError as error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=file_path,
                        message=(
                            "Unable to read file while checking Last Updated "
                            f"markers: {error}"
                        ),
                        can_auto_fix=False,
                    )
                )
                continue

            scan_lines = self._header_scan_lines(text)
            marker_line = None
            marker_text = ""
            for line_number, line in scan_lines:
                if self.LAST_UPDATED_PATTERN.search(line):
                    marker_line = line_number
                    marker_text = line
                    break

            if marker_line is not None and not is_allowlisted:
                allowed = self._format_allowlist_description(
                    allowlist=allowlist,
                    allowed_suffixes=allowed_suffixes,
                    allowed_globs=allowed_globs,
                )
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=file_path,
                        line_number=marker_line,
                        message=(
                            "Last Updated marker found in non-allowlisted "
                            "file"
                        ),
                        suggestion=(
                            "Remove Last Updated marker from this file "
                            f"(only allowed in: {allowed})"
                        ),
                    )
                )
                continue

            marker_date = None
            if marker_line is not None:
                date_match = self.DATE_PATTERN.search(marker_text)
                if date_match:
                    marker_date = date_match.group(0)
                else:
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="warning",
                            file_path=file_path,
                            line_number=marker_line,
                            message=(
                                "Last Updated marker missing ISO date "
                                "(YYYY-MM-DD)."
                            ),
                            can_auto_fix=True,
                        )
                    )

            if required and marker_line is None:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=file_path,
                        message=(
                            "Required documentation is missing a Last "
                            "Updated marker in its generated header zone."
                        ),
                        suggestion=(
                            "Add `**Last Updated:** YYYY-MM-DD` before the "
                            "managed block."
                        ),
                        can_auto_fix=True,
                    )
                )

            if (
                is_allowlisted
                and marker_date
                and relative_path in touched_files
                and marker_date != today
            ):
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=file_path,
                        message=(
                            "Last Updated marker should match the current "
                            "UTC date for touched docs."
                        ),
                        suggestion=(
                            "Update the Last Updated header to today's "
                            f"UTC date ({today})."
                        ),
                        can_auto_fix=True,
                    )
                )

        return violations

    def _header_scan_lines(self, text: str) -> list[tuple[int, str]]:
        """Return header-zone lines from file start to managed block begin."""
        lines = text.splitlines()
        results: list[tuple[int, str]] = []
        for idx, line in enumerate(lines, start=1):
            if line.strip() == self.MANAGED_BLOCK_BEGIN:
                break
            results.append((idx, line))
            if idx >= 25:
                break
        return results

    def _normalize_list(self, raw: object) -> List[str]:
        """Return a list of non-empty string tokens."""
        if raw is None:
            return []
        if isinstance(raw, str):
            candidates = raw.split(",")
        elif isinstance(raw, list):
            candidates = raw
        else:
            candidates = [raw]
        normalized: List[str] = []
        for entry in candidates:
            token = str(entry).strip()
            if token:
                normalized.append(token)
        return normalized

    def _glob_matches(self, rel_text: str, patterns: List[str]) -> bool:
        """Return True when rel_text matches one metadata glob pattern."""
        for pattern in patterns:
            if fnmatch.fnmatch(rel_text, pattern):
                return True
            if pattern.startswith("**/") and fnmatch.fnmatch(
                rel_text, pattern[3:]
            ):
                return True
        return False

    def _is_allowlisted(
        self,
        *,
        file_path,
        relative_path: str,
        allowlist: Set[str],
        allowed_suffixes: Set[str],
        allowed_globs: List[str],
    ) -> bool:
        """Return True when path is allowlisted by file/suffix/glob rules."""
        if relative_path in allowlist:
            return True
        if allowed_suffixes and file_path.suffix in allowed_suffixes:
            return True
        if allowed_globs and self._glob_matches(relative_path, allowed_globs):
            return True
        return False

    def _format_allowlist_description(
        self,
        *,
        allowlist: Set[str],
        allowed_suffixes: Set[str],
        allowed_globs: List[str],
    ) -> str:
        """Render a compact allowlist description for violation suggestions."""
        allowed = sorted(
            {
                *allowlist,
                *allowed_globs,
                *allowed_suffixes,
            }
        )
        return ", ".join(allowed) or "none"

    def _resolve_touched_files(self, context: CheckContext) -> Set[str]:
        """Resolve touched files for date freshness checks."""
        touched: Set[str] = set()
        for path in context.changed_files:
            try:
                touched.add(path.relative_to(context.repo_root).as_posix())
            except ValueError:
                continue
        return touched
