"""
Policy: Line Length Limit

Apply the configured `max_length` to the files selected by the unified
include/exclude metadata (suffixes, prefixes and globs).
"""

from typing import List, Sequence

from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet


class LineLengthLimitCheck(PolicyCheck):
    """Check that each targeted file honors the line-length limit."""

    policy_id = "line-length-limit"
    version = "1.0.0"

    MAX_LINE_LENGTH = 79
    DEFAULT_SUFFIXES = [".py", ".md", ".rst", ".txt"]
    DEFAULT_URL_PREFIXES = [
        "http://",
        "https://",
        "ftp://",
        "ftps://",
        "sftp://",
        "ssh://",
        "ws://",
        "wss://",
        "file://",
        "git://",
        "svn://",
        "mailto:",
        "tel:",
        "magnet:",
        "torrent:",
        "data:",
        "urn:",
    ]

    @classmethod
    def _max_length_option(cls, raw_value: object) -> int:
        """Return the configured max length, falling back on invalid input."""
        token = str(raw_value or "").strip()
        if not token:
            return cls.MAX_LINE_LENGTH
        try:
            parsed = int(token)
        except (TypeError, ValueError):
            return cls.MAX_LINE_LENGTH
        return parsed if parsed > 0 else cls.MAX_LINE_LENGTH

    @staticmethod
    def _truthy(raw_value: object) -> bool:
        """Return True for common boolean-like truthy tokens."""
        if isinstance(raw_value, bool):
            return raw_value
        token = str(raw_value or "").strip().lower()
        return token in {"1", "true", "yes", "on"}

    @staticmethod
    def _option_tokens(raw_value: object) -> list[str]:
        """Normalize one metadata option into a list of non-empty tokens."""
        if raw_value is None:
            return []
        if isinstance(raw_value, str):
            return [
                token.strip()
                for token in raw_value.replace("\n", ",").split(",")
                if token.strip()
            ]
        if isinstance(raw_value, (list, tuple, set)):
            tokens: list[str] = []
            for entry in raw_value:
                token = str(entry or "").strip()
                if token:
                    tokens.append(token)
            return tokens
        token = str(raw_value).strip()
        return [token] if token else []

    @classmethod
    def _between_pairs(cls, raw_value: object) -> list[tuple[str, str]]:
        """
        Normalize `left=>right` markers for `long_lines_between`.

        Invalid entries are ignored so malformed optional metadata does not
        break policy execution.
        """
        pairs: list[tuple[str, str]] = []
        for token in cls._option_tokens(raw_value):
            if "=>" not in token:
                continue
            left, right = token.split("=>", 1)
            left_token = left.strip()
            right_token = right.strip()
            if left_token and right_token:
                pairs.append((left_token, right_token))
        return pairs

    @staticmethod
    def _contains_any_marker(
        line_content: str,
        markers: Sequence[str],
    ) -> bool:
        """Return True when any configured marker appears in one line."""
        return any(marker and marker in line_content for marker in markers)

    @staticmethod
    def _matches_any_between_pair(
        line_content: str,
        between_pairs: Sequence[tuple[str, str]],
    ) -> bool:
        """Return True when one `left=>right` pair appears in-order."""
        for left, right in between_pairs:
            left_index = line_content.find(left)
            if left_index < 0:
                continue
            right_index = line_content.find(right, left_index + len(left))
            if right_index >= 0:
                return True
        return False

    @classmethod
    def _allow_long_line(
        cls,
        line_content: str,
        *,
        allow_long_url_lines: bool,
        url_prefixes: Sequence[str],
        allow_long_lines: bool,
        long_lines_contain: Sequence[str],
        long_lines_between: Sequence[tuple[str, str]],
    ) -> bool:
        """Return True when one configured escape hatch allows the line."""
        if allow_long_url_lines and cls._contains_any_marker(
            line_content,
            url_prefixes,
        ):
            return True
        if not allow_long_lines:
            return False
        if cls._contains_any_marker(line_content, long_lines_contain):
            return True
        return cls._matches_any_between_pair(line_content, long_lines_between)

    def _build_selector(self) -> SelectorSet:
        """Return the selector constructed from policy metadata."""
        defaults = {
            "include_suffixes": self.DEFAULT_SUFFIXES,
        }
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext) -> List[Violation]:
        """
        Check files for lines exceeding the length limit.

        Args:
            context: Check context

        Returns:
            List of violations
        """
        max_length = self._max_length_option(
            self.get_option("max_length", self.MAX_LINE_LENGTH)
        )
        allow_long_url_lines = self._truthy(
            self.get_option("allow_long_url_lines", False)
        )
        url_prefixes = self._option_tokens(self.get_option("url_prefixes", []))
        if not url_prefixes:
            url_prefixes = list(self.DEFAULT_URL_PREFIXES)
        allow_long_lines = self._truthy(
            self.get_option("allow_long_lines", False)
        )
        long_lines_contain = self._option_tokens(
            self.get_option("long_lines_contain", [])
        )
        long_lines_between = self._between_pairs(
            self.get_option("long_lines_between", [])
        )

        violations = []

        selector = self._build_selector()
        files_pool = context.all_files or context.changed_files or []
        files_to_check = [
            path
            for path in files_pool
            if path.is_file() and selector.matches(path, context.repo_root)
        ]

        for file_path in files_to_check:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            except OSError as error:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="warning",
                        file_path=file_path,
                        message=(
                            "Unable to read file while checking line length: "
                            f"{error}"
                        ),
                        can_auto_fix=False,
                    )
                )
                continue

            # Check each line
            for line_num, line in enumerate(lines, start=1):
                # Remove trailing newline for length check
                line_content = line.rstrip("\n")

                if len(line_content) > max_length:
                    if self._allow_long_line(
                        line_content,
                        allow_long_url_lines=allow_long_url_lines,
                        url_prefixes=url_prefixes,
                        allow_long_lines=allow_long_lines,
                        long_lines_contain=long_lines_contain,
                        long_lines_between=long_lines_between,
                    ):
                        continue
                    # Count how many lines are too long to avoid spam
                    # Only report first 5 per file
                    file_violations = [
                        v for v in violations if v.file_path == file_path
                    ]
                    if len(file_violations) >= 5:
                        continue

                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="warning",
                            file_path=file_path,
                            line_number=line_num,
                            message=(
                                f"Line exceeds {max_length} "
                                f"characters (current: {len(line_content)})"
                            ),
                            suggestion=(
                                "Break long lines into multiple lines or "
                                "refactor for clarity"
                            ),
                            can_auto_fix=False,
                        )
                    )

        return violations
