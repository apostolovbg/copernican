"""DevCovenant policy: Detect future dates in Last Updated headers.

This policy ensures that Last Updated timestamps never extend into the
future, which would indicate a dating error or premature commitment.
"""

import datetime as dt
import re
from pathlib import Path

from devcovenant.base import PolicyScript, Violation


DATE_PATTERN = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


class NoFutureDatesPolicy(PolicyScript):
    """Prevent future dates in Last Updated and date-released fields."""

    def check(self, file_paths: list[Path]) -> list[Violation]:
        """Check for future dates in the provided files."""
        violations = []
        today = dt.datetime.now(dt.timezone.utc).date()

        for path in file_paths:
            if not path.is_file():
                continue

            # Skip test files
            try:
                rel_path = path.relative_to(self.repo_root)
                if rel_path.parts and rel_path.parts[0] == "tests":
                    continue
            except ValueError:
                pass

            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            for match in DATE_PATTERN.finditer(text):
                # Find the line containing this date
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end == -1:
                    line_end = len(text)
                context = text[line_start:line_end].lower()

                # Only check dates in Last Updated or date-released contexts
                if (
                    "last updated" not in context
                    and "date-released" not in context
                ):
                    continue

                # Parse and validate the date
                year, month, day = (
                    int(part) for part in match.group(0).split("-")
                )
                try:
                    candidate = dt.date(year, month, day)
                except ValueError:
                    # Invalid date, skip (other policies may catch this)
                    continue

                if candidate > today:
                    violations.append(
                        Violation(
                            policy_id="no-future-dates",
                            path=path,
                            message=(
                                f"Contains future date "
                                f"{candidate.isoformat()} "
                                f"(today is {today.isoformat()})"
                            ),
                        )
                    )
                    break  # Only report once per file

        return violations


def run(repo_root: Path, file_paths: list[Path]) -> list[Violation]:
    """Entry point for DevCovenant engine."""
    policy = NoFutureDatesPolicy(repo_root)
    return policy.check(file_paths)
