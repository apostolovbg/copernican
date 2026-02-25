"""DevCovenant policy: Detect future dates in Last Updated headers.

This policy ensures that Last Updated timestamps never extend into the
future, which would indicate a dating error or premature commitment.
"""

import datetime as dt
import re
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

DATE_PATTERN = re.compile(r"\b(19|20)\d{2}-\d{2}-\d{2}\b")


class NoFutureDatesCheck(PolicyCheck):
    """Prevent future dates in Last Updated and date-released fields."""

    policy_id = "no-future-dates"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Check for future dates in the provided files."""
        violations = []
        today = dt.datetime.now(dt.timezone.utc).date()

        file_paths = context.changed_files or context.all_files

        for path in file_paths:
            if not path.is_file():
                continue

            # Skip test files
            try:
                rel_path = path.relative_to(context.repo_root)
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
                context_line = text[line_start:line_end].lower()

                # Only check dates in Last Updated or date-released
                if (
                    "last updated" not in context_line
                    and "date-released" not in context_line
                ):
                    continue

                # Parse and validate the date
                year, month, day = (
                    int(part) for part in match.group(0).split("-")
                )
                try:
                    candidate = dt.date(year, month, day)
                except ValueError:
                    continue

                if candidate > today:
                    line_number = text.count("\n", 0, match.start()) + 1
                    violation_context = {
                        "match": match.group(0),
                        "field": (
                            "last-updated"
                            if "last updated" in context_line
                            else "date-released"
                        ),
                    }
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=path,
                            line_number=line_number,
                            message=(
                                f"Contains future date "
                                f"{candidate.isoformat()} "
                                f"(today is {today.isoformat()})"
                            ),
                            can_auto_fix=True,
                            context=violation_context,
                        )
                    )
                    break  # Only report once per file

        return violations
