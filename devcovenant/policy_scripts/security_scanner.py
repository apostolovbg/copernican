"""Detect suspicious symbols that historically trigger compliance risks."""

import re
from typing import List, Sequence

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet

PATTERNS = [
    (
        re.compile(r"\beval\s*\("),
        "Avoid `eval`; prefer safer alternatives.",
    ),
    (
        re.compile(r"\bexec\s*\("),
        "Avoid `exec`; prefer explicit parsing.",
    ),
    (
        re.compile(r"\bpickle\.loads\s*\("),
        "Avoid untrusted `pickle.loads`.",
    ),
    (
        re.compile(r"\bsubprocess\.run\s*\([^)]*shell\s*=\s*True"),
        "Avoid `shell=True` in subprocess calls.",
    ),
]

ALLOW_COMMENT = "security-scanner: allow"


class SecurityScannerCheck(PolicyCheck):
    """Flag known insecure constructs that breach compliance guidelines."""

    policy_id = "security-scanner"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Search repository Python modules for risky expressions."""
        violations: List[Violation] = []
        files = context.all_files or context.changed_files or []
        selector = SelectorSet.from_policy(
            self, defaults={"include_suffixes": [".py"]}
        )

        for path in files:
            if not path.is_file():
                continue
            try:
                path.relative_to(context.repo_root)
            except ValueError:
                continue
            if not selector.matches(path, context.repo_root):
                continue

            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()
            for pattern, reason in PATTERNS:
                for match in pattern.finditer(text):
                    line_index = text.count("\n", 0, match.start())
                    if _has_allow_comment(lines, line_index):
                        continue
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=path,
                            line_number=line_index + 1,
                            message=(
                                "Insecure construct detected: "
                                f"{reason} (pattern `{pattern.pattern}`). "
                                "Review the compliance rationale before "
                                "committing."
                            ),
                        )
                    )

        return violations


def _has_allow_comment(lines: Sequence[str], line_index: int) -> bool:
    """Return True when this or a nearby line carries the allow flag."""
    for offset in (0, -1, -2):
        idx = line_index + offset
        if not (0 <= idx < len(lines)):
            continue
        if ALLOW_COMMENT in lines[idx]:
            return True
    return False
