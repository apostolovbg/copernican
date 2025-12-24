"""Detect suspicious symbols that historically trigger compliance risks."""

import re
from pathlib import PurePosixPath
from typing import List, Sequence

from devcovenant.base import CheckContext, PolicyCheck, Violation

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


def _should_scan(rel_path: PurePosixPath) -> bool:
    """Return True when the path points to a module we should scan."""
    if rel_path.suffix != ".py":
        return False
    if (
        rel_path.parts
        and rel_path.parts[0] == "copernican_lib"
        and len(rel_path.parts) > 1
    ):
        if rel_path.parts[1] == "vendor":
            return False
    if "tests" in rel_path.parts:
        return False
    return True


class SecurityScannerCheck(PolicyCheck):
    """Flag known insecure constructs that breach compliance guidelines."""

    policy_id = "security-scanner"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Search repository Python modules for risky expressions."""
        violations: List[Violation] = []
        files = context.all_files or context.changed_files or []

        for path in files:
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            rel_posix = PurePosixPath(rel.as_posix())
            if not _should_scan(rel_posix):
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
