"""Warn when string literals use bare backslashes instead of raw strings.
Encourage raw or explicitly escaped forms for safe literals.
"""

import re
import tokenize
from pathlib import Path
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

_STRING_PREFIX_RE = re.compile(r"(?P<prefix>[rubfRUBF]*)(?P<quote>['\"]{1,3})")
_SUSPICIOUS_ESCAPE_RE = re.compile(r"\\(?![\\'\"abfnrtv0-7xuUN])")


def _should_inspect(rel_path: Path) -> bool:
    """Decide if the file should be scanned for raw-string escapes."""
    if rel_path.parts and rel_path.parts[0] == "copernican_lib":
        if len(rel_path.parts) > 1 and rel_path.parts[1] == "vendor":
            return False
    if "tests" in rel_path.parts:
        return False
    if rel_path.suffix != ".py":
        return False
    return True


def _is_raw_literal(token_value: str) -> bool:
    """Return True if the literal is already raw."""
    match = _STRING_PREFIX_RE.match(token_value)
    if not match:
        return False
    return "r" in match.group("prefix").lower()


def _contains_suspicious_escape(token_value: str) -> bool:
    """Return True when a bare backslash appears outside standard escapes."""
    return bool(_SUSPICIOUS_ESCAPE_RE.search(token_value))


class RawStringEscapesCheck(PolicyCheck):
    """Warn when string literals contain bare backslashes."""

    policy_id = "raw-string-escapes"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Inspect tokens for suspicious escape sequences."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        for path in files:
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue
            if not _should_inspect(rel):
                continue

            try:
                with path.open(encoding="utf-8") as handle:
                    tokens = tokenize.generate_tokens(handle.readline)
                    for token in tokens:
                        if token.type != tokenize.STRING:
                            continue
                        token_text = token.string
                        if _is_raw_literal(token_text):
                            continue
                        if _contains_suspicious_escape(token_text):
                            violations.append(
                                Violation(
                                    policy_id=self.policy_id,
                                    severity="info",
                                    file_path=path,
                                    line_number=token.start[0],
                                    message=(
                                        "String literal has a bare backslash;"
                                        " use a raw string or double-escape"
                                        " the slash to avoid accidental"
                                        " escapes."
                                    ),
                                )
                            )
            except (OSError, tokenize.TokenError):
                continue

        return violations
