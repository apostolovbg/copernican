"""Warn when targeted string literals use bare backslashes instead of
raw strings."""

import re
import tokenize
from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet

_STRING_PREFIX_RE = re.compile(r"(?P<prefix>[rubfRUBF]*)(?P<quote>['\"]{1,3})")
_SUSPICIOUS_ESCAPE_RE = re.compile(r"\\(?![\\'\"abfnrtv0-7xuUN])")


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
    DEFAULT_SUFFIXES = [".py"]

    def _build_selector(self) -> SelectorSet:
        """Return the selector describing files to scan."""
        defaults = {
            "include_suffixes": self.DEFAULT_SUFFIXES,
        }
        return SelectorSet.from_policy(self, defaults=defaults)

    def check(self, context: CheckContext) -> List[Violation]:
        """Inspect tokens for suspicious escape sequences."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        selector = self._build_selector()
        for path in files:
            if not path.is_file():
                continue
            if not selector.matches(path, context.repo_root):
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
                                    severity="warning",
                                    file_path=path,
                                    line_number=token.start[0],
                                    message=(
                                        "String literal has a bare backslash;"
                                        " use a raw string or double-escape"
                                        " the slash to avoid accidental"
                                        " escapes."
                                    ),
                                    can_auto_fix=True,
                                    context={
                                        "start": token.start,
                                        "end": token.end,
                                    },
                                )
                            )
            except (OSError, tokenize.TokenError):
                continue

        return violations
