"""Enforce documentation coverage via shared translator LanguageUnit."""

from __future__ import annotations

from typing import List

import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)
from devcovenant.core.selectors import SelectorSet


class DocstringAndCommentCoverageCheck(PolicyCheck):
    """Treat missing docstrings/comments as policy violations."""

    policy_id = "docstring-and-comment-coverage"
    version = "1.1.0"
    DEFAULT_SUFFIXES = [
        ".py",
        ".pyi",
        ".pyw",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".go",
        ".rs",
        ".java",
        ".cs",
    ]

    def _build_selector(self) -> SelectorSet:
        """Return the unified selector for this policy."""
        return SelectorSet.from_policy(
            self,
            defaults={"include_suffixes": self.DEFAULT_SUFFIXES},
        )

    def check(self, context: CheckContext) -> List[Violation]:
        """Detect symbols without docstrings/comments via LanguageUnit."""
        files = context.all_files or context.changed_files or []
        violations: List[Violation] = []
        selector = self._build_selector()
        runtime = context.translator_runtime
        if runtime is None:
            return violations

        for path in files:
            if not path.is_file() or not selector.matches(
                path, context.repo_root
            ):
                continue
            resolution = runtime.resolve(
                path=path,
                policy_id=self.policy_id,
                context=context,
            )
            if not resolution.is_resolved:
                violations.extend(resolution.violations)
                continue

            source = yaml_cache_service.read_text(path)
            unit = runtime.translate(
                resolution,
                path=path,
                source=source,
                context=context,
            )
            if unit is None:
                continue

            if not unit.module_documented:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        message=(
                            "Module lacks a descriptive top-level docstring "
                            "or preceding comment."
                        ),
                    )
                )

            for fact in unit.symbol_doc_facts:
                if fact.documented:
                    continue
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=path,
                        line_number=fact.line_number,
                        message=(
                            f"{fact.kind.title()} '{fact.name}' is missing "
                            "a docstring or adjacent explanatory comment."
                        ),
                    )
                )

        return violations
