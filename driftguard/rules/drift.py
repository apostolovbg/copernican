# Last Updated: 2025-11-25
"""Drift metric rules for DriftGuard."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

from driftguard.rules import Metric, Rule, RuleContext, Violation
from driftguard.rules.metadata import _header_in_first_three, _utc_today
from driftguard.spec import MetricThreshold
from driftguard.utils import resolve_surface_globs


def _metric_threshold(
    context: RuleContext, name: str
) -> MetricThreshold | None:
    """Return the configured threshold for a drift metric if present."""

    return context.spec.drift.metrics.get(name)


def _python_lib_paths(context: RuleContext) -> List[Path]:
    """Resolve python-lib surface paths if defined in the spec."""

    if "python-lib" not in context.spec.surfaces:
        return []
    return resolve_surface_globs(context.spec, context.repo_root, "python-lib")


class TodoCountRule(Rule):
    """Track TODO occurrences for drift reporting."""

    name = "todo-count"
    _PATTERN = re.compile(r"\b(?:TODO|FIXME|XXX)\b", re.IGNORECASE)

    def _count_markers(self, paths: Sequence) -> int:
        count = 0
        for path in paths:
            try:
                text = path.read_text(encoding="utf-8")
            except (FileNotFoundError, UnicodeDecodeError):
                continue
            count += len(self._PATTERN.findall(text))
        return count

    def check(self, context: RuleContext) -> List[Violation]:
        threshold = _metric_threshold(context, self.name)
        if threshold is None or threshold.max_warning is None:
            return []

        metric = self.metrics(context)[0]
        if metric.value > threshold.max_warning:
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "TODO marker count exceeded the warning threshold "
                        f"({metric.value} > {threshold.max_warning})."
                    ),
                )
            ]
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        paths = _python_lib_paths(context)
        total = self._count_markers(paths)
        threshold = _metric_threshold(context, self.name)
        warning = threshold.max_warning if threshold else None
        return [Metric(name=self.name, value=total, threshold=warning)]


class TestCouplingRule(Rule):
    """Measure coupling between new modules and added tests."""

    name = "test-coupling-ratio"

    def _module_keys(self, path: Path) -> List[str]:
        relative = path.with_suffix("")
        parts = list(relative.parts)
        keys = [parts[-1]] if parts else []
        keys.append("_".join(parts))
        return [key for key in keys if key]

    def _test_keys(self, repo_root: Path) -> set[str]:
        keys: set[str] = set()
        for test_path in repo_root.glob("tests/**/*.py"):
            stem = test_path.stem
            keys.add(stem)
            if stem.startswith("test_"):
                keys.add(stem.removeprefix("test_"))
        return keys

    def _coupling_ratio(self, context: RuleContext) -> float:
        modules = [
            path
            for path in _python_lib_paths(context)
            if path.suffix == ".py" and path.name != "__init__.py"
        ]
        if not modules:
            return 0.0

        test_keys = self._test_keys(context.repo_root)
        covered = 0
        for module in modules:
            keys = self._module_keys(module.relative_to(context.repo_root))
            if any(key in test_keys for key in keys):
                covered += 1
        return covered / len(modules)

    def check(self, context: RuleContext) -> List[Violation]:
        threshold = _metric_threshold(context, self.name)
        if threshold is None or threshold.min_warning is None:
            return []

        ratio = self._coupling_ratio(context)
        if ratio < threshold.min_warning:
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "Test coupling fell below the warning threshold "
                        f"({ratio:.2f} < {threshold.min_warning})."
                    ),
                )
            ]
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        threshold = _metric_threshold(context, self.name)
        warning = threshold.min_warning if threshold else None
        ratio = self._coupling_ratio(context)
        return [Metric(name=self.name, value=ratio, threshold=warning)]


class DocAgeRule(Rule):
    """Report documentation age drift metrics."""

    name = "doc-age-days"

    def _doc_paths(self, context: RuleContext) -> List[Path]:
        if "docs" not in context.spec.surfaces:
            return []
        return resolve_surface_globs(context.spec, context.repo_root, "docs")

    def _oldest_doc_age(self, context: RuleContext) -> tuple[int, Path | None]:
        today = _utc_today()
        oldest_age = 0
        oldest_path = None
        for path in self._doc_paths(context):
            try:
                text = path.read_text(encoding="utf-8")
            except (FileNotFoundError, UnicodeDecodeError):
                continue
            _, header_date = _header_in_first_three(text)
            if header_date is None:
                continue
            age = (today - header_date).days
            if age > oldest_age:
                oldest_age = age
                oldest_path = path
        return oldest_age, oldest_path

    def check(self, context: RuleContext) -> List[Violation]:
        threshold = _metric_threshold(context, self.name)
        if threshold is None or threshold.max_warning is None:
            return []

        age, path = self._oldest_doc_age(context)
        if age > threshold.max_warning:
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "Documentation age exceeded the warning threshold "
                        f"({age} > {threshold.max_warning})."
                    ),
                    path=path,
                )
            ]
        return []

    def metrics(self, context: RuleContext) -> List[Metric]:
        threshold = _metric_threshold(context, self.name)
        warning = threshold.max_warning if threshold else None
        age, path = self._oldest_doc_age(context)
        return [
            Metric(name=self.name, value=age, path=path, threshold=warning)
        ]
