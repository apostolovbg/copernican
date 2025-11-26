"""Workflow-focused DriftGuard rules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

import yaml

from driftguard.rules import Rule, RuleContext, Violation


class FullTestSuiteInCIRule(Rule):
    """Ensure CI workflows and policy docs require the full test suite."""

    name = "full-test-suite-in-ci"

    _WORKFLOW_PATH = Path(".github/workflows/ci.yml")
    _POLICY_DOC = Path("DRIFTGUARD.md")
    _PYTEST_PATTERNS = (r"python\s+-m\s+pytest", r"\bpytest\b")
    _UNITTEST_PATTERNS = (
        r"python\s+-m\s+unittest\s+discover",
        r"\bunittest\s+discover\b",
    )
    _POLICY_SUITE_PHRASE = "full program unit test suite"
    _DRIFTGUARD_PATTERN = r"driftguard\.cli\s+check"

    def _read_text(self, path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def _load_steps(self, workflow_text: str) -> list:
        try:
            parsed = yaml.safe_load(workflow_text)
        except yaml.YAMLError:
            return []

        jobs = parsed.get("jobs", {}) if isinstance(parsed, dict) else {}
        tests_job = jobs.get("tests", {}) if isinstance(jobs, dict) else {}
        steps = tests_job.get("steps", []) if isinstance(tests_job, dict) else []

        return steps if isinstance(steps, list) else []

    def _workflow_includes_pytest(self, workflow_text: str) -> bool:
        steps = self._load_steps(workflow_text)
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name", "")).strip().lower()
            run = str(step.get("run", "")).lower()
            if name == "pytest" and any(
                re.search(pattern, run) for pattern in self._PYTEST_PATTERNS
            ):
                return True
        normalized = workflow_text.lower()
        return any(re.search(pattern, normalized) for pattern in self._PYTEST_PATTERNS)

    def _workflow_includes_unittest(self, workflow_text: str) -> bool:
        steps = self._load_steps(workflow_text)
        for step in steps:
            if not isinstance(step, dict):
                continue
            name = str(step.get("name", "")).strip().lower()
            run = str(step.get("run", "")).lower()
            if name == "unit tests" and any(
                re.search(pattern, run) for pattern in self._UNITTEST_PATTERNS
            ):
                return True
        normalized = workflow_text.lower()
        return any(
            re.search(pattern, normalized) for pattern in self._UNITTEST_PATTERNS
        )

    def _workflow_orders_driftguard_after_tests(self, workflow_text: str) -> bool:
        steps = self._load_steps(workflow_text)
        pytest_index = None
        unittest_index = None
        driftguard_index = None

        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            run = str(step.get("run", "")).lower()
            name = str(step.get("name", "")).strip().lower()

            if pytest_index is None and name == "pytest" and any(
                re.search(pattern, run) for pattern in self._PYTEST_PATTERNS
            ):
                pytest_index = index

            if unittest_index is None and name == "unit tests" and any(
                re.search(pattern, run) for pattern in self._UNITTEST_PATTERNS
            ):
                unittest_index = index

            if driftguard_index is None and re.search(self._DRIFTGUARD_PATTERN, run):
                driftguard_index = index

        if driftguard_index is None or pytest_index is None or unittest_index is None:
            return False

        return driftguard_index > max(pytest_index, unittest_index)

    def _policy_mentions_full_suite(self, policy_text: str) -> bool:
        normalized = policy_text.lower()
        return (
            self._POLICY_SUITE_PHRASE in normalized
            and "/tests" in normalized
            and "before every commit" in normalized
            and "every task" in normalized
            and "python -m pytest -q" in normalized
            and "python -m unittest discover -v" in normalized
        )

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        workflow_path = repo_root / self._WORKFLOW_PATH
        policy_path = repo_root / self._POLICY_DOC

        workflow_text = self._read_text(workflow_path)
        policy_text = self._read_text(policy_path)

        violations: List[Violation] = []

        if not workflow_text:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CI workflow .github/workflows/ci.yml is missing; CI must "
                        "run the full program unit test suite."
                    ),
                    path=workflow_path,
                )
            )
        elif not self._workflow_includes_pytest(workflow_text):
            violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "CI must exercise the full program unit test suite under /tests; "
                            "add a documented python -m pytest invocation named Pytest to "
                            ".github/workflows/ci.yml."
                        ),
                        path=workflow_path,
                    )
            )
        elif not self._workflow_includes_unittest(workflow_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CI must also run python -m unittest discover -v, with the step "
                        "named Unit tests, to cover the full program unit test suite."
                    ),
                    path=workflow_path,
                )
            )
        elif not self._workflow_orders_driftguard_after_tests(workflow_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "DriftGuard must execute after the Pytest and Unit tests steps in "
                        "CI so policy checks run on the tested codebase."
                    ),
                    path=workflow_path,
                )
            )

        if not policy_text:
            violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            "DRIFTGUARD.md must document running the full program unit "
                            "test suite in /tests before each commit."
                        ),
                        path=policy_path,
                    )
                )
        elif not self._policy_mentions_full_suite(policy_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "Document that contributors should run the full program "
                        "unit test suite in /tests before every commit using both the "
                        "pytest and unittest discover commands."
                    ),
                    path=policy_path,
                )
            )

        return violations

    def metrics(self, context: RuleContext) -> List:  # pragma: no cover - no metrics
        return []
