"""Workflow-focused DriftGuard rules."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Set

import yaml

from driftguard.rules import Rule, RuleContext, Violation
from driftguard.utils import resolve_surface_globs


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

            if (
                pytest_index is None
                and name == "pytest"
                and any(re.search(pattern, run) for pattern in self._PYTEST_PATTERNS)
            ):
                pytest_index = index

            if (
                unittest_index is None
                and name == "unit tests"
                and any(re.search(pattern, run) for pattern in self._UNITTEST_PATTERNS)
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
                        "CI workflow .github/workflows/ci.yml is missing; "
                        "CI must run the full program unit test suite."
                    ),
                    path=workflow_path,
                )
            )
        elif not self._workflow_includes_pytest(workflow_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CI must exercise the full program unit "
                        "test suite under /tests; add a documented "
                        "python -m pytest invocation named Pytest "
                        "to .github/workflows/ci.yml."
                    ),
                    path=workflow_path,
                )
            )
        elif not self._workflow_includes_unittest(workflow_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "CI must also run python -m unittest discover -v, "
                        "with the step named Unit tests, to cover the full "
                        "program unit test suite."
                    ),
                    path=workflow_path,
                )
            )
        elif not self._workflow_orders_driftguard_after_tests(workflow_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "DriftGuard must execute after the Pytest and "
                        "Unit tests steps in CI so policy checks run "
                        "on the tested codebase."
                    ),
                    path=workflow_path,
                )
            )

        if not policy_text:
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "DRIFTGUARD.md must document running the full program "
                        "unit test suite in /tests before each commit."
                    ),
                    path=policy_path,
                )
            )
        elif not self._policy_mentions_full_suite(policy_text):
            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "Document that contributors should run the full "
                        "program unit test suite in /tests before every "
                        "commit using both the pytest and unittest discover "
                        "commands."
                    ),
                    path=policy_path,
                )
            )

        return violations

    # pragma: no cover - no metrics
    def metrics(self, context: RuleContext) -> List:
        return []


class DriftGuardPrecommitRequiredRule(Rule):
    """Ensure policy docs tell contributors to run DriftGuard pre-commit."""

    name = "driftguard-precommit-required"
    _POLICY_DOCS = (
        Path("DRIFTGUARD.md"),
        Path("README.md"),
        Path("CONTRIBUTING.md"),
        Path("AGENTS.md"),
    )

    def _mentions_full_run(self, text: str) -> bool:
        normalized = text.lower()
        commit_phrase = (
            "before every commit" in normalized
            or "before each commit" in normalized
        )
        return "driftguard" in normalized and "full" in normalized and commit_phrase

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for relative_path in self._POLICY_DOCS:
            doc_path = context.repo_root / relative_path
            try:
                text = doc_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            f"{relative_path.as_posix()} must document "
                            "running the full DriftGuard check before each "
                            "commit."
                        ),
                        path=doc_path,
                    )
                )
                continue

            if self._mentions_full_run(text):
                continue

            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        f"Document the full DriftGuard run before commits in "
                        f"{relative_path.as_posix()} alongside the test "
                        "commands."
                    ),
                    path=doc_path,
                )
            )

        return violations


class TestsAndDriftGuardFailuresRemediatedRule(Rule):
    """Require documentation to mandate fixing policy and test failures."""

    name = "tests-and-driftguard-failures-remediated"
    _POLICY_DOCS = (Path("DRIFTGUARD.md"), Path("driftguard/repo_policy.yml"))

    def _mentions_remediation(self, text: str) -> bool:
        normalized = text.lower()
        return (
            "python -m pytest -q" in normalized
            and "python -m unittest discover -v" in normalized
            and "driftguard check --scope=staged --mode=full" in normalized
            and ("failure" in normalized or "violation" in normalized)
            and ("fix" in normalized or "remediat" in normalized)
            and (
                "before any commit" in normalized
                or "before every commit" in normalized
                or "before each commit" in normalized
                or "before committing" in normalized
            )
        )

    def check(self, context: RuleContext) -> List[Violation]:
        violations: List[Violation] = []
        for relative_path in self._POLICY_DOCS:
            doc_path = context.repo_root / relative_path
            try:
                text = doc_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                violations.append(
                    Violation(
                        rule_name=self.name,
                        message=(
                            f"{relative_path.as_posix()} must state that test and "
                            "DriftGuard failures are fixed before committing."
                        ),
                        path=doc_path,
                    )
                )
                continue

            if self._mentions_remediation(text):
                continue

            violations.append(
                Violation(
                    rule_name=self.name,
                    message=(
                        "Document that failures from python -m pytest -q, "
                        "python -m unittest discover -v, and driftguard check "
                        "--scope=staged --mode=full must be remediated before "
                        "committing."
                    ),
                    path=doc_path,
                )
            )

        return violations


class DependencyLicenseAuditRule(Rule):
    """Require license updates when dependencies change."""

    name = "dependency-license-audit"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []
        if status.returncode != 0:
            return []

        changed: List[Path] = []
        for line in status.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            changed.append(repo_root / parts[1])

        dependency_files = {
            repo_root / "requirements.in",
            repo_root / "pyproject.toml",
            repo_root / "requirements.lock",
        }
        if not dependency_files & {path.resolve() for path in changed}:
            return []

        licenses = repo_root / "THIRD_PARTY_LICENSES.md"
        if licenses.resolve() in {path.resolve() for path in changed}:
            return []

        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Update THIRD_PARTY_LICENSES.md and licenses/ when "
                    "dependencies change."
                ),
                path=licenses if licenses.exists() else repo_root,
            )
        ]


class DependencyRefreshRule(Rule):
    """Require lockfile refresh when dependency manifests change."""

    name = "dependency-refresh"

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []
        if status.returncode != 0:
            return []

        changed: List[Path] = []
        for line in status.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            changed.append(repo_root / parts[1])

        manifests_changed = any(
            path.name in {"requirements.in", "pyproject.toml"} for path in changed
        )
        lock_changed = any(path.name == "requirements.lock" for path in changed)
        if manifests_changed and not lock_changed:
            return [
                Violation(
                    rule_name=self.name,
                    message=(
                        "Regenerate requirements.lock when dependency "
                        "manifests change."
                    ),
                    path=repo_root / "requirements.lock",
                )
            ]
        return []


class FormatterCleanRule(Rule):
    """Require Python sources to be Black-clean before committing."""

    name = "formatter-clean"

    _BLACK_CMD = ["black", "--check", "--diff"]

    def _surfaces(self, spec) -> Set[str]:
        return {
            name
            for name, surface in spec.surfaces.items()
            if self.name in surface.rules
        }

    def _surface_files(self, context: RuleContext) -> List[Path]:
        repo_root = context.repo_root
        targets: Set[Path] = set()
        for surface_name in self._surfaces(context.spec):
            for path in resolve_surface_globs(context.spec, repo_root, surface_name):
                if path.is_file() and path.suffix == ".py":
                    targets.add(path)
        return sorted(targets)

    def _collect_changed_python(
        self, repo_root: Path, allowed: Set[Path]
    ) -> List[Path]:
        try:
            status = subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return []

        if status.returncode != 0:
            return []

        changed: List[Path] = []
        for line in status.stdout.splitlines():
            if not line.strip():
                continue
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            path = repo_root / parts[1]
            if path.suffix == ".py" and (not allowed or path in allowed):
                changed.append(path)

        return changed

    def _run_black_check(self, repo_root: Path, files: List[Path]) -> bool:
        try:
            result = subprocess.run(
                self._BLACK_CMD + [str(path) for path in files],
                cwd=repo_root,
                check=False,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            return False

        return result.returncode == 0

    def check(self, context: RuleContext) -> List[Violation]:
        repo_root = context.repo_root
        surface_files = self._surface_files(context)

        if context.scope == "staged":
            allowed = set(surface_files)
            targets = self._collect_changed_python(repo_root, allowed)
        else:
            targets = surface_files

        if not targets:
            return []

        if self._run_black_check(repo_root, targets):
            return []

        return [
            Violation(
                rule_name=self.name,
                message=(
                    "Run Black before committing: DriftGuard detected Python "
                    "files that would be reformatted."
                ),
                path=targets[0],
            )
        ]
