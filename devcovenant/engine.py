"""
Main DevCovenant engine - orchestrates policy checking and enforcement.
"""

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml

from .base import CheckContext, PolicyCheck, Violation
from .parser import PolicyDefinition, PolicyParser
from .registry import PolicyRegistry, PolicySyncIssue


class DevCovenantEngine:
    """
    Main engine for devcovenant policy enforcement.
    """

    # Directories we never traverse for policy checks
    _IGNORED_DIRS = frozenset(
        {
            ".git",
            ".venv",
            ".python",
            "output",
            "logs",
            "build",
            "dist",
            "node_modules",
            "copernican_suite.egg-info",
            "__pycache__",
            ".cache",
            ".venv.lock",
        }
    )

    def __init__(self, repo_root: Optional[Path] = None):
        """
        Initialize the engine.

        Args:
            repo_root: Root directory of the repository (default: current dir)
        """
        if repo_root is None:
            repo_root = Path.cwd()

        self.repo_root = Path(repo_root).resolve()
        self.devcovenant_dir = self.repo_root / "devcovenant"
        self.agents_md_path = self.repo_root / "AGENTS.md"
        self.config_path = self.devcovenant_dir / "config.yaml"
        self.registry_path = self.devcovenant_dir / "registry.json"

        # Load configuration
        self.config = self._load_config()

        # Initialize parser and registry
        self.parser = PolicyParser(self.agents_md_path)
        self.registry = PolicyRegistry(self.registry_path, self.repo_root)

        # Statistics
        self.passed_count = 0
        self.failed_count = 0

    def _load_config(self) -> Dict:
        """Load configuration from config.yaml."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def check(self, mode: str = "normal") -> "CheckResult":
        """
        Main entry point for policy checking.

        Args:
            mode: Check mode (startup, lint, pre-commit, normal)

        Returns:
            CheckResult object
        """
        # Parse policies from AGENTS.md
        policies = self.parser.parse_agents_md()

        # Check for policy sync issues (hash mismatches)
        sync_issues = self.registry.check_policy_sync(policies)

        if sync_issues:
            self.report_sync_issues(sync_issues)
            if mode == "startup":
                return CheckResult(
                    [], should_block=True, sync_issues=sync_issues
                )

        # Load and run policy checks
        violations = self.run_policy_checks(policies, mode)

        # Report violations
        self.report_violations(violations, mode)

        # Determine if should block
        should_block = self.should_block(violations)

        return CheckResult(violations, should_block, sync_issues=[])

    def report_sync_issues(self, issues: List[PolicySyncIssue]):
        """
        Report policy sync issues in AI-friendly format.

        Args:
            issues: List of PolicySyncIssue objects
        """
        print("\n" + "=" * 70)
        print("🔄 POLICY SYNC REQUIRED")
        print("=" * 70)
        print()

        for issue in issues:
            print(f"Policy '{issue.policy_id}' requires attention.")
            print(f"Issue: {issue.issue_type.replace('_', ' ').title()}")
            print()

            print("📋 Current Policy (from AGENTS.md):")
            print("━" * 70)
            # Print first 500 chars of policy text
            policy_preview = issue.policy_text[:500]
            if len(issue.policy_text) > 500:
                policy_preview += "..."
            print(policy_preview)
            print("━" * 70)
            print()

            print("🎯 Action Required:")
            is_new = (
                issue.issue_type == "script_missing"
                or issue.issue_type == "new_policy"
            )
            if is_new:
                print(f"1. Create: {issue.script_path}")
                print("2. Implement the policy described above")
                print(
                    "3. Use the PolicyCheck base class from "
                    "devcovenant.base"
                )
                test_file = (
                    f"devcovenant/tests/test_policies/"
                    f"test_{issue.policy_id}.py"
                )
                print(f"4. Add tests in {test_file}")
                print(f"5. Run tests: pytest {test_file} -v")
            else:
                print(f"1. Update: {issue.script_path}")
                print("2. Modify the script to implement the updated policy")
                test_file = (
                    f"devcovenant/tests/test_policies/"
                    f"test_{issue.policy_id}.py"
                )
                print(f"3. Update tests in {test_file}")
                print(f"4. Run tests: pytest {test_file} -v")

            print(
                "6. Re-run devcovenant to update hash and "
                "clear 'updated' flag"
            )
            print()
            print("⚠️  Complete this BEFORE working on user's request.")
            print()
            print("=" * 70)
            print()

    def run_policy_checks(
        self, policies: List[PolicyDefinition], mode: str
    ) -> List[Violation]:
        """
        Load and run all policy check scripts.

        Args:
            policies: List of policy definitions
            mode: Check mode

        Returns:
            List of all violations found
        """
        violations = []

        # Build check context
        context = self._build_check_context(mode)

        for policy in policies:
            # Skip inactive policies
            if policy.status not in ["active", "new"]:
                continue

            # Try to load and run the policy script
            try:
                checker = self._load_policy_script(policy.policy_id)
                if checker:
                    policy_violations = checker.check(context)
                    violations.extend(policy_violations)
                    if not policy_violations:
                        self.passed_count += 1
                    else:
                        self.failed_count += 1
            except Exception as e:
                # If script fails, report but don't block
                print(
                    f"⚠️  Warning: Policy '{policy.policy_id}' "
                    f"check failed: {e}"
                )

        return violations

    def _build_check_context(self, mode: str) -> CheckContext:
        """
        Build the CheckContext for policy checks.

        Args:
            mode: Check mode

        Returns:
            CheckContext object
        """
        changed_files = []
        all_files = []

        if mode == "pre-commit":
            # Only check changed files (git diff)
            import subprocess

            try:
                result = subprocess.run(
                    ["git", "diff", "--cached", "--name-only"],
                    cwd=self.repo_root,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                changed_files = [
                    self.repo_root / f
                    for f in result.stdout.strip().split("\n")
                    if f
                ]
            except Exception:
                pass
        else:
            # Check all Python files
            suffixes = {".py", ".md", ".yml", ".yaml"}
            all_files = self._collect_all_files(suffixes)

        return CheckContext(
            repo_root=self.repo_root,
            changed_files=changed_files,
            all_files=all_files,
            mode=mode,
        )

    def _collect_all_files(self, suffixes: Set[str]) -> List[Path]:
        """
        Walk the repository tree and collect files matching the given suffixes,
        skipping large or third-party directories.
        """
        matched: List[Path] = []

        for root, dirs, files in os.walk(self.repo_root):
            # Filter out ignored directories
            dirs[:] = [
                d for d in dirs if self._should_descend_dir(Path(root) / d)
            ]

            for name in files:
                file_path = Path(root) / name
                if file_path.suffix.lower() in suffixes:
                    matched.append(file_path)

        return matched

    def _should_descend_dir(self, candidate: Path) -> bool:
        """
        Decide whether to continue walking into a directory.
        """
        name = candidate.name

        if name in self._IGNORED_DIRS:
            return False

        # Always skip __pycache__ variants
        if name.startswith("__pycache__"):
            return False

        return True

    def _load_policy_script(self, policy_id: str) -> Optional[PolicyCheck]:
        """
        Dynamically load a policy script.

        Args:
            policy_id: ID of the policy

        Returns:
            PolicyCheck instance or None if not found
        """
        # Convert hyphens to underscores for Python module names
        script_name = policy_id.replace("-", "_")
        script_path = (
            self.devcovenant_dir / "policy_scripts" / f"{script_name}.py"
        )

        if not script_path.exists():
            return None

        # Load the module
        spec = importlib.util.spec_from_file_location(
            f"devcovenant.policy_scripts.{policy_id}", script_path
        )
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            # Find the PolicyCheck subclass
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, PolicyCheck)
                    and attr is not PolicyCheck
                ):
                    return attr()

        return None

    def report_violations(self, violations: List[Violation], mode: str):
        """
        Report violations in AI-friendly, actionable format.

        Args:
            violations: List of violations
            mode: Check mode
        """
        if not violations:
            print("\n✅ All policy checks passed!")
            return

        print("\n" + "=" * 70)
        print("📊 DEVCOVENANT CHECK RESULTS")
        print("=" * 70)
        print()
        print(f"✅ Passed: {self.passed_count} policies")
        print(f"⚠️  Violations: {len(violations)} issues found")
        print()

        # Group by severity
        by_severity = {}
        for v in violations:
            if v.severity not in by_severity:
                by_severity[v.severity] = []
            by_severity[v.severity].append(v)

        # Report in order: critical, error, warning, info
        for severity in ["critical", "error", "warning", "info"]:
            if severity not in by_severity:
                continue

            for violation in by_severity[severity]:
                self._report_single_violation(violation)

        # Summary
        print("=" * 70)
        self._report_summary(by_severity)

    def _report_single_violation(self, violation: Violation):
        """Report a single violation with full context."""
        # Icon based on severity
        icons = {
            "critical": "❌",
            "error": "🚫",
            "warning": "⚠️",
            "info": "💡",
        }
        icon = icons.get(violation.severity, "•")

        print(f"{icon} {violation.severity.upper()}: {violation.policy_id}")

        if violation.file_path:
            location = str(violation.file_path)
            if violation.line_number:
                location += f":{violation.line_number}"
            print(f"📍 {location}")

        print()
        print(f"Issue: {violation.message}")

        if violation.suggestion:
            print()
            print("Fix:")
            print(violation.suggestion)

        if violation.can_auto_fix:
            print()
            print("Auto-fix: Available (run with --fix)")

        print()
        print(f"Policy: AGENTS.md#{violation.policy_id}")
        print("━" * 70)
        print()

    def _report_summary(self, by_severity: Dict[str, List[Violation]]):
        """Report summary of violations."""
        critical = len(by_severity.get("critical", []))
        errors = len(by_severity.get("error", []))
        warnings = len(by_severity.get("warning", []))
        info = len(by_severity.get("info", []))

        print(
            f"Summary: {critical} critical, {errors} errors, "
            f"{warnings} warnings, {info} info"
        )
        print()

        # Determine status
        if critical > 0:
            print("Status: 🚫 BLOCKED (critical violations must be fixed)")
        elif errors > 0:
            fail_threshold = self.config.get("engine", {}).get(
                "fail_threshold", "error"
            )
            if fail_threshold in ["error", "warning", "info"]:
                print("Status: 🚫 BLOCKED (violations >= error threshold)")
        else:
            print("Status: ✅ PASSED")

        print()
        if self.config.get("engine", {}).get("auto_fix_enabled", True):
            print(
                "💡 Quick fix: Run 'devcovenant check --fix' to "
                "auto-fix fixable violations"
            )

        print("=" * 70)

    def should_block(self, violations: List[Violation]) -> bool:
        """
        Determine if violations should block the commit/operation.

        Args:
            violations: List of violations

        Returns:
            True if should block
        """
        if not violations:
            return False

        fail_threshold = self.config.get("engine", {}).get(
            "fail_threshold", "error"
        )

        # Map severity to numeric level
        severity_levels = {
            "critical": 4,
            "error": 3,
            "warning": 2,
            "info": 1,
        }

        threshold_level = severity_levels.get(fail_threshold, 3)

        # Check if any violation meets or exceeds threshold
        for violation in violations:
            violation_level = severity_levels.get(violation.severity, 1)
            if violation_level >= threshold_level:
                return True

        return False


class CheckResult:
    """Result of a devcovenant check operation."""

    def __init__(
        self,
        violations: List[Violation],
        should_block: bool,
        sync_issues: List[PolicySyncIssue],
    ):
        self.violations = violations
        self.should_block = should_block
        self.sync_issues = sync_issues

    def has_sync_issues(self) -> bool:
        """Check if there are policy sync issues."""
        return len(self.sync_issues) > 0

    def has_violations(self) -> bool:
        """Check if there are any violations."""
        return len(self.violations) > 0
