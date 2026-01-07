"""Ensure modules under `module_roots` keep tests rooted beneath
`tests_root`."""

import subprocess
from pathlib import Path
from typing import List, Set

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet, build_watchlists


class NewModulesNeedTestsCheck(PolicyCheck):
    """Ensure new Python modules ship with accompanying tests."""

    policy_id = "new-modules-need-tests"
    version = "1.0.0"

    def _collect_repo_changes(
        self, repo_root: Path
    ) -> tuple[Set[Path], Set[Path], Set[Path]]:
        """Return added and modified files reported by Git."""
        try:
            output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return set(), set(), set()

        added: Set[Path] = set()
        modified: Set[Path] = set()
        deleted: Set[Path] = set()

        for line in output.splitlines():
            if not line or len(line) < 4:
                continue
            status, path_str = line[:2], line[3:]
            path = repo_root / path_str
            index_state, worktree_state = status[0], status[1]

            if index_state == "D" or worktree_state == "D":
                deleted.add(path)
                continue
            if index_state in {"A", "C", "R"} or worktree_state in {
                "A",
                "?",
            }:
                added.add(path)
            elif index_state == "?":
                added.add(path)
            elif index_state in {"M", "R", "C"} or worktree_state == "M":
                modified.add(path)

        return added, modified, deleted

    def check(self, context: CheckContext) -> List[Violation]:
        """Check that new Python modules have corresponding tests."""
        violations = []

        (
            added,
            modified,
            deleted,
        ) = self._collect_repo_changes(context.repo_root)
        module_selector = SelectorSet.from_policy(
            self, defaults={"include_suffixes": [".py"]}
        )
        _, watch_dirs = build_watchlists(
            self, defaults={"watch_dirs": ["tests"]}
        )
        tests_dirs = watch_dirs or ["tests"]
        tests_label = (
            ", ".join(sorted(tests_dirs))
            if len(tests_dirs) > 1
            else tests_dirs[0]
        )

        def _is_library_or_engine_module(path: Path) -> bool:
            """Return True when motion paths point at core Python modules."""
            if path.suffix != ".py":
                return False
            return module_selector.matches(path, context.repo_root)

        def _collect_changed_tests(paths: Set[Path]) -> List[Path]:
            """Collect touched files that live under tests/."""
            tests = []
            for path in paths:
                try:
                    rel = path.relative_to(context.repo_root).as_posix()
                except ValueError:
                    continue
                if any(
                    rel == test_dir or rel.startswith(f"{test_dir}/")
                    for test_dir in tests_dirs
                ):
                    tests.append(path)
            return tests

        # Find new Python modules outside tests/
        new_modules = []
        for path in added:
            if _is_library_or_engine_module(path):
                new_modules.append(path)

        removed_modules = []
        for path in deleted:
            if _is_library_or_engine_module(path):
                removed_modules.append(path)

        tests_changed = _collect_changed_tests(added | modified | deleted)

        if new_modules and not tests_changed:
            targets = ", ".join(
                sorted(
                    path.relative_to(context.repo_root).as_posix()
                    for path in new_modules
                )
            )
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=new_modules[0],
                    message=(
                        f"Add or update tests under {tests_label}/ "
                        f"for new modules: {targets}"
                    ),
                )
            )

        if removed_modules and not tests_changed:
            targets = ", ".join(
                sorted(
                    path.relative_to(context.repo_root).as_posix()
                    for path in removed_modules
                )
            )
            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=removed_modules[0],
                    message=(
                        f"Adjust tests under {tests_label}/ "
                        f"when removing modules: {targets}"
                    ),
                )
            )

        return violations
