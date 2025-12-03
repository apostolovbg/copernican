"""DevCovenant policy: Ensure new or removed Python modules ship with tests.

This policy ensures that new Python modules under copernican_lib/ and
engines/ are accompanied by new or updated tests under tests/, preventing
untested code from entering the repository. Tests should evolve with the code,
and removing a module must adjust its associated tests rather than leaving
stale coverage behind.
"""

import subprocess
from pathlib import Path
from typing import List, Set

from devcovenant.base import CheckContext, PolicyCheck, Violation


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

        def _is_library_or_engine_module(relative: Path) -> bool:
            return (
                relative.suffix == ".py"
                and relative.parts
                and relative.parts[0] in ("copernican_lib", "engines")
            )

        def _collect_changed_tests(paths: Set[Path]) -> List[Path]:
            tests = []
            for path in paths:
                try:
                    rel = path.relative_to(context.repo_root)
                except ValueError:
                    continue
                if rel.parts and rel.parts[0] == "tests":
                    tests.append(path)
            return tests

        # Find new Python modules outside tests/
        new_modules = []
        for path in added:
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            if _is_library_or_engine_module(rel):
                new_modules.append(path)

        removed_modules = []
        for path in deleted:
            try:
                rel = path.relative_to(context.repo_root)
            except ValueError:
                continue

            if _is_library_or_engine_module(rel):
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
                        f"Add or update tests under tests/ "
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
                        f"Adjust tests under tests/ when removing modules: "
                        f"{targets}"
                    ),
                )
            )

        return violations
