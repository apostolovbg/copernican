"""DevCovenant policy: Ensure new Python modules ship with tests.

This policy ensures that new Python modules under copernican_lib/ and engines/
are accompanied by new or updated tests under tests/, preventing untested
code from entering the repository.
"""

import subprocess
from pathlib import Path

from devcovenant.base import PolicyScript, Violation


class NewModulesNeedTestsPolicy(PolicyScript):
    """Ensure new Python modules ship with accompanying tests."""

    def _collect_repo_changes(self) -> tuple[set[Path], set[Path]]:
        """Return added and modified files reported by Git."""
        try:
            output = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=self.repo_root,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return set(), set()

        added: set[Path] = set()
        modified: set[Path] = set()

        for line in output.splitlines():
            if not line or len(line) < 4:
                continue
            status, path_str = line[:2], line[3:]
            path = self.repo_root / path_str
            index_state, worktree_state = status[0], status[1]

            if index_state in {"A", "C", "R"} or worktree_state in {"A", "?"}:
                added.add(path)
            elif index_state == "?":
                added.add(path)
            elif index_state in {"M", "R", "C"} or worktree_state == "M":
                modified.add(path)

        return added, modified

    def check(self, file_paths: list[Path]) -> list[Violation]:
        """Check that new Python modules have corresponding tests."""
        violations = []

        added, modified = self._collect_repo_changes()

        # Find new Python modules outside tests/
        new_modules = []
        for path in added:
            if path.suffix != ".py" or not path.is_file():
                continue
            try:
                rel = path.relative_to(self.repo_root)
            except ValueError:
                continue

            if rel.parts and rel.parts[0] == "tests":
                continue

            # Check if it's in copernican_lib/ or engines/
            if rel.parts and rel.parts[0] in ("copernican_lib", "engines"):
                new_modules.append(path)

        if not new_modules:
            return violations

        # Check if any test files were changed
        changed_tests = []
        for path in modified | added:
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(self.repo_root)
            except ValueError:
                continue

            if rel.parts and rel.parts[0] == "tests":
                changed_tests.append(path)

        if not changed_tests:
            targets = ", ".join(
                sorted(
                    path.relative_to(self.repo_root).as_posix()
                    for path in new_modules
                )
            )
            violations.append(
                Violation(
                    policy_id="new-modules-need-tests",
                    path=new_modules[0],
                    message=(
                        f"Add or update tests under tests/ for new modules: "
                        f"{targets}"
                    ),
                )
            )

        return violations


def run(repo_root: Path, file_paths: list[Path]) -> list[Violation]:
    """Entry point for DevCovenant engine."""
    policy = NewModulesNeedTestsPolicy(repo_root)
    return policy.check(file_paths)
