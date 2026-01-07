"""Guard editing access to paths selected by the metadata-defined globs."""

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import SelectorSet


class ReadOnlyDirectoriesCheck(PolicyCheck):
    """Block changes inside read-only directories declared via metadata."""

    policy_id = "read-only-directories"
    version = "1.0.0"

    def _selector(self) -> SelectorSet:
        """Return the selector describing protected globs."""
        return SelectorSet.from_policy(self)

    def check(self, context: CheckContext):
        """Ensure modified files stay outside read-only globs unless waived."""
        files = context.changed_files or []
        if not files:
            return []

        selector = self._selector()
        violations = []

        for path in files:
            if not selector.matches(path, context.repo_root):
                continue

            violations.append(
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    file_path=path,
                    message=(
                        "Read-only directories configured via this policy "
                        "were modified without prior approval."
                    ),
                )
            )

        return violations
