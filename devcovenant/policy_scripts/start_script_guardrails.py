"""Ensure launchers retain mandated guardrails (sudo prompts + notices)."""

from typing import List

from devcovenant.base import CheckContext, PolicyCheck, Violation

REQUIRED_PATTERNS = {
    "start.sh": ["sudo -k", "pkg_notice"],
    "start.command": ["sudo -k", "pkg_notice"],
    "start.bat": ["PKG_NOTICE"],
}


class StartScriptGuardrailsCheck(PolicyCheck):
    """Validate that the start launchers keep their security guards."""

    policy_id = "start-script-guardrails"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Report missing guardrail snippets inside start scripts."""
        violations: List[Violation] = []

        for name, patterns in REQUIRED_PATTERNS.items():
            target = context.repo_root / name
            if not target.exists():
                continue
            content = target.read_text(encoding="utf-8")
            for pattern in patterns:
                if pattern not in content:
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=target,
                            message=(
                                f"Launcher `{name}` is missing `{pattern}`. "
                                "Preserve the guardrails (sudo notice and "
                                "package-manager warnings) before committing."
                            ),
                        )
                    )
        return violations
