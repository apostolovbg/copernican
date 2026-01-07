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
        scripts = self._load_scripts_option()

        for script_entry in scripts:
            name = script_entry.get("path")
            patterns = script_entry.get(
                "required", REQUIRED_PATTERNS.get(name, [])
            )
            if not name or not patterns:
                continue
            target = context.repo_root / name
            if not target.exists():
                continue
            content = target.read_text(encoding="utf-8")
            missing_patterns = [
                pattern for pattern in patterns if pattern not in content
            ]
            if missing_patterns:
                violations.append(
                    Violation(
                        policy_id=self.policy_id,
                        severity="error",
                        file_path=target,
                        message=(
                            f"Launcher `{name}` is missing "
                            f"{', '.join(missing_patterns)}. Preserve the "
                            "guardrails (sudo notice and package-manager "
                            "warnings) before committing."
                        ),
                        can_auto_fix=True,
                        context={"missing_patterns": missing_patterns},
                    )
                )
        return violations

    def _load_scripts_option(self) -> List[dict]:
        """Parse metadata/config for script guard requirements."""
        default = [
            {"path": name, "required": patterns}
            for name, patterns in REQUIRED_PATTERNS.items()
        ]
        option = self.get_option("scripts", default)
        if option is None:
            return default
        if isinstance(option, list):
            return option
        if isinstance(option, str):
            entries: List[dict] = []
            for raw_entry in option.split(";"):
                entry = raw_entry.strip()
                if not entry:
                    continue
                if ":" not in entry:
                    continue
                path, patterns_raw = entry.split(":", 1)
                patterns = [
                    token.strip()
                    for token in patterns_raw.split("|")
                    if token.strip()
                ]
                entries.append({"path": path.strip(), "required": patterns})
            return entries or default
        return default
