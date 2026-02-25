"""Ensure every launcher listed in the metadata changes together."""

from typing import List, Set

from devcovenant.base import CheckContext, PolicyCheck, Violation
from devcovenant.selectors import build_watchlists

START_SCRIPTS = ("start.sh", "start.command", "start.bat")


class StartScriptParityCheck(PolicyCheck):
    """Enforce parity across the start.* launchers."""

    policy_id = "start-script-parity"
    version = "1.0.0"

    def check(self, context: CheckContext) -> List[Violation]:
        """Warn when only a subset of the launchers changes."""
        defaults = {"watch_files": START_SCRIPTS}
        watch_files, _ = build_watchlists(self, defaults=defaults)
        scripts = set(watch_files or START_SCRIPTS)
        changed_scripts: Set[str] = set()
        for path in context.changed_files or []:
            name = path.name
            if name in scripts:
                changed_scripts.add(name)

        if not changed_scripts:
            return []

        existing_scripts = {
            name for name in scripts if (context.repo_root / name).exists()
        }

        missing = sorted(existing_scripts - changed_scripts)
        if not missing:
            return []

        representative = context.repo_root / next(iter(changed_scripts))
        message = (
            "Launcher changes touched "
            f"{', '.join(sorted(changed_scripts))} but not "
            f"{', '.join(missing)}; update the remaining launchers so "
            "all start scripts stay in sync."
        )

        return [
            Violation(
                policy_id=self.policy_id,
                severity="error",
                file_path=representative,
                message=message,
                can_auto_fix=True,
                context={
                    "changed": sorted(changed_scripts),
                    "missing": missing,
                },
            )
        ]
