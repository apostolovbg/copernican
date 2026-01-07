"""
Auto-fixer for start-script-guardrails.
"""

from __future__ import annotations

from pathlib import Path

from devcovenant.base import FixResult, PolicyFixer, Violation

SHELL_BLOCK = """pkg_notice() {
    echo 'A package manager may request your password.'
    echo 'The Copernican Suite never reads or stores it.'
}

sudo_pkg() {
    pkg_notice
    sudo -k -p '[sudo] password for package manager: ' "$@"
}
"""

BREW_BLOCK = """brew_pkg() {
    pkg_notice
    brew "$@"
}
"""

BATCH_BLOCK = (
    'set "PKG_NOTICE=Package managers may request your password. '
    'The Copernican"\n'
    'set "PKG_NOTICE=%PKG_NOTICE% Suite never reads or stores it."\n'
)


class StartScriptGuardrailsFixer(PolicyFixer):
    """Inject the canonical guardrail snippets into launcher scripts."""

    policy_id = "start-script-guardrails"

    def can_fix(self, violation: Violation) -> bool:
        """Only handle guardrail violations for known start scripts."""
        return (
            violation.policy_id == self.policy_id
            and violation.file_path is not None
        )

    def fix(self, violation: Violation) -> FixResult:
        """Append the missing guardrail snippets to the launcher."""
        target = violation.file_path
        if target is None:
            return FixResult(success=False, message="Missing launcher path")
        try:
            content = Path(target).read_text(encoding="utf-8")
        except OSError as exc:
            return FixResult(success=False, message=str(exc))

        new_content = self._ensure_guardrails(
            Path(target), content, violation.context
        )
        if new_content == content:
            return FixResult(
                success=False, message="Guardrail snippets already present"
            )

        Path(target).write_text(new_content, encoding="utf-8")
        return FixResult(
            success=True,
            message=f"Reinstated guardrails in {target}",
            files_modified=[Path(target)],
        )

    def _ensure_guardrails(
        self, target: Path, content: str, context: dict
    ) -> str:
        """Append guardrail snippets tailored for the script type."""
        if target.suffix in {".sh", ".command"}:
            block = SHELL_BLOCK + "\n" + BREW_BLOCK
            if "pkg_notice" in context.get("missing_patterns", []):
                # Guarantee both pkg_notice and sudo helpers exist.
                missing = True
            else:
                missing = any(
                    token in context.get("missing_patterns", [])
                    for token in ("sudo -k",)
                )
            if missing:
                snippet = block.rstrip() + "\n"
            else:
                snippet = ""
        elif target.suffix == ".bat":
            if "PKG_NOTICE" in context.get("missing_patterns", []):
                snippet = BATCH_BLOCK
            else:
                snippet = ""
        else:
            snippet = ""

        if not snippet:
            return content

        if not content.endswith("\n"):
            content += "\n"
        return content + "\n" + snippet
