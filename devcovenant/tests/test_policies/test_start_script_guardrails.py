"""Tests for the start script guardrail policy."""

from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.fixers.start_script_guardrails import (
    StartScriptGuardrailsFixer,
)
from devcovenant.policy_scripts.start_script_guardrails import (
    StartScriptGuardrailsCheck,
)


def _write_script(tmp_path: Path, name: str, content: str) -> Path:
    target = tmp_path / name
    target.write_text(content, encoding="utf-8")
    return target


def test_detects_missing_guardrails(tmp_path: Path):
    """Lack of the required `sudo -k` snippet flags a violation."""
    script = _write_script(tmp_path, "start.sh", "pkg_notice()\n")
    checker = StartScriptGuardrailsCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[script])

    violations = checker.check(context)
    assert violations
    assert any("sudo -k" in v.message for v in violations)


def test_allows_scripts_with_guardrails(tmp_path: Path):
    """Scripts that include every guard snippet pass."""
    start_sh = _write_script(
        tmp_path, "start.sh", "pkg_notice()\nsudo -k -p 'pwd' ...\n"
    )
    start_command = _write_script(
        tmp_path, "start.command", "pkg_notice()\nsudo -k -p 'pwd' ...\n"
    )
    start_bat = _write_script(tmp_path, "start.bat", "set PKG_NOTICE=ok\n")
    checker = StartScriptGuardrailsCheck()
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[start_sh, start_command, start_bat],
    )

    assert checker.check(context) == []


def test_auto_fix_injects_guardrails(tmp_path: Path):
    script = _write_script(tmp_path, "start.command", "#!/bin/bash\n")
    context = CheckContext(repo_root=tmp_path, changed_files=[script])
    checker = StartScriptGuardrailsCheck()
    violations = checker.check(context)
    assert violations
    fixer = StartScriptGuardrailsFixer()
    fixer.repo_root = tmp_path
    result = fixer.fix(violations[0])
    assert result.success
    content = script.read_text()
    assert "pkg_notice()" in content
    assert "sudo -k" in content
