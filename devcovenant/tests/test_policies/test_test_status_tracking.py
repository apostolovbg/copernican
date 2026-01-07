"""Tests for the test status tracking policy."""

import json
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.test_status_tracking import (
    STATUS_RELATIVE,
    TestStatusTrackingCheck,
)


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _policy() -> TestStatusTrackingCheck:
    policy = TestStatusTrackingCheck()
    policy.set_options(
        {
            "watch_dirs": [
                "project_lib",
                "engines",
                "tests",
                "tools",
                "scripts",
            ],
            "watch_files": ["project.py", "pyproject.toml"],
        },
        {},
    )
    return policy


def test_flags_missing_status_update(tmp_path: Path):
    code_path = _write(
        tmp_path / "project_lib" / "module.py",
        "def demo():\n    return 1\n",
    )
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[code_path, tmp_path / "scripts" / "run_tests.sh"],
    )
    violations = _policy().check(context)
    assert violations
    assert "test status" in violations[0].message.lower()


def test_accepts_recent_status(tmp_path: Path):
    code_path = _write(
        tmp_path / "project_lib" / "module.py",
        "def demo():\n    return 1\n",
    )
    status_path = tmp_path / STATUS_RELATIVE
    payload = {
        "last_run": "2025-12-24T12:00:00+00:00",
        "command": "pytest && python -m unittest discover",
        "sha": "a" * 40,
        "notes": "",
    }
    _write(status_path, json.dumps(payload))
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[code_path, status_path],
    )
    assert _policy().check(context) == []


def test_rejects_invalid_payload(tmp_path: Path):
    code_path = _write(
        tmp_path / "project_lib" / "module.py",
        "def demo():\n    return 1\n",
    )
    status_path = _write(
        tmp_path / STATUS_RELATIVE,
        '{"last_run": "", "sha": ""}',
    )
    context = CheckContext(
        repo_root=tmp_path,
        changed_files=[code_path, status_path],
    )
    violations = _policy().check(context)
    assert violations
    assert "invalid" in violations[0].message.lower()
