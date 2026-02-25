from __future__ import annotations

import json
import time
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.devflow_run_gates import DevflowRunGates


def make_ctx(
    tmp_path: Path,
    changed: list[str],
    config: dict | None = None,
) -> CheckContext:
    files = [tmp_path / path for path in changed]
    for f in files:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("# code", encoding="utf-8")
    return CheckContext(
        repo_root=tmp_path,
        changed_files=files,
        all_files=files,
        mode="pre-commit",
        config=config or {},
    )


def test_requires_tests_for_code_change(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, ["src/example.py"])
    check = DevflowRunGates()
    violations = check.check(ctx)
    assert violations, "missing test_status should trigger a violation"


def test_passes_when_tests_are_fresh(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, ["src/example.py"])
    status_path = tmp_path / "devcovenant" / "test_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time() + 10
    status = {
        "last_run_utc": "2025-12-27T00:00:00Z",
        "last_run_epoch": now,
        "commands": ["pytest", "python -m unittest discover"],
    }
    status_path.write_text(json.dumps(status), encoding="utf-8")

    check = DevflowRunGates()
    violations = check.check(ctx)
    assert not violations


def test_non_code_changes_do_not_require_tests(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, ["docs/readme.md"])
    check = DevflowRunGates()
    violations = check.check(ctx)
    assert not violations


def test_custom_status_path(tmp_path: Path) -> None:
    ctx = make_ctx(
        tmp_path,
        ["src/example.py"],
        config={
            "policies": {
                "devflow-run-gates": {
                    "test_status_file": "alt/status.json",
                    "required_commands": ["pytest"],
                    "code_extensions": [".py"],
                }
            }
        },
    )
    status_path = tmp_path / "alt" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_utc": "2025-12-27T00:00:00Z",
        "last_run_epoch": time.time() + 10,
        "commands": ["pytest"],
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    check = DevflowRunGates()
    check.set_options({}, ctx.get_policy_config("devflow-run-gates"))
    violations = check.check(ctx)
    assert not violations, "Custom path should be respected"


def test_metadata_config_overrides(tmp_path: Path) -> None:
    """Policy-def metadata should configure file paths and commands."""
    ctx = make_ctx(tmp_path, ["src/example.py"])
    status_path = tmp_path / "alt" / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "last_run_utc": "2025-12-27T00:00:00Z",
        "last_run_epoch": time.time() + 10,
        "commands": ["pytest"],
    }
    status_path.write_text(json.dumps(payload), encoding="utf-8")

    check = DevflowRunGates()
    check.set_options(
        {
            "test_status_file": "alt/status.json",
            "required_commands": ["pytest"],
            "code_extensions": [".py"],
        },
        {},
    )
    violations = check.check(ctx)
    assert (
        violations == []
    ), "Metadata-provided options should satisfy path/command checks"
