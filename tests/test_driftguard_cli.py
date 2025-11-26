# Last Updated: 2025-11-26
"""CLI wiring tests for the DriftGuard stub."""

from pathlib import Path

from driftguard.cli import main
from driftguard.rules import Violation


class _DummyEngine:
    repo_root = Path(".")

    def __init__(self, violations=None):
        self._violations = violations or []

    def check(self, scope: str, mode: str):
        _ = (scope, mode)
        return {"violations": self._violations, "metrics": []}

    def fix(self, scope: str, mode: str):
        _ = (scope, mode)
        return {"violations": self._violations, "metrics": []}

    def metrics(self, scope: str, mode: str):
        _ = (scope, mode)
        return []


def test_cli_accepts_check_command(monkeypatch) -> None:
    """``driftguard check`` should exit cleanly when no violations appear."""

    monkeypatch.setattr(
        "driftguard.cli.load_engine", lambda repo_root=None: _DummyEngine([])
    )
    exit_code = main(["check", "--scope", "repo", "--mode", "full"])
    assert exit_code == 0


def test_cli_exits_nonzero_on_violations(monkeypatch) -> None:
    """Any violation should force a non-zero exit status."""

    violation = Violation(rule_name="test", message="fail", path=Path("README.md"))
    monkeypatch.setattr(
        "driftguard.cli.load_engine", lambda repo_root=None: _DummyEngine([violation])
    )
    exit_code = main(["check", "--scope", "repo", "--mode", "full"])
    assert exit_code == 1


def test_cli_accepts_metrics_command(monkeypatch) -> None:
    """``driftguard metrics`` should also parse successfully."""

    monkeypatch.setattr(
        "driftguard.cli.load_engine", lambda repo_root=None: _DummyEngine([])
    )
    exit_code = main(["metrics", "--scope", "repo", "--mode", "fast"])
    assert exit_code == 0


def test_cli_accepts_repo_root_after_command(monkeypatch, tmp_path) -> None:
    """The repo-root flag should parse when placed after the sub-command."""

    monkeypatch.setattr(
        "driftguard.cli.load_engine", lambda repo_root=None: _DummyEngine([])
    )
    exit_code = main(
        [
            "check",
            "--scope",
            "repo",
            "--mode",
            "full",
            "--repo-root",
            str(tmp_path),
        ]
    )
    assert exit_code == 0
