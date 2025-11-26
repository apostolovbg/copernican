# Last Updated: 2025-11-26
"""CLI wiring tests for the DriftGuard stub."""

from driftguard import load_engine
from driftguard.cli import main


def test_cli_accepts_check_command(monkeypatch) -> None:
    """``driftguard check`` should parse and return a zero exit status."""

    monkeypatch.setattr("driftguard.cli.load_engine", load_engine)
    exit_code = main(["check", "--scope", "repo", "--mode", "full"])
    assert exit_code == 0


def test_cli_accepts_metrics_command(monkeypatch) -> None:
    """``driftguard metrics`` should also parse successfully."""

    monkeypatch.setattr("driftguard.cli.load_engine", load_engine)
    exit_code = main(["metrics", "--scope", "repo", "--mode", "fast"])
    assert exit_code == 0


def test_cli_accepts_repo_root_after_command(monkeypatch, tmp_path) -> None:
    """The repo-root flag should parse when placed after the sub-command."""

    monkeypatch.setattr("driftguard.cli.load_engine", load_engine)
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
