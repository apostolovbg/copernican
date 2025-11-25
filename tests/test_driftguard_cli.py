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
