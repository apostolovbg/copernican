# Last Updated: 2025-11-25
"""Smoke tests for the DriftGuard engine scaffolding."""

from pathlib import Path

from driftguard import load_engine
from driftguard.core import PolicyEngine
from driftguard.spec import DriftConfig, DriftGuardSpec


def test_load_engine_returns_policy_engine(
    monkeypatch, tmp_path: Path
) -> None:
    """Ensure ``load_engine`` wires the spec into a ``PolicyEngine``."""

    stub_spec = DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={},
        surfaces={},
        drift=DriftConfig(),
    )
    monkeypatch.setattr(
        "driftguard.load_spec", lambda repo_root=None: stub_spec
    )

    engine = load_engine(repo_root=tmp_path)

    assert isinstance(engine, PolicyEngine)
    assert engine.spec is stub_spec
    assert engine.repo_root == tmp_path

    result = engine.check()
    assert result == {"violations": [], "metrics": []}
