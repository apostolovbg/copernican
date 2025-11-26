# Last Updated: 2025-11-26
"""Logging coverage for the verbose DriftGuard scaffolding."""

import logging

import driftguard
from driftguard.spec import DriftConfig, DriftGuardSpec, load_spec


def test_load_engine_emits_configuration_logs(monkeypatch, tmp_path, caplog):
    """Loading the engine should log the target root and spec details."""

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

    caplog.set_level(logging.INFO, logger="driftguard")

    driftguard.load_engine(repo_root=tmp_path)

    messages = [record.message for record in caplog.records]
    assert any(
        "Loading DriftGuard engine for repo root" in msg for msg in messages
    )
    assert any("Loaded DriftGuard spec for project" in msg for msg in messages)


def test_load_spec_logs_fallback(tmp_path, caplog):
    """Missing specs should trigger a fallback warning and parse log."""

    caplog.set_level(logging.INFO, logger="driftguard")

    load_spec(repo_root=tmp_path)

    messages = [record.message for record in caplog.records]
    assert any("repo_policy.yml missing" in msg for msg in messages)
    assert any("Parsed DriftGuard spec version" in msg for msg in messages)
