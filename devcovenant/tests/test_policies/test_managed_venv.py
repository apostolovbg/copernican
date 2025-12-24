"""Tests for the managed virtual environment policy."""

import sys
from pathlib import Path

from devcovenant.base import CheckContext
from devcovenant.policy_scripts.managed_venv import ManagedVenvCheck


def test_detects_external_interpreter(tmp_path: Path, monkeypatch):
    (tmp_path / ".venv").mkdir()
    fake_python = tmp_path / "external" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("", encoding="utf-8")
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_python.parent))
    monkeypatch.setattr(sys, "executable", str(fake_python))

    checker = ManagedVenvCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[])
    violations = checker.check(context)
    assert violations
    assert "virtual environment" in violations[0].message.lower()


def test_allows_managed_venv(tmp_path: Path, monkeypatch):
    managed = tmp_path / ".venv"
    managed.mkdir()
    venv_python = managed / "bin"
    venv_python.mkdir()
    venv_executable = venv_python / "python"
    venv_executable.write_text("", encoding="utf-8")
    monkeypatch.setenv("VIRTUAL_ENV", str(managed))
    monkeypatch.setattr(sys, "executable", str(venv_executable))

    checker = ManagedVenvCheck()
    context = CheckContext(repo_root=tmp_path, changed_files=[])
    assert checker.check(context) == []
