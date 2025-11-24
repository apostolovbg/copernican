# Last Updated: 2025-11-24
"""Regression tests for module-level placeholder setup in :mod:`copernican`."""

import importlib
import sys
from pathlib import Path

import pytest


def test_optional_modules_are_defined_before_dependency_load(monkeypatch):
    """Ensure dependency failures cannot trigger NameError in cleanup logic."""

    repo_root = Path(__file__).resolve().parent.parent
    expected_venv = repo_root / ".venv"
    monkeypatch.setenv("VIRTUAL_ENV", str(expected_venv))

    sys.modules.pop("copernican", None)

    copernican = importlib.import_module("copernican")

    assert hasattr(copernican, "plt")
    assert copernican.plt is None

