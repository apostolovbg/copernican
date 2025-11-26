"""Validation tests for the DriftGuard specification loader and helpers."""

from pathlib import Path

import pytest

from driftguard.spec import (
    DriftConfig,
    DriftGuardSpec,
    MetricThreshold,
    SpecValidationError,
    SurfaceSpec,
    load_spec,
)
from driftguard.utils import resolve_surface_globs


def _policy_path(root: Path) -> Path:
    policy_dir = root / "driftguard"
    policy_dir.mkdir(parents=True, exist_ok=True)
    return policy_dir / "repo_policy.yml"


def test_load_spec_parses_full_schema(tmp_path: Path) -> None:
    """The loader should preserve every section of the spec file."""

    spec_text = """
# Last Updated: 2025-11-25
version: 1
project: Copernican Suite
rulesets:
  core-metadata: hard
  docs-drift: warn
  python-quality: hard
  bugfix-coverage: warn
surfaces:
  docs:
    include:
      - README.md
    exclude: []
    rules:
      - last-updated-header
  python-lib:
    include:
      - copernican_lib/**/*.py
    exclude: []
    rules:
      - no-print
      - new-modules-need-tests
drift:
  metrics:
    - name: todo-count
      description: Count of TODO markers.
      max_warning: 50
    - name: test-coupling-ratio
      min_warning: 0.8
"""
    spec_path = _policy_path(tmp_path)
    spec_path.write_text(spec_text)

    spec = load_spec(tmp_path)

    assert spec.version == 1
    assert spec.project == "Copernican Suite"
    assert spec.rulesets["core-metadata"] == "hard"
    assert set(spec.surfaces) == {"docs", "python-lib"}
    assert spec.surfaces["docs"].include == ["README.md"]
    assert spec.surfaces["python-lib"].rules == [
        "no-print",
        "new-modules-need-tests",
    ]
    todo_metric = spec.drift.metrics["todo-count"]
    assert todo_metric.max_warning == 50
    assert todo_metric.description.startswith("Count of TODO")


def test_load_spec_rejects_unknown_keys(tmp_path: Path) -> None:
    """Unknown keys should raise explicit validation errors."""

    spec_path = _policy_path(tmp_path)
    spec_path.write_text(
        """
# Last Updated: 2025-11-25
version: 1
project: Example
rulesets: {}
surfaces: {}
drift:
  metrics: []
extra: true
"""
    )

    with pytest.raises(SpecValidationError):
        load_spec(tmp_path)


def test_missing_surface_fields_raise(tmp_path: Path) -> None:
    """Surfaces missing required sections should fail early."""

    spec_path = _policy_path(tmp_path)
    spec_path.write_text(
        """
# Last Updated: 2025-11-25
version: 1
project: Example
rulesets: {}
surfaces:
  docs:
    exclude: []
    rules: []
drift:
  metrics: []
"""
    )

    with pytest.raises(SpecValidationError):
        load_spec(tmp_path)


def test_resolve_surface_globs_respects_excludes(tmp_path: Path) -> None:
    """Surface resolution should honour include and exclude patterns."""

    lib_dir = tmp_path / "copernican_lib"
    exp_dir = lib_dir / "experimental"
    engine_dir = tmp_path / "engines"
    for path in (lib_dir, exp_dir, engine_dir):
        path.mkdir(parents=True, exist_ok=True)
    (lib_dir / "main.py").write_text("print('ok')\n")
    (exp_dir / "sketch.py").write_text("print('skip')\n")
    (engine_dir / "kernel.py").write_text("print('ok')\n")

    spec = DriftGuardSpec(
        version=1,
        project="Example",
        rulesets={"python-quality": "hard"},
        surfaces={
            "python-lib": SurfaceSpec(
                name="python-lib",
                include=["copernican_lib/**/*.py", "engines/**/*.py"],
                exclude=["copernican_lib/experimental/**/*.py"],
                rules=["no-print"],
            )
        },
        drift=DriftConfig(
            metrics={"todo-count": MetricThreshold(name="todo-count")}
        ),
    )

    resolved = resolve_surface_globs(spec, tmp_path, "python-lib")
    resolved_rel = {path.relative_to(tmp_path).as_posix() for path in resolved}

    assert "copernican_lib/main.py" in resolved_rel
    assert "engines/kernel.py" in resolved_rel
    assert "copernican_lib/experimental/sketch.py" not in resolved_rel
