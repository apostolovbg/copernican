# Last Updated: 2025-11-25
"""Unit tests for DriftGuard drift metric rules."""

from __future__ import annotations

import datetime
from pathlib import Path

from driftguard.core import PolicyEngine
from driftguard.rules import RuleContext
from driftguard.rules.drift import DocAgeRule, TestCouplingRule, TodoCountRule
from driftguard.spec import (
    DriftConfig,
    DriftGuardSpec,
    MetricThreshold,
    SurfaceSpec,
)


def _spec_with_metrics(metrics: dict[str, MetricThreshold]) -> DriftGuardSpec:
    return DriftGuardSpec(
        version=1,
        project="Tests",
        rulesets={},
        surfaces={
            "python-lib": SurfaceSpec(
                name="python-lib",
                include=["copernican_lib/**/*.py", "engines/**/*.py"],
                exclude=[
                    "tests/**/*.py",
                    "tools/**/*.py",
                    "driftguard/**/*.py",
                ],
                rules=[],
            ),
            "docs": SurfaceSpec(
                name="docs",
                include=[
                    "README.md",
                    "AGENTS.md",
                    "CONTRIBUTING.md",
                    "docs/**/*.md",
                ],
                exclude=[],
                rules=[],
            ),
        },
        drift=DriftConfig(metrics=metrics),
    )


def _context(
    tmp_path: Path, metrics: dict[str, MetricThreshold]
) -> RuleContext:
    return RuleContext(
        repo_root=tmp_path,
        spec=_spec_with_metrics(metrics),
        scope="repo",
        mode="full",
    )


def test_todo_count_metrics_and_thresholds(tmp_path: Path) -> None:
    """TodoCountRule should count markers and respect thresholds."""

    lib_dir = tmp_path / "copernican_lib"
    lib_dir.mkdir()
    (lib_dir / "alpha.py").write_text("# TODO: refine logic\n# FIXME later\n")
    (tmp_path / "engines").mkdir()
    (tmp_path / "engines" / "engine.py").write_text("# XXX legacy compat\n")

    metrics = {"todo-count": MetricThreshold(name="todo-count", max_warning=2)}
    context = _context(tmp_path, metrics)

    rule = TodoCountRule()
    metric = rule.metrics(context)[0]

    assert metric.name == "todo-count"
    assert metric.value == 3
    assert metric.threshold == 2

    violations = rule.check(context)
    assert violations  # threshold breached


def test_test_coupling_ratio(tmp_path: Path) -> None:
    """TestCouplingRule should compute a coverage-style ratio."""

    lib_dir = tmp_path / "copernican_lib"
    engine_dir = tmp_path / "engines"
    test_dir = tmp_path / "tests"
    lib_dir.mkdir()
    engine_dir.mkdir()
    test_dir.mkdir()
    (lib_dir / "alpha.py").write_text("print('alpha')\n")
    (engine_dir / "beta.py").write_text("print('beta')\n")
    (test_dir / "test_alpha.py").write_text("def test_alpha():\n    pass\n")

    metrics = {
        "test-coupling-ratio": MetricThreshold(
            name="test-coupling-ratio", min_warning=0.75
        )
    }
    context = _context(tmp_path, metrics)

    rule = TestCouplingRule()
    metric = rule.metrics(context)[0]

    assert metric.value == 0.5
    assert metric.threshold == 0.75
    assert rule.check(context)  # ratio falls below threshold


def test_doc_age_metrics(tmp_path: Path) -> None:
    """DocAgeRule should report the oldest Last Updated header."""

    today = datetime.date.today()
    older = today - datetime.timedelta(days=10)
    newer = today - datetime.timedelta(days=2)
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        f"**Last Updated:** {older.isoformat()}\nReadme body.\n",
        encoding="utf-8",
    )
    (tmp_path / "docs" / "guide.md").write_text(
        f"**Last Updated:** {newer.isoformat()}\nGuide body.\n",
        encoding="utf-8",
    )

    metrics = {
        "doc-age-days": MetricThreshold(name="doc-age-days", max_warning=5)
    }
    context = _context(tmp_path, metrics)

    rule = DocAgeRule()
    metric = rule.metrics(context)[0]

    assert metric.value == 10
    assert metric.threshold == 5
    assert metric.path == tmp_path / "README.md"
    assert rule.check(context)


def test_policy_engine_collects_metrics(tmp_path: Path) -> None:
    """PolicyEngine should return drift metrics alongside violations."""

    lib_dir = tmp_path / "copernican_lib"
    lib_dir.mkdir()
    (lib_dir / "alpha.py").write_text("# TODO: implement\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_alpha.py").write_text("pass\n")
    (tmp_path / "README.md").write_text(
        "**Last Updated:** 2025-01-01\nReadme.\n", encoding="utf-8"
    )

    metrics = {
        "todo-count": MetricThreshold(name="todo-count", max_warning=5),
        "test-coupling-ratio": MetricThreshold(
            name="test-coupling-ratio", min_warning=0.0
        ),
        "doc-age-days": MetricThreshold(name="doc-age-days", max_warning=9999),
    }
    spec = _spec_with_metrics(metrics)
    engine = PolicyEngine(spec=spec, repo_root=tmp_path)

    result = engine.check()

    metric_names = {metric.name for metric in result["metrics"]}
    assert metric_names == set(metrics)
