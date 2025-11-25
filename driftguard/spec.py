"""DriftGuard policy specification data structures and loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class SurfaceConfig:
    """Map rules to the file globs that define a surface."""

    name: str
    globs: List[str]
    rules: List[str]
    description: Optional[str] = None


@dataclass
class DriftMetricConfig:
    """Configuration for a drift metric defined in the policy spec."""

    name: str
    threshold: Optional[float] = None
    description: Optional[str] = None


@dataclass
class DriftGuardSpec:
    """Aggregate policy configuration loaded from ``driftguard.yml``."""

    surfaces: List[SurfaceConfig] = field(default_factory=list)
    metrics: List[DriftMetricConfig] = field(default_factory=list)
    raw: Optional[Dict[str, Any]] = None


def _parse_surfaces(raw_spec: Dict[str, Any]) -> List[SurfaceConfig]:
    """Convert raw surface definitions into data classes.

    The parser performs minimal validation for now to keep the scaffolding
    lightweight. TODO: expand validation and error handling once the spec
    stabilises.
    """

    surfaces: List[SurfaceConfig] = []
    for surface in raw_spec.get("surfaces", []) or []:
        surfaces.append(
            SurfaceConfig(
                name=surface.get("name", ""),
                globs=list(surface.get("globs", []) or []),
                rules=list(surface.get("rules", []) or []),
                description=surface.get("description"),
            )
        )
    return surfaces


def _parse_metrics(raw_spec: Dict[str, Any]) -> List[DriftMetricConfig]:
    """Convert raw drift metric definitions into data classes."""

    metrics: List[DriftMetricConfig] = []
    for metric in raw_spec.get("metrics", []) or []:
        metrics.append(
            DriftMetricConfig(
                name=metric.get("name", ""),
                threshold=metric.get("threshold"),
                description=metric.get("description"),
            )
        )
    return metrics


def load_spec(repo_root: Optional[Path | str] = None) -> DriftGuardSpec:
    """Load ``driftguard.yml`` from ``repo_root`` and parse the policy spec.

    The loader defaults to returning an empty specification when the YAML file
    is missing so callers can still exercise the CLI and engine scaffolding.
    TODO: enforce presence and schema validation once the initial spec lands.
    """

    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    spec_path = root_path / "driftguard.yml"
    raw_spec: Dict[str, Any] = {}

    if spec_path.exists():
        raw_spec = yaml.safe_load(spec_path.read_text()) or {}

    surfaces = _parse_surfaces(raw_spec)
    metrics = _parse_metrics(raw_spec)
    return DriftGuardSpec(surfaces=surfaces, metrics=metrics, raw=raw_spec)
