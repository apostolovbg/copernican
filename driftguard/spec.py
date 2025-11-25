# Last Updated: 2025-11-25
"""Specification loader for DriftGuard policy definitions.

The policy format is intentionally lightweight and YAML-based so it can be
updated alongside repository metadata without recompiling any code. Only the
filesystem and the spec contents are consulted; no Copernican-specific modules
are imported here so the implementation can be lifted into a standalone
package later.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Optional

import yaml


@dataclass(frozen=True)
class RuleSurface:
    """Describe a set of files or directories targeted by rules."""

    name: str
    include: List[str]
    exclude: List[str] = field(default_factory=list)
    rules: Optional[List[str]] = None


@dataclass(frozen=True)
class DriftGuardSpec:
    """Container for the policy specification."""

    version: int
    project: str
    rulesets: Mapping[str, str]
    surfaces: Mapping[str, RuleSurface]


def _load_yaml(path: Path) -> MutableMapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_surfaces(
    raw_surfaces: Mapping[str, object]
) -> Dict[str, RuleSurface]:
    parsed: Dict[str, RuleSurface] = {}
    for name, raw_surface in raw_surfaces.items():
        surface_data = raw_surface or {}
        include = list(surface_data.get("include", []))
        exclude = list(surface_data.get("exclude", []))
        rules = surface_data.get("rules")
        parsed[name] = RuleSurface(
            name=name,
            include=include,
            exclude=exclude,
            rules=list(rules) if rules is not None else None,
        )
    return parsed


def load_spec(path: Path) -> DriftGuardSpec:
    """Load and parse the DriftGuard YAML specification.

    Parameters
    ----------
    path:
        Path to the ``driftguard.yml`` file.

    Returns
    -------
    DriftGuardSpec
        Parsed specification instance ready for engine construction.
    """

    data = _load_yaml(path)
    surfaces_raw = data.get("surfaces", {})
    return DriftGuardSpec(
        version=int(data.get("version", 1)),
        project=str(data.get("project", "")),
        rulesets=data.get("rulesets", {}),
        surfaces=_parse_surfaces(surfaces_raw),
    )
