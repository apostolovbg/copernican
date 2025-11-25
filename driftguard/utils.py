# Last Updated: 2025-11-25
"""Utility helpers shared by DriftGuard rules.

Functions remain intentionally small until rule implementations land. The
module keeps filesystem and parsing helpers close to the rules while avoiding
Copernican-specific imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Set

from driftguard.spec import DriftGuardSpec


def iter_globs(root: Path, patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns relative to ``root`` with deduplication."""

    results: List[Path] = []
    seen: Set[Path] = set()
    for pattern in patterns:
        for candidate in root.glob(pattern):
            if candidate not in seen:
                results.append(candidate)
                seen.add(candidate)
    return results


def resolve_surface_globs(
    spec: DriftGuardSpec, repo_root: Path, surface_name: str
) -> List[Path]:
    """Resolve a surface's include and exclude globs to concrete paths.

    Resolving through the spec keeps rule logic aligned with ``driftguard.yml``
    rather than ad-hoc path lists. Excludes are applied after all includes so
    the spec can intentionally shadow broader includes with narrower ignore
    patterns.
    """

    if surface_name not in spec.surfaces:
        raise KeyError(f"Unknown surface {surface_name!r} requested.")
    surface = spec.surfaces[surface_name]
    includes = iter_globs(repo_root, surface.include)
    excluded: Set[Path] = set()
    for pattern in surface.exclude:
        excluded.update(repo_root.glob(pattern))
    resolved = [path for path in includes if path not in excluded]
    resolved.sort()
    return resolved
