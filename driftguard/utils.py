"""Utility helpers shared by DriftGuard rules.

Functions remain intentionally small until rule implementations land. The
module keeps filesystem and parsing helpers close to the rules while avoiding
Copernican-specific imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def iter_globs(root: Path, patterns: Iterable[str]) -> List[Path]:
    """Expand glob patterns relative to ``root``.

    TODO: add deduplication and ignore handling once surfaces stabilise.
    """

    results: List[Path] = []
    for pattern in patterns:
        results.extend(root.glob(pattern))
    return results
