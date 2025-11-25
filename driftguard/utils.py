# Last Updated: 2025-11-25
"""Utility helpers for DriftGuard modules.

Only filesystem and YAML inputs are used here so the package remains
self-contained and ready for a future spin-off.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def resolve_repo_root(repo_root: Optional[Path | str]) -> Path:
    """Resolve the repository root path.

    Parameters
    ----------
    repo_root:
        Optional repository root hint. When ``None`` the parent of the
        ``driftguard`` package directory is used.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """

    if repo_root is None:
        return Path(__file__).resolve().parent.parent
    if isinstance(repo_root, Path):
        return repo_root.resolve()
    return Path(repo_root).expanduser().resolve()


def ensure_mode(mode: str, allowed: Iterable[str]) -> str:
    """Validate a mode argument against an allowed set."""

    normalized = mode.lower()
    if normalized not in {item.lower() for item in allowed}:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(f"Unsupported mode '{mode}'. Allowed: {allowed_str}.")
    return normalized


def ensure_scope(scope: str, allowed: Iterable[str]) -> str:
    """Validate a scope argument against an allowed set."""

    normalized = scope.lower()
    if normalized not in {item.lower() for item in allowed}:
        allowed_str = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unsupported scope '{scope}'. Allowed: {allowed_str}."
        )
    return normalized
