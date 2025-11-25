# Last Updated: 2025-11-25
"""Convenience helpers for constructing the DriftGuard policy engine.

This module exposes :func:`load_engine`, a small wrapper that resolves the
repository root, loads the DriftGuard specification from ``driftguard.yml`` and
returns a ready-to-run :class:`~driftguard.core.PolicyEngine` instance. The
function keeps imports lightweight and intentionally avoids pulling in any
Copernican modules so the package can be spun off cleanly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from driftguard.core import PolicyEngine
from driftguard.spec import DriftGuardSpec, load_spec
from driftguard.utils import resolve_repo_root


def load_engine(repo_root: Optional[Path | str] = None) -> PolicyEngine:
    """Load the DriftGuard policy engine for the given repository root.

    Parameters
    ----------
    repo_root:
        The repository root containing ``driftguard.yml``. When ``None`` the
        parent directory of this module is used as the fallback so the helper
        works when imported from the checked-out repository without additional
        configuration.

    Returns
    -------
    PolicyEngine
        An engine configured with the loaded specification and ready to run
        ``check`` or ``fix`` actions.
    """

    resolved_root = resolve_repo_root(repo_root)
    spec: DriftGuardSpec = load_spec(resolved_root)
    return PolicyEngine(spec=spec, repo_root=resolved_root)
