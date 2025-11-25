"""Lightweight DriftGuard package bootstrap.

The package is intentionally self contained so the policy engine can be
split into a standalone project later without inheriting Copernican-specific
imports or side effects.
"""

from pathlib import Path
from typing import Optional

from driftguard.core import PolicyEngine
from driftguard.spec import DriftGuardSpec, load_spec

__all__ = ["DriftGuardSpec", "PolicyEngine", "load_engine"]


def load_engine(repo_root: Optional[Path | str] = None) -> PolicyEngine:
    """Create a :class:`PolicyEngine` configured from ``driftguard.yml``.

    The helper keeps orchestration intentionally thin so the engine is easy to
    reuse outside this repository. It defers all parsing and rule selection
    to :func:`driftguard.spec.load_spec` and the
    :class:`~driftguard.core.PolicyEngine` constructor.

    Args:
        repo_root: Repository root to scan. Defaults to the current working
            directory when omitted.

    Returns:
        ``PolicyEngine`` configured with the loaded policy specification.
    """

    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    spec = load_spec(repo_root=root_path)
    return PolicyEngine(spec=spec, repo_root=root_path)
