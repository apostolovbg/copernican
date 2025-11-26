# Last Updated: 2025-11-26
"""Lightweight DriftGuard package bootstrap.

The package is intentionally self contained so the policy engine can be
split into a standalone project later without inheriting Copernican-specific
imports or side effects.
"""

from pathlib import Path
from typing import Optional

from driftguard.core import PolicyEngine
from driftguard.logging_utils import ensure_logger, get_logger
from driftguard.spec import DriftGuardSpec, load_spec

__all__ = ["DriftGuardSpec", "PolicyEngine", "load_engine"]


logger = get_logger()


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

    ensure_logger()
    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    logger.info("Loading DriftGuard engine for repo root %s", root_path)
    spec = load_spec(repo_root=root_path)
    logger.info(
        "Loaded DriftGuard spec for project %s (version %s)",
        spec.project,
        spec.version,
    )
    return PolicyEngine(spec=spec, repo_root=root_path)
