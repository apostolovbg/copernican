"""
DevCovenant - Self-enforcing policy system.

This system parses policy definitions from AGENTS.md, maintains policy
scripts, and enforces policies automatically during development.
"""

from __future__ import annotations

import atexit
import shutil
import sys
from pathlib import Path


def _source_checkout_root(
    package_file: str | Path | None = None,
) -> Path | None:
    """Return the repository root when this package is imported from source."""
    module_path = Path(package_file or __file__).resolve()
    package_dir = module_path.parent
    if package_dir.name != "devcovenant":
        return None
    repo_root = package_dir.parent
    if not (repo_root / ".git").exists():
        return None
    if not (package_dir / "__main__.py").exists():
        return None
    if not (package_dir / "cli.py").exists():
        return None
    return repo_root


def _source_checkout_cache_roots(
    package_file: str | Path | None = None,
) -> tuple[Path, ...]:
    """Return owned source trees that must stay free of Python caches."""
    repo_root = _source_checkout_root(package_file)
    if repo_root is None:
        return ()
    return (
        repo_root / "devcovenant",
        repo_root / "tests" / "devcovenant",
    )


def _disable_source_checkout_bytecode(
    package_file: str | Path | None = None,
) -> bool:
    """Disable Python cache-file writes when imported from source."""
    if not _source_checkout_cache_roots(package_file):
        return False
    sys.dont_write_bytecode = True
    return True


def _cleanup_source_checkout_import_cache(
    package_file: str | Path | None = None,
) -> bool:
    """Remove owned source-tree cache files that should not linger in repo."""
    removed = False
    for cache_root in _source_checkout_cache_roots(package_file):
        if not cache_root.exists():
            continue
        for cache_dir in cache_root.rglob("__pycache__"):
            shutil.rmtree(cache_dir, ignore_errors=True)
            removed = True
        for compiled_file in cache_root.rglob("*.py[co]"):
            try:
                compiled_file.unlink()
            except OSError:
                continue
            removed = True
    return removed


def _register_source_checkout_import_cleanup(
    package_file: str | Path | None = None,
) -> bool:
    """Register exit-time cleanup for source-package import cache."""
    if not _source_checkout_cache_roots(package_file):
        return False
    atexit.register(_cleanup_source_checkout_import_cache, package_file)
    return True


_SOURCE_CHECKOUT_BYTECODE_DISABLED = _disable_source_checkout_bytecode()
_SOURCE_CHECKOUT_IMPORT_CACHE_CLEANED = _cleanup_source_checkout_import_cache()
_SOURCE_CHECKOUT_IMPORT_CACHE_CLEANUP_REGISTERED = (
    _register_source_checkout_import_cleanup()
)


def _read_package_version() -> str:
    """Read the packaged DevCovenant version from the bundled VERSION file."""
    version_path = Path(__file__).with_name("VERSION")
    try:
        version_text = version_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"
    return version_text or "0.0.0"


__version__ = _read_package_version()
__all__ = ["__version__"]
