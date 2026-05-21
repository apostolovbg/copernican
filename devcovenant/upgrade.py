#!/usr/bin/env python3
"""Upgrade DevCovenant core in the current repository."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import importlib.metadata as importlib_metadata
import shutil
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from packaging.version import InvalidVersion, Version

import devcovenant.core.cli_support as cli_args_module

_UPGRADE_RUNTIME_PRESERVE_DIRS = (
    Path("devcovenant/registry/runtime"),
    Path("devcovenant/logs"),
)
_UPGRADE_RUNTIME_PRESERVE_FILES = (
    Path("devcovenant/config.yaml"),
    Path("devcovenant/registry/registry.yaml"),
)


def _read_version(path: Path) -> str:
    """Read version text from a file, falling back to 0.0.0."""
    if not path.exists():
        return "0.0.0"
    version_text = path.read_text(encoding="utf-8").strip()
    return version_text or "0.0.0"


def _normalize_version_for_compare(raw: str) -> str:
    """Normalize DevCovenant package version text into canonical PEP 440."""
    token = str(raw or "").strip()
    if not token:
        return "0"
    try:
        return str(Version(token))
    except InvalidVersion as exc:
        raise ValueError(
            "Invalid DevCovenant package version string "
            f"`{raw}` for upgrade compare."
        ) from exc


def _parse_version_for_compare(raw: str) -> Version:
    """Parse one DevCovenant package version into ordering-safe form."""
    return Version(_normalize_version_for_compare(raw))


def _preserve_upgrade_runtime_state(repo_root: Path, temp_root: Path) -> None:
    """Copy runtime-local state that should survive core replacement."""
    for rel_path in _UPGRADE_RUNTIME_PRESERVE_DIRS:
        source_path = repo_root / rel_path
        if not source_path.exists():
            continue
        if not source_path.is_dir():
            continue
        preserved_path = temp_root / rel_path
        preserved_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_path, preserved_path)
    for rel_path in _UPGRADE_RUNTIME_PRESERVE_FILES:
        source_path = repo_root / rel_path
        if not source_path.exists():
            continue
        if not source_path.is_file():
            continue
        preserved_path = temp_root / rel_path
        preserved_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, preserved_path)


def _restore_upgrade_runtime_state(repo_root: Path, temp_root: Path) -> None:
    """Restore runtime-local state after core replacement during upgrade."""
    local_registry_rel = Path("devcovenant/registry/runtime")
    logs_rel = Path("devcovenant/logs")
    preserved_local = temp_root / local_registry_rel
    preserved_logs = temp_root / logs_rel

    if preserved_local.exists():
        target_local = repo_root / local_registry_rel
        if target_local.exists():
            shutil.rmtree(target_local)
        target_local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(preserved_local, target_local)

    if preserved_logs.exists():
        target_logs = repo_root / logs_rel
        target_logs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(preserved_logs, target_logs, dirs_exist_ok=True)

    for rel_path in _UPGRADE_RUNTIME_PRESERVE_FILES:
        preserved_file = temp_root / rel_path
        if not preserved_file.exists():
            continue
        target_path = repo_root / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(preserved_file, target_path)


def _ensure_upgrade_config(repo_root: Path) -> None:
    """Seed review-required config when missing after core refresh."""
    from devcovenant.core.execution import print_step

    config_path = repo_root / "devcovenant" / "config.yaml"
    if config_path.exists():
        return
    template_path = (
        repo_root
        / "devcovenant"
        / "builtin"
        / "profiles"
        / "global"
        / "assets"
        / "config.yaml"
    )
    if not template_path.exists():
        return
    config_path.write_text(
        template_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print_step(
        "Config missing after upgrade; seeded review-required config baseline",
        "ℹ️",
    )


def _resolve_upgrade_source_dir(
    repo_root: Path,
    distribution_lookup: Callable[
        [str], importlib_metadata.Distribution
    ] = importlib_metadata.distribution,
) -> Path:
    """Resolve source package path, guarding against local import shadowing."""
    from devcovenant import install
    from devcovenant.core.execution import print_step

    source_dir = Path(install.__file__).resolve().parent
    target_dir = (repo_root / "devcovenant").resolve()
    if source_dir != target_dir:
        return source_dir
    try:
        distribution = distribution_lookup("devcovenant")
    except importlib_metadata.PackageNotFoundError:
        return source_dir
    dist_source_dir = Path(distribution.locate_file("devcovenant")).resolve()
    if dist_source_dir.exists() and dist_source_dir != target_dir:
        print_step(
            (
                "Detected local package shadow; "
                "using installed package source for upgrade"
            ),
            "ℹ️",
        )
        return dist_source_dir
    return source_dir


def _replace_core_package_for_upgrade(repo_root: Path) -> None:
    """Replace core package while preserving upgrade-owned runtime state."""
    from devcovenant import install

    source_dir = _resolve_upgrade_source_dir(repo_root)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_root = Path(temp_dir)
        _preserve_upgrade_runtime_state(repo_root, temp_root)
        install.replace_core_package(repo_root, source_dir=source_dir)
        _restore_upgrade_runtime_state(repo_root, temp_root)


def upgrade_repo(repo_root: Path) -> int:
    """Upgrade DevCovenant core and run full refresh."""
    from devcovenant.core.execution import (
        merge_active_run_phase_timings,
        print_step,
    )
    from devcovenant.core.refresh_runtime import refresh_repo

    phase_timings: list[dict[str, object]] = []
    source_version_path = Path(__file__).resolve().parent / "VERSION"
    target_version_path = repo_root / "devcovenant" / "VERSION"

    version_started = time.perf_counter()
    source_version = _read_version(source_version_path)
    target_version = _read_version(target_version_path)
    print_step(
        (
            "Version compare: source="
            f"{source_version}, installed={target_version}"
        ),
        "ℹ️",
    )
    try:
        source_key = _parse_version_for_compare(source_version)
        target_key = _parse_version_for_compare(target_version)
    except ValueError as error:
        raise SystemExit(f"Upgrade blocked: {error}") from error
    phase_timings.append(
        {
            "phase": "version_compare",
            "duration_seconds": round(
                time.perf_counter() - version_started, 6
            ),
            "changed": source_key != target_key,
        }
    )

    replace_started = time.perf_counter()
    _replace_core_package_for_upgrade(repo_root)
    phase_timings.append(
        {
            "phase": "replace_core_package",
            "duration_seconds": round(
                time.perf_counter() - replace_started, 6
            ),
            "changed": True,
        }
    )

    target_version_path.write_text(f"{source_version}\n", encoding="utf-8")
    if source_key > target_key:
        print_step("Core package replaced with newer version", "✅")
    elif source_key == target_key:
        print_step("Core package refreshed from source (same version)", "✅")
    else:
        print_step(
            (
                "Core package reconciled from source "
                "(installed version was newer)"
            ),
            "✅",
        )

    config_started = time.perf_counter()
    _ensure_upgrade_config(repo_root)
    phase_timings.append(
        {
            "phase": "ensure_config",
            "duration_seconds": round(time.perf_counter() - config_started, 6),
            "changed": True,
        }
    )
    print_step("Running full refresh after upgrade", "🔄")
    refresh_started = time.perf_counter()
    result = refresh_repo(repo_root)
    phase_timings.append(
        {
            "phase": "refresh",
            "duration_seconds": round(
                time.perf_counter() - refresh_started, 6
            ),
            "changed": result == 0,
        }
    )
    if result != 0:
        print_step(
            "Upgrade refresh failed; inspect run logs for details.",
            "🚫",
        )
    merge_active_run_phase_timings("upgrade", phase_timings)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for upgrade command."""
    return cli_args_module.build_command_parser(
        "upgrade",
        "Upgrade DevCovenant core in the current repository.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute upgrade command."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )

    del args
    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: upgrade", "🧭")
    print_banner("Upgrade", "⬆️")

    return upgrade_repo(repo_root)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
