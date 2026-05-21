#!/usr/bin/env python3
"""Install DevCovenant into the current repository."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import devcovenant.core.cli_support as cli_args_module
import devcovenant.core.managed_docs as managed_docs_service
import devcovenant.core.refresh_runtime as refresh_runtime_service


def _source_package_dir() -> Path:
    """Return the packaged devcovenant source directory."""
    return Path(__file__).resolve().parent


def _target_package_dir(repo_root: Path) -> Path:
    """Return the destination devcovenant directory for a repo."""
    return repo_root / "devcovenant"


_CUSTOM_SCAFFOLD_FILES = {"README.md", "__init__.py"}


def _detect_importable_managed_docs(
    repo_root: Path,
    source_dir: Path,
) -> list[str]:
    """Return existing repo docs eligible for first managed-doc adoption."""
    return managed_docs_service.detect_importable_managed_docs(
        repo_root,
        source_dir,
    )


def _copy_ignore_builder(source_dir: Path):
    """Return copy ignore callback scoped to one source directory."""

    def _copy_ignore(directory: str, names: list[str]) -> set[str]:
        """Ignore runtime state, tracked outputs, and package-owned payload."""
        ignored = set()
        current = Path(directory)
        try:
            rel_path = current.relative_to(source_dir).as_posix()
        except ValueError:
            rel_path = current.name

        if rel_path == "registry":
            if "runtime" in names:
                ignored.add("runtime")
            if "registry.yaml" in names:
                ignored.add("registry.yaml")
        if rel_path == "logs":
            for name in names:
                if name == "README.md":
                    continue
                ignored.add(name)

        if rel_path in {"custom/policies", "custom/profiles"}:
            for name in names:
                if name in _CUSTOM_SCAFFOLD_FILES:
                    continue
                ignored.add(name)

        for name in names:
            if name == "__pycache__":
                ignored.add(name)
            if name.endswith(".pyc"):
                ignored.add(name)
        return ignored

    return _copy_ignore


def _collect_custom_payload_dirs(custom_dir: Path) -> list[tuple[Path, Path]]:
    """Collect user custom payload dirs to preserve across replacement."""
    collected: list[tuple[Path, Path]] = []
    sections = (
        "policies",
        "profiles",
    )
    for section in sections:
        section_root = custom_dir / section
        if not section_root.exists():
            continue
        for entry in sorted(section_root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith(".") or entry.name == "__pycache__":
                continue
            rel = Path("custom") / section / entry.name
            collected.append((entry, rel))
    return collected


def replace_core_package(
    repo_root: Path,
    source_dir: Path | None = None,
) -> None:
    """Replace the repository-root devcovenant package with packaged source."""
    source_dir = (source_dir or _source_package_dir()).resolve()
    target_dir = _target_package_dir(repo_root).resolve()
    if source_dir == target_dir:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        preserved_payload_root = temp_path / "custom_payload"
        custom_dir = target_dir / "custom"

        if custom_dir.exists():
            for payload_dir, rel_path in _collect_custom_payload_dirs(
                custom_dir
            ):
                destination = preserved_payload_root / rel_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(payload_dir, destination, dirs_exist_ok=True)

        if target_dir.exists():
            shutil.rmtree(target_dir)

        shutil.copytree(
            source_dir,
            target_dir,
            ignore=_copy_ignore_builder(source_dir),
        )

        if preserved_payload_root.exists():
            for preserved_dir in sorted(preserved_payload_root.rglob("*")):
                if not preserved_dir.is_dir():
                    continue
                rel_path = preserved_dir.relative_to(preserved_payload_root)
                if len(rel_path.parts) != 3:
                    continue
                destination = target_dir / rel_path
                if destination.exists():
                    shutil.rmtree(destination)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(preserved_dir, destination)


def _ensure_review_required_config(
    repo_root: Path,
    *,
    import_managed_docs: list[str] | None = None,
) -> None:
    """Write/install a review-required config stub for post-install editing."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    config_path.write_text(
        refresh_runtime_service.render_review_required_config_yaml(
            repo_root,
            import_managed_docs=list(import_managed_docs or []),
        ),
        encoding="utf-8",
    )


def install_repo(repo_root: Path) -> int:
    """Install DevCovenant core and review-required config in a repository."""
    import devcovenant.core.repository_validation as manifest_module
    from devcovenant.core.execution import merge_active_run_phase_timings

    phase_timings: list[dict[str, object]] = []
    source_dir = _source_package_dir()

    detect_started = time.perf_counter()
    import_managed_docs = _detect_importable_managed_docs(
        repo_root,
        source_dir,
    )
    phase_timings.append(
        {
            "phase": "detect_importable_docs",
            "duration_seconds": round(time.perf_counter() - detect_started, 6),
            "changed": bool(import_managed_docs),
        }
    )

    replace_started = time.perf_counter()
    replace_core_package(repo_root, source_dir=source_dir)
    phase_timings.append(
        {
            "phase": "replace_core_package",
            "duration_seconds": round(
                time.perf_counter() - replace_started, 6
            ),
            "changed": True,
        }
    )

    cleanup_started = time.perf_counter()
    runtime_registry = repo_root / "devcovenant" / "registry" / "runtime"
    runtime_registry_removed = runtime_registry.exists()
    if runtime_registry_removed:
        shutil.rmtree(runtime_registry)
    phase_timings.append(
        {
            "phase": "runtime_registry_cleanup",
            "duration_seconds": round(
                time.perf_counter() - cleanup_started, 6
            ),
            "changed": runtime_registry_removed,
            "skipped": not runtime_registry_removed,
        }
    )

    config_started = time.perf_counter()
    _ensure_review_required_config(
        repo_root,
        import_managed_docs=import_managed_docs,
    )
    phase_timings.append(
        {
            "phase": "seed_review_required_config",
            "duration_seconds": round(time.perf_counter() - config_started, 6),
            "changed": True,
        }
    )

    manifest_started = time.perf_counter()
    manifest_module.ensure_manifest(repo_root)
    phase_timings.append(
        {
            "phase": "manifest_inventory",
            "duration_seconds": round(
                time.perf_counter() - manifest_started, 6
            ),
            "changed": True,
        }
    )
    merge_active_run_phase_timings("install", phase_timings)
    return 0


def _is_existing_install(repo_root: Path) -> bool:
    """Return True when DevCovenant is already present in repo_root."""
    target_dir = _target_package_dir(repo_root)
    if not target_dir.exists():
        return False
    return (target_dir / "__init__.py").exists() or (
        target_dir / "VERSION"
    ).exists()


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for install command."""
    return cli_args_module.build_command_parser(
        "install",
        "Install DevCovenant into the current repository.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute install command."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )

    del args
    repo_root = resolve_repo_root(require_install=False)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: install", "🧭")
    print_banner("Install", "📦")

    if _is_existing_install(repo_root):
        print_step("DevCovenant is already present in this repository.", "ℹ️")
        print_step("Run `devcovenant upgrade` to replace core files.", "ℹ️")
        return 1

    result = install_repo(repo_root)
    if result != 0:
        return result

    print_step("Installed devcovenant/ core package", "✅")
    print_step(
        (
            "Config reset to review-required baseline. Review "
            "devcovenant/config.yaml, set "
            "`install.config_reviewed: true`, then run "
            "`devcovenant deploy`."
        ),
        "ℹ️",
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
