#!/usr/bin/env python3
"""Deploy DevCovenant managed artifacts for the current repository."""

from __future__ import annotations

if __package__ in {None, ""}:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import shutil
import time
from pathlib import Path

import yaml

import devcovenant.core.cli_support as cli_args_module
import devcovenant.core.repository_paths as yaml_cache_service

NORMAL_REPO_PRUNE_PATHS = (Path("tests/devcovenant/core"),)


def _read_yaml(path: Path) -> dict[str, object]:
    """Load YAML mapping payload from disk."""
    if not path.exists():
        raise SystemExit(
            f"Deploy blocked: missing required config file: {path}."
        )
    try:
        payload = yaml_cache_service.load_yaml(path)
    except yaml.YAMLError as exc:
        raise SystemExit(
            f"Deploy blocked: invalid YAML in {path}: {exc}"
        ) from exc
    except OSError as exc:
        raise SystemExit(
            f"Deploy blocked: unable to read {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise SystemExit(
            f"Deploy blocked: {path} must contain a YAML mapping."
        )
    return payload


def _is_config_reviewed(config: dict[str, object]) -> bool:
    """Return True when install.config_reviewed is complete."""
    install_block = config.get("install")
    if not isinstance(install_block, dict):
        raise SystemExit(
            "Deploy blocked: `install` must be present as a mapping in "
            "devcovenant/config.yaml."
        )
    config_reviewed = install_block.get("config_reviewed")
    if not isinstance(config_reviewed, bool):
        raise SystemExit(
            "Deploy blocked: `install.config_reviewed` must be boolean."
        )
    return config_reviewed


def _is_developer_mode(config: dict[str, object]) -> bool:
    """Return True when the repository is used to develop DevCovenant
    itself."""
    developer_mode = config.get("developer_mode")
    if not isinstance(developer_mode, bool):
        raise SystemExit("Deploy blocked: `developer_mode` must be boolean.")
    return developer_mode


def _remove_path(target: Path) -> bool:
    """Delete a file or directory if it exists."""
    if not target.exists():
        return False
    if target.is_file() or target.is_symlink():
        target.unlink()
        return True
    shutil.rmtree(target)
    return True


def _prune_repo_only_developer_paths(repo_root: Path) -> list[str]:
    """Delete DevCovenant-only development paths from normal repos."""
    removed: list[str] = []
    for relative_path in NORMAL_REPO_PRUNE_PATHS:
        target = repo_root / relative_path
        if _remove_path(target):
            removed.append(str(relative_path))
    return removed


def deploy_repo(repo_root: Path) -> int:
    """Deploy managed DevCovenant docs/assets to a repo."""
    from devcovenant.core.execution import (
        merge_active_run_phase_timings,
        print_step,
    )
    from devcovenant.core.refresh_runtime import refresh_repo

    phase_timings: list[dict[str, object]] = []
    config_started = time.perf_counter()
    config_path = repo_root / "devcovenant" / "config.yaml"
    config = _read_yaml(config_path)
    if not _is_config_reviewed(config):
        raise SystemExit(
            "Deploy blocked: config review is not complete. Set "
            "`install.config_reviewed: true` first."
        )
    phase_timings.append(
        {
            "phase": "config_validation",
            "duration_seconds": round(time.perf_counter() - config_started, 6),
            "changed": False,
        }
    )

    prune_started = time.perf_counter()
    removed: list[str] = []
    if not _is_developer_mode(config):
        removed = _prune_repo_only_developer_paths(repo_root)
        if removed:
            print_step(
                "Deploy cleanup removed: " + ", ".join(removed),
                "🧹",
            )
    phase_timings.append(
        {
            "phase": "repo_only_prune",
            "duration_seconds": round(time.perf_counter() - prune_started, 6),
            "changed": bool(removed),
            "skipped": _is_developer_mode(config),
        }
    )

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
    merge_active_run_phase_timings("deploy", phase_timings)
    return result


def _build_parser() -> argparse.ArgumentParser:
    """Build parser for deploy command."""
    return cli_args_module.build_command_parser(
        "deploy",
        "Deploy managed docs/assets in the current repository.",
    )


def run(args: argparse.Namespace) -> int:
    """Execute deploy command."""
    from devcovenant.core.execution import (
        devcovenant_banner_title,
        print_banner,
        print_step,
        resolve_repo_root,
    )

    del args
    repo_root = resolve_repo_root(require_install=True)

    print_banner(devcovenant_banner_title(), "🚀")
    print_step("Command: deploy", "🧭")
    print_banner("Deploy", "📤")

    return deploy_repo(repo_root)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    cli_args_module.apply_output_mode_override_from_namespace(args)
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
