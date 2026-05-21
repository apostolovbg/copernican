"""Cleanup target resolution and safe deletion helpers."""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml

import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.workflow_support as registry_runtime
from devcovenant.builtin.policies.managed_environment import (
    managed_environment_runtime,
)
from devcovenant.core.execution import (
    get_active_run_log_context,
    merge_active_run_log_metadata,
    print_step,
    runtime_print,
)
from devcovenant.core.repository_paths import display_path

_CLEAN_TARGET_KEYS = (
    "build_dirs",
    "build_globs",
    "cache_dirs",
    "cache_globs",
    "runtime_registry_dirs",
    "runtime_registry_globs",
    "logs_dirs",
    "logs_globs",
    "protected_dirs",
    "protected_globs",
)
_HARD_PROTECTED_DIRS = (".git",)
_HARD_PROTECTED_GLOBS = (
    "devcovenant/logs/README.md",
    "devcovenant/registry/README.md",
    "devcovenant/registry/registry.yaml",
)
_RELEASE_TREE_SUFFIX_PATTERN = re.compile(r"^(?:v?\d)[A-Za-z0-9._+-]*$")


@dataclass(frozen=True)
class CleanConfig:
    """Resolved cleanup configuration for one repository."""

    build_dirs: tuple[str, ...]
    build_globs: tuple[str, ...]
    cache_dirs: tuple[str, ...]
    cache_globs: tuple[str, ...]
    runtime_registry_dirs: tuple[str, ...]
    runtime_registry_globs: tuple[str, ...]
    logs_dirs: tuple[str, ...]
    logs_globs: tuple[str, ...]
    protected_dirs: tuple[str, ...]
    protected_globs: tuple[str, ...]


@dataclass(frozen=True)
class CleanSelection:
    """Selected cleanup categories for one command invocation."""

    include_build: bool
    include_cache: bool
    include_runtime_registry: bool
    include_logs: bool

    def labels(self) -> tuple[str, ...]:
        """Return human-readable labels for the selected cleanup set."""
        labels: list[str] = []
        if self.include_build:
            labels.append("build")
        if self.include_cache:
            labels.append("cache")
        if self.include_runtime_registry:
            labels.append("registry")
        if self.include_logs:
            labels.append("logs")
        return tuple(labels)


@dataclass(frozen=True)
class CleanResult:
    """Structured result for one cleanup execution."""

    selection: CleanSelection
    removed_paths: tuple[str, ...]
    skipped_protected_paths: tuple[str, ...]
    skipped_protected_match_count: int


def _dedupe(items: Iterable[str]) -> list[str]:
    """Return a deterministic de-duplicated list preserving order."""
    seen: set[str] = set()
    resolved: list[str] = []
    for raw_item in items:
        token = str(raw_item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        resolved.append(token)
    return resolved


def _read_config_payload(repo_root: Path) -> dict[str, object]:
    """Load repo config payload from `devcovenant/config.yaml`."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _normalize_string_list(raw_value: object) -> list[str]:
    """Normalize a config/profile list-like value into strings."""
    if isinstance(raw_value, str):
        items = [raw_value]
    elif isinstance(raw_value, list):
        items = raw_value
    else:
        return []
    return _dedupe(str(item or "").strip() for item in items)


def _normalize_clean_mapping(raw_value: object) -> dict[str, list[str]]:
    """Normalize one clean overlay/override mapping."""
    if not isinstance(raw_value, dict):
        return {}
    normalized: dict[str, list[str]] = {}
    for key in _CLEAN_TARGET_KEYS:
        if key not in raw_value:
            continue
        normalized[key] = _normalize_string_list(raw_value.get(key))
    return normalized


def _merge_clean_layers(
    base: dict[str, list[str]], incoming: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Return one clean mapping with additive de-duplicated merge semantics."""
    merged = {key: list(base.get(key, [])) for key in _CLEAN_TARGET_KEYS}
    for key in _CLEAN_TARGET_KEYS:
        merged[key] = _dedupe([*merged[key], *incoming.get(key, [])])
    return merged


def resolve_clean_selection(
    *,
    include_all: bool,
    include_build: bool,
    include_cache: bool,
    include_registry: bool,
    include_logs: bool,
) -> CleanSelection:
    """Resolve effective cleanup categories from CLI flags."""
    if include_all:
        return CleanSelection(
            include_build=True,
            include_cache=True,
            include_runtime_registry=True,
            include_logs=True,
        )
    if not any((include_build, include_cache, include_registry, include_logs)):
        raise ValueError(
            "Select at least one cleanup scope: --all, --build, --cache, "
            "--registry, or --logs."
        )
    return CleanSelection(
        include_build=include_build,
        include_cache=include_cache,
        include_runtime_registry=include_registry,
        include_logs=include_logs,
    )


def resolve_clean_config(repo_root: Path) -> CleanConfig:
    """Resolve effective cleanup configuration from profiles and config."""
    payload = _read_config_payload(repo_root)
    active_profiles = profile_registry_service.parse_active_profiles(
        payload,
        include_global=True,
    )
    profile_registry = profile_registry_service.load_profile_registry(
        repo_root
    )
    profile_clean = profile_registry_service.resolve_profile_clean_overlays(
        profile_registry,
        active_profiles,
    )

    clean_block = payload.get("clean") if isinstance(payload, dict) else {}
    if not isinstance(clean_block, dict):
        clean_block = {}
    config_overlays = _normalize_clean_mapping(clean_block.get("overlays"))
    config_overrides = _normalize_clean_mapping(clean_block.get("overrides"))

    resolved = _merge_clean_layers(profile_clean, config_overlays)
    for key, values in config_overrides.items():
        resolved[key] = list(values)

    protected_dirs = _dedupe(
        [
            *resolved.get("protected_dirs", []),
            *_HARD_PROTECTED_DIRS,
        ]
    )
    protected_globs = _dedupe(
        [
            *resolved.get("protected_globs", []),
            *_HARD_PROTECTED_GLOBS,
        ]
    )

    return CleanConfig(
        build_dirs=tuple(_dedupe(resolved.get("build_dirs", []))),
        build_globs=tuple(_dedupe(resolved.get("build_globs", []))),
        cache_dirs=tuple(_dedupe(resolved.get("cache_dirs", []))),
        cache_globs=tuple(_dedupe(resolved.get("cache_globs", []))),
        runtime_registry_dirs=tuple(
            _dedupe(resolved.get("runtime_registry_dirs", []))
        ),
        runtime_registry_globs=tuple(
            _dedupe(resolved.get("runtime_registry_globs", []))
        ),
        logs_dirs=tuple(_dedupe(resolved.get("logs_dirs", []))),
        logs_globs=tuple(_dedupe(resolved.get("logs_globs", []))),
        protected_dirs=tuple(protected_dirs),
        protected_globs=tuple(protected_globs),
    )


def _resolve_path_under_root(repo_root: Path, raw_path: str) -> Path | None:
    """Resolve one relative cleanup path under the repository root safely."""
    token = str(raw_path or "").strip()
    if not token:
        return None
    candidate = Path(token)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    root_path = Path(os.path.realpath(repo_root))
    resolved = Path(os.path.realpath(root_path / candidate))
    common = os.path.commonpath([str(root_path), str(resolved)])
    if common != str(root_path):
        return None
    return resolved


def _valid_glob_pattern(raw_pattern: str) -> str:
    """Return a safe repo-relative glob pattern or empty string."""
    token = str(raw_pattern or "").strip()
    if not token:
        return ""
    candidate = Path(token)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return token


def _is_relative_to(path: Path, other: Path) -> bool:
    """Return True when `path` is the same as or beneath `other`."""
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def _normalized_project_name_variants(raw_name: str) -> list[str]:
    """Return deterministic normalized project-name variants."""
    token = str(raw_name or "").strip()
    if not token:
        return []
    collapsed = re.sub(r"\s+", "-", token)
    variants = [
        token,
        token.lower(),
        collapsed,
        collapsed.lower(),
        collapsed.replace("_", "-"),
        collapsed.replace("_", "-").lower(),
        collapsed.replace("-", "_"),
        collapsed.replace("-", "_").lower(),
    ]
    return _dedupe(variants)


def _project_name_candidates(repo_root: Path) -> tuple[str, ...]:
    """Resolve plausible repo/project names for release-tree detection."""
    candidates: list[str] = []
    candidates.extend(_normalized_project_name_variants(repo_root.name))

    pyproject_path = repo_root / "pyproject.toml"
    if pyproject_path.exists():
        try:
            pyproject_payload = tomllib.loads(
                pyproject_path.read_text(encoding="utf-8")
            )
        except (OSError, tomllib.TOMLDecodeError):
            pyproject_payload = {}
        if isinstance(pyproject_payload, dict):
            project_block = pyproject_payload.get("project")
            if isinstance(project_block, dict):
                candidates.extend(
                    _normalized_project_name_variants(
                        str(project_block.get("name", ""))
                    )
                )
            tool_block = pyproject_payload.get("tool")
            if isinstance(tool_block, dict):
                poetry_block = tool_block.get("poetry")
                if isinstance(poetry_block, dict):
                    candidates.extend(
                        _normalized_project_name_variants(
                            str(poetry_block.get("name", ""))
                        )
                    )

    package_json_path = repo_root / "package.json"
    if package_json_path.exists():
        try:
            package_payload = json.loads(
                package_json_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            package_payload = {}
        if isinstance(package_payload, dict):
            candidates.extend(
                _normalized_project_name_variants(
                    str(package_payload.get("name", ""))
                )
            )

    return tuple(_dedupe(candidates))


def _collect_release_tree_targets(repo_root: Path) -> list[Path]:
    """Return repository-root unpacked release-tree directories for cleanup."""
    name_candidates = _project_name_candidates(repo_root)
    if not name_candidates:
        return []
    targets: list[Path] = []
    try:
        children = list(repo_root.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir():
            continue
        child_name = child.name
        for candidate in name_candidates:
            prefix = f"{candidate}-"
            if not child_name.startswith(prefix):
                continue
            suffix = child_name[len(prefix) :]
            if _RELEASE_TREE_SUFFIX_PATTERN.fullmatch(suffix):
                targets.append(child)
                break
    return _prune_nested_paths(targets)


def _prune_nested_paths(paths: Iterable[Path]) -> list[Path]:
    """Keep parent cleanup targets and drop redundant descendants."""
    unique = sorted(
        {path for path in paths}, key=lambda item: (len(item.parts), str(item))
    )
    kept: list[Path] = []
    for path in unique:
        if any(_is_relative_to(path, existing) for existing in kept):
            continue
        kept.append(path)
    return kept


def _collect_protected_paths(
    repo_root: Path,
    config: CleanConfig,
    *,
    extra_protected_paths: Iterable[Path] = (),
) -> list[Path]:
    """Resolve protected paths from explicit dirs and glob patterns."""
    protected: list[Path] = []
    normalized_repo_root = Path(os.path.realpath(repo_root))
    for raw_dir in config.protected_dirs:
        path = _resolve_path_under_root(repo_root, raw_dir)
        if path is None:
            continue
        protected.append(path)
    for raw_pattern in config.protected_globs:
        pattern = _valid_glob_pattern(raw_pattern)
        if not pattern:
            continue
        try:
            matches = repo_root.glob(pattern)
        except (OSError, ValueError):
            continue
        for path in matches:
            protected.append(path)
    for raw_path in extra_protected_paths:
        try:
            resolved = Path(os.path.realpath(raw_path))
        except OSError:
            continue
        if _is_relative_to(resolved, normalized_repo_root):
            protected.append(resolved)
    return _prune_nested_paths(protected)


def _managed_environment_protected_paths(repo_root: Path) -> tuple[Path, ...]:
    """Return cleanup roots resolved from managed-environment metadata."""
    try:
        return managed_environment_runtime.resolve_cleanup_protected_paths(
            repo_root
        )
    except ValueError:
        return ()


def _collect_requested_targets(
    repo_root: Path,
    config: CleanConfig,
    selection: CleanSelection,
) -> list[Path]:
    """Resolve requested cleanup targets for the selected categories."""
    requested: list[Path] = []
    if selection.include_build:
        requested.extend(_collect_release_tree_targets(repo_root))
        for raw_dir in config.build_dirs:
            path = _resolve_path_under_root(repo_root, raw_dir)
            if path is not None:
                requested.append(path)
        for raw_pattern in config.build_globs:
            pattern = _valid_glob_pattern(raw_pattern)
            if not pattern:
                continue
            try:
                matches = repo_root.glob(pattern)
            except (OSError, ValueError):
                continue
            requested.extend(matches)
    if selection.include_cache:
        for raw_dir in config.cache_dirs:
            path = _resolve_path_under_root(repo_root, raw_dir)
            if path is not None:
                requested.append(path)
        for raw_pattern in config.cache_globs:
            pattern = _valid_glob_pattern(raw_pattern)
            if not pattern:
                continue
            try:
                matches = repo_root.glob(pattern)
            except (OSError, ValueError):
                continue
            requested.extend(matches)
    if selection.include_runtime_registry:
        for raw_dir in config.runtime_registry_dirs:
            path = _resolve_path_under_root(repo_root, raw_dir)
            if path is not None:
                requested.append(path)
        for raw_pattern in config.runtime_registry_globs:
            pattern = _valid_glob_pattern(raw_pattern)
            if not pattern:
                continue
            try:
                matches = repo_root.glob(pattern)
            except (OSError, ValueError):
                continue
            requested.extend(matches)
    if selection.include_logs:
        for raw_dir in config.logs_dirs:
            path = _resolve_path_under_root(repo_root, raw_dir)
            if path is not None:
                requested.append(path)
        for raw_pattern in config.logs_globs:
            pattern = _valid_glob_pattern(raw_pattern)
            if not pattern:
                continue
            try:
                matches = repo_root.glob(pattern)
            except (OSError, ValueError):
                continue
            requested.extend(matches)
    existing = [
        path for path in requested if path.exists() or path.is_symlink()
    ]
    return _prune_nested_paths(existing)


def _conflicting_protected_path(
    path: Path, protected: Iterable[Path]
) -> Path | None:
    """Return the protecting path that conflicts with deleting `path`."""
    resolved_path = Path(os.path.realpath(path))
    for protected_path in protected:
        resolved_protected = Path(os.path.realpath(protected_path))
        if _is_relative_to(resolved_path, resolved_protected):
            return resolved_protected
        if _is_relative_to(resolved_protected, resolved_path):
            return resolved_protected
    return None


def _format_skipped_protected_summary(display_path: str, count: int) -> str:
    """Return a human-readable skipped-protected summary token."""
    if count <= 1:
        return display_path
    return f"{display_path} ({count} matches skipped)"


def _delete_path(target: Path) -> None:
    """Delete one file/symlink/directory target."""
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
        return
    if target.is_dir():
        shutil.rmtree(target)


def _repo_relative(path: Path, repo_root: Path) -> str:
    """Return one repo-relative display path from resolved cleanup paths."""
    resolved = Path(os.path.realpath(path))
    return display_path(resolved, repo_root=repo_root)


def execute_cleanup(
    repo_root: Path,
    selection: CleanSelection,
    *,
    extra_protected_paths: Iterable[Path] = (),
) -> CleanResult:
    """Execute cleanup for one repository and return structured results."""
    config = resolve_clean_config(repo_root)
    managed_environment_paths = _managed_environment_protected_paths(repo_root)
    protected = _collect_protected_paths(
        repo_root,
        config,
        extra_protected_paths=(
            *tuple(extra_protected_paths),
            *managed_environment_paths,
        ),
    )
    requested = _collect_requested_targets(repo_root, config, selection)

    removed_paths: list[str] = []
    skipped_counts: dict[str, int] = {}
    for path in requested:
        conflicting = _conflicting_protected_path(path, protected)
        if conflicting is not None:
            key = _repo_relative(conflicting, repo_root)
            skipped_counts[key] = skipped_counts.get(key, 0) + 1
            continue
        _delete_path(path)
        removed_paths.append(_repo_relative(path, repo_root))

    skipped_paths = tuple(
        _format_skipped_protected_summary(path, skipped_counts[path])
        for path in sorted(skipped_counts)
    )
    return CleanResult(
        selection=selection,
        removed_paths=tuple(sorted(removed_paths)),
        skipped_protected_paths=skipped_paths,
        skipped_protected_match_count=sum(skipped_counts.values()),
    )


def clean_repo(
    repo_root: Path,
    *,
    include_all: bool,
    include_build: bool,
    include_cache: bool,
    include_registry: bool,
    include_logs: bool,
) -> int:
    """Run repository cleanup for the selected cleanup categories."""
    if _gate_session_is_open(repo_root):
        runtime_print(
            (
                "Error: Cannot run `clean` while a gate session is open. "
                "Run `devcovenant gate --close` first, then run `devcovenant "
                "clean ...` outside the active session."
            ),
            file=sys.stderr,
        )
        return 1
    selection = resolve_clean_selection(
        include_all=include_all,
        include_build=include_build,
        include_cache=include_cache,
        include_registry=include_registry,
        include_logs=include_logs,
    )
    labels = ", ".join(selection.labels()) or "none"
    print_step(f"Cleanup scope: {labels}", "🧹")
    active_run_context = get_active_run_log_context()
    protected_run_dirs: tuple[Path, ...] = ()
    if active_run_context is not None:
        protected_run_dirs = (active_run_context.require_paths().run_dir,)

    result = execute_cleanup(
        repo_root,
        selection,
        extra_protected_paths=protected_run_dirs,
    )
    merge_active_run_log_metadata(
        {
            "clean_summary": {
                "selected_scopes": list(selection.labels()),
                "removed_count": len(result.removed_paths),
                "removed_paths": list(result.removed_paths),
                "skipped_protected_count": len(result.skipped_protected_paths),
                "skipped_protected_match_count": (
                    result.skipped_protected_match_count
                ),
                "skipped_protected_paths": list(
                    result.skipped_protected_paths
                ),
            }
        }
    )
    if result.removed_paths:
        print_step(
            f"Removed {len(result.removed_paths)} cleanup target(s)",
            "✅",
        )
        for path in result.removed_paths:
            runtime_print(f"Removed: {path}", verbose_only=True)
    else:
        print_step("No cleanup targets matched", "✅")

    if result.skipped_protected_paths:
        print_step(
            (
                "Skipped protected cleanup target(s): "
                + ", ".join(result.skipped_protected_paths)
            ),
            "🛡️",
        )
    return 0


def _gate_session_is_open(repo_root: Path) -> bool:
    """Return True when the runtime gate status records an open session."""
    status_path = registry_runtime.gate_status_path(repo_root)
    if not status_path.exists():
        return False
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    return str(payload.get("session_state", "")).strip().lower() == "open"
