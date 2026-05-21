"""Builtin policy/profile customization helpers."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import devcovenant.core.test_blueprints as test_blueprint_service
from devcovenant.core.repository_paths import display_path

CustomizationKind = Literal["policy", "profile"]


@dataclass(frozen=True)
class CustomizationPaths:
    """Resolved source and mirror paths for one builtin customization."""

    kind: CustomizationKind
    name: str
    builtin_source_root: Path
    custom_source_root: Path
    builtin_test_root: Path
    custom_test_root: Path
    builtin_blueprint_path: Path


def _normalize_policy_slug(policy_id: str) -> str:
    """Return the builtin filesystem slug for one policy id."""
    return str(policy_id or "").strip().lower().replace("-", "_")


def _normalize_profile_slug(profile_name: str) -> str:
    """Return the builtin filesystem slug for one profile name."""
    return str(profile_name or "").strip().lower()


def resolve_customization_paths(
    repo_root: Path,
    *,
    kind: CustomizationKind,
    name: str,
) -> CustomizationPaths:
    """Resolve builtin and custom paths for one policy or profile."""
    repo_root = Path(repo_root).resolve()
    normalized_name = str(name or "").strip()
    if kind == "policy":
        slug = _normalize_policy_slug(normalized_name)
        builtin_source_root = (
            repo_root / "devcovenant" / "builtin" / "policies" / slug
        )
        custom_source_root = (
            repo_root / "devcovenant" / "custom" / "policies" / slug
        )
        builtin_test_root = (
            repo_root / "tests" / "devcovenant" / "builtin" / "policies" / slug
        )
        custom_test_root = (
            repo_root / "tests" / "devcovenant" / "custom" / "policies" / slug
        )
    elif kind == "profile":
        slug = _normalize_profile_slug(normalized_name)
        builtin_source_root = (
            repo_root / "devcovenant" / "builtin" / "profiles" / slug
        )
        custom_source_root = (
            repo_root / "devcovenant" / "custom" / "profiles" / slug
        )
        builtin_test_root = (
            repo_root / "tests" / "devcovenant" / "builtin" / "profiles" / slug
        )
        custom_test_root = (
            repo_root / "tests" / "devcovenant" / "custom" / "profiles" / slug
        )
    else:  # pragma: no cover - defensive guard for literal narrowing
        raise ValueError(f"Unsupported customization kind: {kind}")

    return CustomizationPaths(
        kind=kind,
        name=slug,
        builtin_source_root=builtin_source_root,
        custom_source_root=custom_source_root,
        builtin_test_root=builtin_test_root,
        custom_test_root=custom_test_root,
        builtin_blueprint_path=builtin_source_root / "test_blueprints.yaml",
    )


def _copy_tree(source_root: Path, target_root: Path) -> None:
    """Copy one directory tree into place, replacing the old target."""
    if target_root.exists():
        shutil.rmtree(target_root)
    shutil.copytree(source_root, target_root)


def copy_builtin_customization(paths: CustomizationPaths) -> bool:
    """Copy one builtin policy or profile into the custom tree."""
    if not paths.builtin_source_root.exists():
        raise SystemExit(
            "Builtin customization target not found: "
            f"{display_path(paths.builtin_source_root)}."
        )
    _copy_tree(paths.builtin_source_root, paths.custom_source_root)
    return True


def remove_customization(paths: CustomizationPaths) -> bool:
    """Remove one custom policy or profile copy when it exists."""
    if not paths.custom_source_root.exists():
        return False
    shutil.rmtree(paths.custom_source_root)
    return True


def load_builtin_test_blueprints(
    paths: CustomizationPaths,
) -> list[test_blueprint_service.TestBlueprintFile]:
    """Load shipped test blueprints or fall back to the source test tree."""
    return test_blueprint_service.load_test_blueprints_for_tree(
        paths.builtin_blueprint_path,
        paths.builtin_test_root,
    )


def materialize_custom_tests(paths: CustomizationPaths) -> list[Path]:
    """Materialize custom test mirrors from builtin blueprints."""
    blueprints = load_builtin_test_blueprints(paths)
    if not blueprints:
        if paths.custom_test_root.exists():
            shutil.rmtree(paths.custom_test_root)
        return []
    return test_blueprint_service.materialize_test_blueprints(
        paths.custom_test_root,
        blueprints,
        overwrite=True,
    )


def remove_custom_tests(paths: CustomizationPaths) -> bool:
    """Remove mirrored custom tests for one builtin customization."""
    if not paths.custom_test_root.exists():
        return False
    shutil.rmtree(paths.custom_test_root)
    return True
