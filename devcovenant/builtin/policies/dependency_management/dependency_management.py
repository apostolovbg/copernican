"""DevCovenant policy: keep dependency artifacts synchronized."""

import fnmatch
import functools
import importlib.metadata as importlib_metadata
import io
import re
import tarfile
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

from packaging.requirements import Requirement

import devcovenant.core.policy_commands as policy_commands_service
import devcovenant.core.project_governance as project_governance_service
from devcovenant.core.policy_contract import (
    CheckContext,
    PolicyCheck,
    Violation,
)

LICENSES_README_NAME = "README.md"
LICENSE_INVENTORY_HEADING = "## Dependency License Inventory"
DEFAULT_REPORT_HEADING = "## License Report"
CANONICAL_DEPENDENCY_ROLES = (
    "intent",
    "resolved",
    "package_manifest",
)
RUNTIME_ACTION_REFRESH_ALL = "refresh-all"
_PROJECT_DEPENDENCIES_RE = re.compile(
    r"^\s*dependencies\s*=\s*\[(?P<rest>.*)$"
)
_PYTHON_LOCK_PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[^\s;\\]+)"
)
_DEPENDENCY_INVENTORY_RE = re.compile(
    r"^-\s+`(?P<name>[^`]+)`:\s+`(?P<path>[^`]+)`\s*$"
)
_LICENSE_NAME_RE = re.compile(
    r"^(license|licence|copying|notice)([.-]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class DependencySurfaceTarget:
    """One target used when hash-lock generation resolves a full closure."""

    target_id: str
    marker: str
    pip: Dict[str, str]


@dataclass(frozen=True)
class DependencySurface:
    """One declared dependency-management artifact surface."""

    surface_id: str
    enabled: bool
    active: bool
    lock_file: str
    direct_dependency_files: List[str]
    dependency_files: List[str]
    dependency_globs: List[str]
    dependency_dirs: List[str]
    third_party_file: str
    licenses_dir: str
    report_heading: str
    manage_licenses_readme: bool
    generate_hashes: bool
    required_paths: List[str]
    hash_targets: List[DependencySurfaceTarget]
    audit_service: str
    audit_ignore_ids: List[str]


@dataclass(frozen=True)
class DependencyLicenseSourceOverride:
    """One fallback source used when installed metadata lacks licenses."""

    package_name: str
    kind: str
    url: str
    member_globs: List[str]


@dataclass(frozen=True)
class DependencyLicenseSourceBundle:
    """Resolved upstream license texts and their provenance."""

    origin_description: str
    sources: List[tuple[str, str]]


def _normalize_list(value: object) -> list[str]:
    """Normalize metadata option values into non-empty string tokens."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = [value]
    elif isinstance(value, list):
        raw = [str(entry) for entry in value]
    else:
        raw = [str(value)]
    normalized = [entry.strip() for entry in raw]
    return [entry for entry in normalized if entry]


def _normalized_rel(path_text: str) -> str:
    """Normalize relative path tokens to forward-slash form."""
    return path_text.replace("\\", "/").strip()


def _validate_repo_relative_target(
    *,
    repo_root: Path,
    raw_value: str,
    label: str,
) -> Path:
    """Return validated absolute path for one repo-relative metadata target."""
    token = str(raw_value or "").strip()
    if not token:
        raise ValueError(
            f"dependency-management metadata is missing `{label}`."
        )
    relative = Path(token)
    if relative.is_absolute():
        raise ValueError(
            f"dependency-management `{label}` must be repo-relative."
        )
    repo_root_resolved = repo_root.resolve()
    absolute = (repo_root / relative).resolve()
    try:
        absolute.relative_to(repo_root_resolved)
    except ValueError as error:
        raise ValueError(
            "dependency-management metadata path must stay inside the "
            f"repository: `{label}` = `{token}`."
        ) from error
    return absolute


def _resolve_artifact_targets(
    *,
    repo_root: Path,
    third_party_file: str,
    licenses_dir: str,
) -> tuple[Path, Path]:
    """Resolve and validate report and license-directory metadata targets."""
    report_path = _validate_repo_relative_target(
        repo_root=repo_root,
        raw_value=third_party_file,
        label="third_party_file",
    )
    licenses_path = _validate_repo_relative_target(
        repo_root=repo_root,
        raw_value=licenses_dir,
        label="licenses_dir",
    )
    return report_path, licenses_path


def _normalize_bool(
    value: object,
    *,
    default: bool,
    label: str,
) -> bool:
    """Normalize one boolean-like metadata value."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    lowered = text.lower()
    if lowered in {"true", "1", "yes", "y", "on"}:
        return True
    if lowered in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"dependency-management `{label}` must be boolean.")


def _normalize_identifier_list(
    value: object,
    *,
    label: str,
) -> list[str]:
    """Normalize one identifier list into unique uppercase tokens."""

    del label
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_entry in _normalize_list(value):
        token = str(raw_entry).strip().upper()
        if not token or token in seen:
            continue
        seen.add(token)
        normalized.append(token)
    return normalized


def _normalize_audit_service(
    value: object,
    *,
    label: str,
) -> str:
    """Normalize one supported dependency vulnerability audit service."""

    token = str(value or "").strip().lower().replace("_", "-")
    if not token:
        return ""
    if token != "pypi":
        raise ValueError(
            f"dependency-management `{label}` must be `pypi` when set."
        )
    return token


def _render_surface_template(
    repo_root: Path,
    raw_value: object,
) -> str:
    """Render project-governance placeholders inside one surface value."""

    text = str(raw_value or "").strip()
    if not text:
        return ""
    if "{{" not in text or "}}" not in text:
        return text
    state = project_governance_service.resolve_runtime_state(repo_root)
    return project_governance_service.render_identity_placeholders(
        text,
        state,
    ).strip()


def _normalize_surface_paths(
    repo_root: Path,
    values: object,
    *,
    label: str,
) -> List[str]:
    """Normalize one repo-relative path list for a surface field."""

    normalized: List[str] = []
    for raw_entry in _normalize_list(values):
        rendered = _render_surface_template(repo_root, raw_entry)
        if not rendered:
            continue
        _validate_repo_relative_target(
            repo_root=repo_root,
            raw_value=rendered,
            label=label,
        )
        normalized.append(_normalized_rel(rendered))
    return normalized


def _normalize_surface_globs(values: object) -> List[str]:
    """Normalize one surface glob list."""

    return [
        _normalized_rel(entry)
        for entry in _normalize_list(values)
        if _normalized_rel(entry)
    ]


def _normalize_surface_dirs(
    repo_root: Path,
    values: object,
    *,
    label: str,
) -> List[str]:
    """Normalize one surface directory selector list."""

    normalized: List[str] = []
    for raw_entry in _normalize_list(values):
        rendered = _render_surface_template(repo_root, raw_entry)
        if not rendered:
            continue
        _validate_repo_relative_target(
            repo_root=repo_root,
            raw_value=rendered,
            label=label,
        )
        normalized.append(_normalized_rel(rendered).rstrip("/"))
    return normalized


def _normalize_surface_target(
    repo_root: Path,
    raw_value: object,
    *,
    surface_id: str,
    index: int,
) -> DependencySurfaceTarget:
    """Validate one hash-target mapping."""

    if not isinstance(raw_value, Mapping):
        raise ValueError(
            "dependency-management `surfaces` hash_targets entries must "
            "be mappings."
        )
    target_id = str(raw_value.get("id", "")).strip() or (
        f"{surface_id}-target-{index + 1}"
    )
    marker = str(raw_value.get("marker", "")).strip()
    if not marker:
        raise ValueError(
            "dependency-management `surfaces[].hash_targets[].marker` is "
            f"missing for `{surface_id}`."
        )
    raw_pip = raw_value.get("pip", {})
    if not isinstance(raw_pip, Mapping):
        raise ValueError(
            "dependency-management `surfaces[].hash_targets[].pip` must "
            f"be a mapping for `{surface_id}`."
        )
    pip_options: Dict[str, str] = {}
    for raw_key, raw_option in raw_pip.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        option = _render_surface_template(repo_root, raw_option)
        if option:
            pip_options[key] = option
    if not pip_options:
        raise ValueError(
            "dependency-management `surfaces[].hash_targets[].pip` is "
            f"empty for `{surface_id}`."
        )
    return DependencySurfaceTarget(
        target_id=target_id,
        marker=marker,
        pip=pip_options,
    )


def resolve_dependency_surfaces(
    *,
    repo_root: Path,
    raw_surfaces: object,
    include_inactive: bool = False,
) -> List[DependencySurface]:
    """Return normalized dependency surfaces from structured metadata."""

    if raw_surfaces in (None, "", []):
        return []
    if not isinstance(raw_surfaces, list):
        raise ValueError(
            "dependency-management `surfaces` must be a list of mappings."
        )
    surfaces: List[DependencySurface] = []
    seen_ids: set[str] = set()
    for index, raw_entry in enumerate(raw_surfaces):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                "dependency-management `surfaces` entries must be mappings."
            )
        surface_id = str(raw_entry.get("id", "")).strip()
        if not surface_id:
            raise ValueError(
                "dependency-management `surfaces[].id` is required."
            )
        if surface_id in seen_ids:
            raise ValueError(
                "dependency-management `surfaces` contains duplicate id "
                f"`{surface_id}`."
            )
        seen_ids.add(surface_id)

        enabled = _normalize_bool(
            raw_entry.get("enabled"),
            default=True,
            label=f"surfaces[{surface_id}].enabled",
        )
        lock_file = _render_surface_template(
            repo_root,
            raw_entry.get("lock_file", ""),
        )
        if lock_file:
            _validate_repo_relative_target(
                repo_root=repo_root,
                raw_value=lock_file,
                label=f"surfaces[{surface_id}].lock_file",
            )
        third_party_file = _render_surface_template(
            repo_root,
            raw_entry.get("third_party_file", ""),
        )
        licenses_dir = _render_surface_template(
            repo_root,
            raw_entry.get("licenses_dir", ""),
        )
        if third_party_file:
            _validate_repo_relative_target(
                repo_root=repo_root,
                raw_value=third_party_file,
                label=f"surfaces[{surface_id}].third_party_file",
            )
        if licenses_dir:
            _validate_repo_relative_target(
                repo_root=repo_root,
                raw_value=licenses_dir,
                label=f"surfaces[{surface_id}].licenses_dir",
            )
        direct_dependency_files = _normalize_surface_paths(
            repo_root,
            raw_entry.get("direct_dependency_files", []),
            label=f"surfaces[{surface_id}].direct_dependency_files",
        )
        _validate_dependency_input_paths(
            direct_dependency_files,
            label=f"surfaces[{surface_id}].direct_dependency_files",
        )
        dependency_roles = _normalize_dependency_roles(
            raw_entry.get(
                "dependency_roles",
                list(CANONICAL_DEPENDENCY_ROLES),
            )
        )
        role_dependency_files = _normalize_surface_paths(
            repo_root,
            _expand_role_selectors(
                entries=_normalize_list(
                    raw_entry.get("dependency_role_files", [])
                ),
                allowed_roles=dependency_roles,
                metadata_key=("surfaces[].dependency_role_files"),
            ),
            label=f"surfaces[{surface_id}].dependency_role_files",
        )
        dependency_files = _normalize_surface_paths(
            repo_root,
            raw_entry.get("dependency_files", []),
            label=f"surfaces[{surface_id}].dependency_files",
        )
        _validate_dependency_input_paths(
            [*role_dependency_files, *dependency_files],
            label=f"surfaces[{surface_id}].dependency_files",
        )
        dependency_globs = _normalize_surface_globs(
            [
                *(_normalize_list(raw_entry.get("dependency_globs", []))),
                *(
                    _expand_role_selectors(
                        entries=_normalize_list(
                            raw_entry.get("dependency_role_globs", [])
                        ),
                        allowed_roles=dependency_roles,
                        metadata_key=("surfaces[].dependency_role_globs"),
                    )
                ),
            ]
        )
        dependency_dirs = _normalize_surface_dirs(
            repo_root,
            [
                *(_normalize_list(raw_entry.get("dependency_dirs", []))),
                *(
                    _expand_role_selectors(
                        entries=_normalize_list(
                            raw_entry.get("dependency_role_dirs", [])
                        ),
                        allowed_roles=dependency_roles,
                        metadata_key=("surfaces[].dependency_role_dirs"),
                    )
                ),
            ],
            label=f"surfaces[{surface_id}].dependency_dirs",
        )
        required_paths = _normalize_surface_paths(
            repo_root,
            raw_entry.get("required_paths", []),
            label=f"surfaces[{surface_id}].required_paths",
        )
        report_heading = (
            str(raw_entry.get("report_heading", "")).strip()
            or DEFAULT_REPORT_HEADING
        )
        manage_licenses_readme = _normalize_bool(
            raw_entry.get("manage_licenses_readme"),
            default=True,
            label=f"surfaces[{surface_id}].manage_licenses_readme",
        )
        generate_hashes = _normalize_bool(
            raw_entry.get("generate_hashes"),
            default=False,
            label=f"surfaces[{surface_id}].generate_hashes",
        )
        audit_service = _normalize_audit_service(
            raw_entry.get("audit_service"),
            label=f"surfaces[{surface_id}].audit_service",
        )
        audit_ignore_ids = _normalize_identifier_list(
            raw_entry.get("audit_ignore_ids", []),
            label=f"surfaces[{surface_id}].audit_ignore_ids",
        )
        raw_hash_targets = raw_entry.get("hash_targets", [])
        if raw_hash_targets in ("", None):
            raw_hash_targets = []
        if not isinstance(raw_hash_targets, list):
            raise ValueError(
                "dependency-management `surfaces[].hash_targets` must be "
                f"a list for `{surface_id}`."
            )
        hash_targets = [
            _normalize_surface_target(
                repo_root,
                item,
                surface_id=surface_id,
                index=target_index,
            )
            for target_index, item in enumerate(raw_hash_targets)
        ]
        if enabled and not lock_file:
            raise ValueError(
                "dependency-management `surfaces[].lock_file` is missing "
                f"for `{surface_id}`."
            )
        if enabled and direct_dependency_files and not third_party_file:
            raise ValueError(
                "dependency-management `surfaces[].third_party_file` is "
                f"missing for `{surface_id}`."
            )
        if enabled and direct_dependency_files and not licenses_dir:
            raise ValueError(
                "dependency-management `surfaces[].licenses_dir` is "
                f"missing for `{surface_id}`."
            )
        if (
            enabled
            and generate_hashes
            and direct_dependency_files
            and not hash_targets
        ):
            raise ValueError(
                "dependency-management hash-locked surface "
                f"`{surface_id}` must declare `hash_targets`."
            )
        active = enabled
        for required in required_paths:
            if not (repo_root / required).exists():
                active = False
                break
        effective_dependency_files = list(
            dict.fromkeys(
                [
                    *direct_dependency_files,
                    *role_dependency_files,
                    *dependency_files,
                ]
            )
        )
        surface = DependencySurface(
            surface_id=surface_id,
            enabled=enabled,
            active=active,
            lock_file=_normalized_rel(lock_file),
            direct_dependency_files=direct_dependency_files,
            dependency_files=effective_dependency_files,
            dependency_globs=dependency_globs,
            dependency_dirs=dependency_dirs,
            third_party_file=_normalized_rel(third_party_file),
            licenses_dir=_normalized_rel(licenses_dir),
            report_heading=report_heading,
            manage_licenses_readme=manage_licenses_readme,
            generate_hashes=generate_hashes,
            required_paths=required_paths,
            hash_targets=hash_targets,
            audit_service=audit_service,
            audit_ignore_ids=audit_ignore_ids,
        )
        if include_inactive or surface.active:
            surfaces.append(surface)
    return surfaces


def dependency_surface_trigger_files(
    surface: DependencySurface,
) -> List[str]:
    """Return lock/manifests that trigger one dependency surface."""

    entries = [surface.lock_file, *surface.dependency_files]
    return [
        _normalized_rel(entry) for entry in entries if _normalized_rel(entry)
    ]


def dependency_surface_matches(
    surface: DependencySurface,
    rel_path: str,
) -> bool:
    """Return True when one repo-relative path belongs to a surface."""

    normalized = _normalized_rel(rel_path)
    if not normalized:
        return False
    if _is_profile_asset_template_dependency_input(normalized):
        return False
    if normalized == _normalized_rel(surface.lock_file):
        return True
    return _matches_dependency(
        normalized,
        dependency_files=surface.dependency_files,
        dependency_globs=surface.dependency_globs,
        dependency_dirs=surface.dependency_dirs,
    )


def dependency_surface_lock_refresh_requested(
    surface: DependencySurface,
    changed_dependency_files: Sequence[str],
) -> bool:
    """Return whether one surface's lock should refresh for changed files."""

    normalized = [
        _normalized_rel(str(entry))
        for entry in changed_dependency_files
        if _normalized_rel(str(entry))
    ]
    if not normalized:
        return True
    direct_triggers = {
        _normalized_rel(surface.lock_file),
        *[_normalized_rel(entry) for entry in surface.direct_dependency_files],
    }
    return any(entry in direct_triggers for entry in normalized)


def _relative_posix(path: Path, repo_root: Path) -> str | None:
    """Return repository-relative POSIX path for a changed file."""
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return None


def _is_profile_asset_template_dependency_input(rel_path: str) -> bool:
    """Return True when a path points at a profile-asset `.in` template."""
    normalized = _normalized_rel(rel_path)
    if not normalized or Path(normalized).suffix != ".in":
        return False
    parts = Path(normalized).parts
    return (
        len(parts) >= 6
        and parts[0] == "devcovenant"
        and parts[1] in {"builtin", "custom"}
        and parts[2] == "profiles"
        and parts[4] == "assets"
    )


def _validate_dependency_input_paths(
    values: Sequence[str],
    *,
    label: str,
) -> None:
    """Reject dependency inputs that point at profile-asset templates."""
    for entry in values:
        normalized = _normalized_rel(entry)
        if not _is_profile_asset_template_dependency_input(normalized):
            continue
        raise ValueError(
            "dependency-management "
            f"`{label}` may not target DevCovenant profile asset templates: "
            f"`{normalized}`."
        )


def _matches_dependency(
    rel_path: str,
    *,
    dependency_files: list[str],
    dependency_globs: list[str],
    dependency_dirs: list[str],
) -> bool:
    """Return True when a path matches dependency selector metadata."""
    for token in dependency_files:
        normalized = _normalized_rel(token)
        if not normalized:
            continue
        if rel_path == normalized:
            return True

    for token in dependency_globs:
        normalized = _normalized_rel(token)
        if normalized and fnmatch.fnmatch(rel_path, normalized):
            return True

    for token in dependency_dirs:
        normalized = _normalized_rel(token).strip("/")
        if not normalized:
            continue
        if rel_path == normalized or rel_path.startswith(f"{normalized}/"):
            return True

    return False


def _normalize_dependency_roles(raw: object) -> list[str]:
    """Normalize configured dependency roles."""
    tokens = _normalize_list(raw)
    if not tokens:
        return list(CANONICAL_DEPENDENCY_ROLES)
    normalized = [token.lower() for token in tokens]
    if len(set(normalized)) != len(normalized):
        raise ValueError(
            "dependency-management `dependency_roles` contains duplicates."
        )
    unknown = [
        token
        for token in normalized
        if token not in CANONICAL_DEPENDENCY_ROLES
    ]
    if unknown:
        listed = ", ".join(sorted(unknown))
        raise ValueError(
            "dependency-management `dependency_roles` contains unsupported "
            f"roles: {listed}."
        )
    return normalized


def resolve_dependency_roles(raw: object) -> list[str]:
    """Public helper for validating dependency role metadata."""
    return _normalize_dependency_roles(raw)


def resolve_license_source_overrides(
    raw: object,
) -> dict[str, DependencyLicenseSourceOverride]:
    """Return normalized fallback license sources keyed by package name."""

    if raw in (None, "", []):
        return {}
    if not isinstance(raw, list):
        raise ValueError(
            "dependency-management `license_source_overrides` must be a "
            "list of mappings."
        )
    overrides: dict[str, DependencyLicenseSourceOverride] = {}
    for raw_entry in raw:
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                "dependency-management `license_source_overrides` entries "
                "must be mappings."
            )
        raw_id = str(raw_entry.get("id", "")).strip()
        if not raw_id:
            raise ValueError(
                "dependency-management "
                "`license_source_overrides[].id` is required."
            )
        package_name = _normalize_distribution_name(raw_id)
        if not package_name:
            raise ValueError(
                "dependency-management "
                "`license_source_overrides[].id` must name a package."
            )
        if package_name in overrides:
            raise ValueError(
                "dependency-management `license_source_overrides` contains "
                f"duplicate id `{package_name}`."
            )
        kind = str(raw_entry.get("kind", "")).strip().lower()
        if kind != "archive_url":
            raise ValueError(
                "dependency-management "
                "`license_source_overrides[].kind` must be "
                "`archive_url`."
            )
        url = str(raw_entry.get("url", "")).strip()
        if not url:
            raise ValueError(
                "dependency-management "
                "`license_source_overrides[].url` is required."
            )
        member_globs = _normalize_surface_globs(
            raw_entry.get("member_globs", [])
        )
        if not member_globs:
            raise ValueError(
                "dependency-management "
                "`license_source_overrides[].member_globs` is required."
            )
        overrides[package_name] = DependencyLicenseSourceOverride(
            package_name=package_name,
            kind=kind,
            url=url,
            member_globs=member_globs,
        )
    return overrides


def parse_role_selector_entries(
    *,
    entries: list[str],
    allowed_roles: list[str],
    metadata_key: str,
) -> list[tuple[str, str]]:
    """
    Parse `role=>selector` entries and validate declared roles.
    """
    pairs: list[tuple[str, str]] = []
    for entry in entries:
        if "=>" not in entry:
            raise ValueError(
                "dependency-management role selector entries must use "
                f"`role=>selector` format in `{metadata_key}`."
            )
        role, selector = entry.split("=>", 1)
        role_token = role.strip().lower()
        selector_token = selector.strip()
        if not role_token or not selector_token:
            raise ValueError(
                "dependency-management role selector entries must include "
                f"both role and selector in `{metadata_key}`."
            )
        if role_token not in allowed_roles:
            raise ValueError(
                "dependency-management role selector uses role "
                f"`{role_token}` outside configured `dependency_roles`."
            )
        pairs.append((role_token, selector_token))
    return pairs


def _expand_role_selectors(
    *,
    entries: list[str],
    allowed_roles: list[str],
    metadata_key: str,
) -> list[str]:
    """Expand `role=>selector` entries into selector tokens."""
    return [
        selector
        for _, selector in parse_role_selector_entries(
            entries=entries,
            allowed_roles=allowed_roles,
            metadata_key=metadata_key,
        )
    ]


def _render_licenses_readme(third_party_file: str) -> str:
    """Build generic README text for the licenses directory."""
    lines = [
        "# License Assets",
        "",
        "## Table of Contents",
        "- [Overview](#overview)",
        "- [Contents](#contents)",
        "- [Notes](#notes)",
        "",
        "## Overview",
        "This directory ships generated license artifacts for one",
        "dependency surface.",
        "Operators, auditors, and downstream recipients can use these",
        "files to inspect the reported dependencies and bundled license",
        "texts that shipped with this artifact.",
        "For most users they are reference material rather than files",
        "that need direct maintenance.",
        "",
        "## Contents",
        f"- `{Path(third_party_file).name}` records the dependency inputs and",
        "  generated license inventory for this dependency surface.",
        "- `*.txt` files store the generated upstream license texts that",
        "  match the current direct dependency set.",
        "",
        "## Notes",
        "- Most users can treat these files as shipped reference material.",
        "- DevCovenant regenerates the matching report and license texts",
        "  together during dependency refresh work.",
        "- Direct manual edits should be rare because refresh owns the",
        "  generated contents.",
        "",
    ]
    return "\n".join(lines)


def _extract_license_report(text: str, heading: str) -> str:
    """Extract the text inside the License Report section."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == heading.lower():
            start = index
            break

    if start is None:
        return ""

    # Collect lines until the next section header
    section_lines: List[str] = [lines[start]]
    remaining = iter(lines)
    for _ in range(start + 1):
        next(remaining, None)
    for line in remaining:
        stripped = line.strip()
        header_prefix = stripped.startswith("## ")
        header_not_report = not stripped.lower().startswith(heading.lower())
        if header_prefix and header_not_report:
            break
        section_lines.append(line)

    return "\n".join(section_lines)


def _extract_section(text: str, heading: str) -> str:
    """Extract the text inside one markdown section by heading label."""
    return _extract_license_report(text, heading)


def _contains_reference(section: str, needle: str) -> bool:
    """Case-insensitive search inside the license report."""
    return needle.lower() in section.lower()


def _normalize_report_entries(
    changed_dependency_files: Iterable[str],
    *,
    resolved_lock_file: str = "",
) -> list[str]:
    """Normalize dependency entries for deterministic report rendering."""
    entries: set[str] = set()
    for entry in changed_dependency_files:
        normalized = _normalized_rel(entry)
        if normalized:
            entries.add(normalized)
    normalized_lock = _normalized_rel(resolved_lock_file)
    if normalized_lock:
        entries.add(normalized_lock)
    return sorted(entries)


def _render_report_section(
    heading: str,
    changed_dependency_files: Iterable[str],
    *,
    resolved_lock_file: str = "",
) -> str:
    """Render deterministic `License Report` section content."""
    lines: List[str] = [heading]
    for dep_file in _normalize_report_entries(
        changed_dependency_files,
        resolved_lock_file=resolved_lock_file,
    ):
        lines.append(f"- `{dep_file}`")
    return "\n".join(lines)


def _normalize_distribution_name(name: str) -> str:
    """Return the PEP 503-normalized form of a distribution name."""
    return re.sub(r"[-_.]+", "-", str(name or "").strip()).lower()


def _parse_requirement_strings(requirement_lines: Iterable[str]) -> list[str]:
    """Extract dependency names from requirement-style strings."""
    names: list[str] = []
    for raw_line in requirement_lines:
        stripped = str(raw_line).split("#", 1)[0].strip()
        if not stripped or stripped.startswith("-"):
            continue
        requirement = Requirement(stripped)
        names.append(requirement.name)
    return names


def _extract_requirements_include_target(raw_line: str) -> str | None:
    """Return the referenced path for supported requirements includes."""

    stripped = str(raw_line).strip()
    if not stripped:
        return None
    for prefix in ("-r", "--requirement"):
        if stripped == prefix:
            return None
        if stripped.startswith(prefix + "="):
            return stripped.split("=", 1)[1].strip()
        if stripped.startswith(prefix + " "):
            return stripped.split(None, 1)[1].strip()
        if prefix == "-r" and stripped.startswith("-r") and len(stripped) > 2:
            return stripped[2:].strip()
    return None


def _normalized_requirement_manifest_line(raw_line: str) -> str:
    """Normalize one requirement-manifest line for inventory parsing."""

    raw_text = str(raw_line).split("#", 1)[0].rstrip()
    stripped = raw_text.strip()
    if not stripped:
        return ""
    if raw_text[:1].isspace():
        return ""
    if stripped.startswith("--hash="):
        return ""
    if stripped.endswith("\\"):
        stripped = stripped[:-1].strip()
    return stripped


def _parse_requirements_in(path: Path) -> list[str]:
    """Return direct dependency names declared in one requirements file."""
    if not path.exists():
        return []
    return _parse_requirement_strings(
        path.read_text(encoding="utf-8").splitlines()
    )


def _parse_pyproject_dependency_strings(path: Path) -> list[str]:
    """Return raw dependency requirement strings from pyproject metadata."""
    if not path.exists():
        return []
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    project_block = payload.get("project", {})
    raw_dependencies = project_block.get("dependencies", [])
    if isinstance(raw_dependencies, list):
        return [str(entry) for entry in raw_dependencies]
    return []


def _parse_pyproject_dependencies(path: Path) -> list[str]:
    """Return direct dependency names declared in pyproject metadata."""
    return _parse_requirement_strings(
        _parse_pyproject_dependency_strings(path)
    )


def _direct_dependency_strings_from_file(path: Path) -> list[str]:
    """Return raw dependency strings from one supported manifest."""
    if not path.exists() or not path.is_file():
        return []
    if path.name == "pyproject.toml":
        return _parse_pyproject_dependency_strings(path)
    return [
        str(raw_line).split("#", 1)[0].strip()
        for raw_line in path.read_text(encoding="utf-8").splitlines()
        if str(raw_line).split("#", 1)[0].strip()
    ]


def _direct_dependency_names_from_file(path: Path) -> list[str]:
    """Return direct dependency names declared in one supported manifest."""
    return _parse_requirement_strings(
        _direct_dependency_strings_from_file(path)
    )


def _inventory_dependency_strings_from_manifest(
    manifest_path: Path,
    *,
    seen_paths: set[Path],
) -> list[str]:
    """Collect inventory dependency strings with recursive include support."""

    if not manifest_path.exists() or not manifest_path.is_file():
        return []
    resolved_path = manifest_path.resolve()
    if resolved_path in seen_paths:
        return []
    seen_paths.add(resolved_path)
    if manifest_path.name == "pyproject.toml":
        return _parse_pyproject_dependency_strings(manifest_path)
    collected: list[str] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        normalized_line = _normalized_requirement_manifest_line(raw_line)
        if not normalized_line:
            continue
        include_target = _extract_requirements_include_target(normalized_line)
        if include_target is None:
            collected.append(normalized_line)
            continue
        include_path = (manifest_path.parent / include_target).resolve()
        collected.extend(
            _inventory_dependency_strings_from_manifest(
                include_path,
                seen_paths=seen_paths,
            )
        )
    return collected


def _direct_dependency_display_names(
    repo_root: Path,
    *,
    direct_dependency_files: Iterable[str] | None = None,
) -> dict[str, str]:
    """Return normalized direct dependency names mapped to display casing."""
    display_names: dict[str, str] = {}
    selector_paths = [
        repo_root / "requirements.in",
        repo_root / "pyproject.toml",
    ]
    if direct_dependency_files is not None:
        selector_paths = []
        for raw_path in direct_dependency_files:
            token = _normalized_rel(str(raw_path))
            if token:
                selector_paths.append(repo_root / token)
    candidates: list[str] = []
    for manifest_path in selector_paths:
        candidates.extend(
            _parse_requirement_strings(
                _inventory_dependency_strings_from_manifest(
                    manifest_path,
                    seen_paths=set(),
                )
            )
        )
    for name in candidates:
        normalized = _normalize_distribution_name(name)
        if normalized and normalized not in display_names:
            display_names[normalized] = name
    return display_names


def _resolved_python_lock_versions(
    repo_root: Path,
    *,
    resolved_lock_file: str = "requirements.lock",
) -> dict[str, tuple[str, str]]:
    """Return normalized lockfile names mapped to display name/version."""
    lock_path = repo_root / _normalized_rel(resolved_lock_file)
    if not lock_path.exists():
        return {}
    resolved: dict[str, tuple[str, str]] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or raw_line[:1].isspace() or stripped.startswith("#"):
            continue
        match = _PYTHON_LOCK_PIN_RE.match(stripped)
        if match is None:
            continue
        display_name = str(match.group("name"))
        version = str(match.group("version"))
        normalized = _normalize_distribution_name(display_name)
        resolved[normalized] = (display_name, version)
    return resolved


def _find_distribution(name: str):
    """Load installed distribution metadata for one dependency name."""
    candidates = [name]
    normalized = _normalize_distribution_name(name)
    if normalized not in candidates:
        candidates.append(normalized)
    for candidate in candidates:
        try:
            return importlib_metadata.distribution(candidate)
        except importlib_metadata.PackageNotFoundError:
            continue
    raise importlib_metadata.PackageNotFoundError(name)


def _installed_distribution_license_sources(dist) -> list[tuple[str, str]]:
    """Return bundled upstream license texts for one installed distribution."""
    files = dist.files or []
    collected: list[tuple[str, str]] = []
    for entry in sorted(files, key=lambda item: str(item).lower()):
        entry_text = str(entry)
        if (
            ".dist-info/" not in entry_text
            and ".dist-info\\" not in entry_text
        ):
            continue
        name = Path(entry_text).name
        if not _LICENSE_NAME_RE.match(name):
            continue
        located = Path(dist.locate_file(entry))
        if not located.exists() or not located.is_file():
            continue
        collected.append((name, located.read_text(encoding="utf-8")))
    return collected


def _render_license_source_template(
    template: str,
    *,
    package_name: str,
    normalized_name: str,
    version: str,
) -> str:
    """Render package/version placeholders inside license-source metadata."""

    try:
        return str(template).format(
            package_name=package_name,
            normalized_name=normalized_name,
            version=version,
        )
    except KeyError as error:
        missing_key = str(error).strip("'")
        raise ValueError(
            "dependency-management license-source templates may use only "
            "`{package_name}`, `{normalized_name}`, and `{version}`; "
            f"`{missing_key}` is unsupported."
        ) from error
    except ValueError as error:
        raise ValueError(
            "dependency-management license-source templates must use valid "
            f"`str.format` syntax: {error}"
        ) from error


@functools.lru_cache(maxsize=32)
def _read_url_bytes(url: str) -> bytes:
    """Return cached bytes fetched from one remote license-source URL."""

    with urllib.request.urlopen(url) as response:  # nosec B310
        return response.read()


def _archive_license_sources_from_bytes(
    raw_bytes: bytes,
    *,
    member_globs: Sequence[str],
    source_label: str,
) -> list[tuple[str, str]]:
    """Extract matching UTF-8 license texts from one archive payload."""

    collected: list[tuple[str, str]] = []
    buffer = io.BytesIO(raw_bytes)
    if zipfile.is_zipfile(buffer):
        buffer.seek(0)
        with zipfile.ZipFile(buffer) as archive:
            for member in sorted(
                archive.infolist(),
                key=lambda item: item.filename.lower(),
            ):
                member_name = str(member.filename)
                if member.is_dir() or not any(
                    fnmatch.fnmatch(member_name, pattern)
                    for pattern in member_globs
                ):
                    continue
                try:
                    source_text = archive.read(member).decode("utf-8")
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        "Archive license source "
                        f"`{source_label}` contains non-UTF-8 member "
                        f"`{member_name}`."
                    ) from error
                collected.append((member_name, source_text))
        return collected

    buffer.seek(0)
    try:
        with tarfile.open(fileobj=buffer, mode="r:*") as archive:
            for member in sorted(
                archive.getmembers(),
                key=lambda item: item.name.lower(),
            ):
                member_name = str(member.name)
                if not member.isfile() or not any(
                    fnmatch.fnmatch(member_name, pattern)
                    for pattern in member_globs
                ):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                try:
                    source_text = extracted.read().decode("utf-8")
                except UnicodeDecodeError as error:
                    raise RuntimeError(
                        "Archive license source "
                        f"`{source_label}` contains non-UTF-8 member "
                        f"`{member_name}`."
                    ) from error
                collected.append((member_name, source_text))
    except tarfile.TarError as error:
        raise RuntimeError(
            "dependency-management archive license source "
            f"`{source_label}` must be a supported tar or zip archive."
        ) from error
    return collected


def _archive_license_source_bundle(
    *,
    package_name: str,
    normalized_name: str,
    version: str,
    override: DependencyLicenseSourceOverride,
) -> DependencyLicenseSourceBundle:
    """Resolve one archive-backed fallback license-source bundle."""

    resolved_url = _render_license_source_template(
        override.url,
        package_name=package_name,
        normalized_name=normalized_name,
        version=version,
    )
    resolved_member_globs = [
        _render_license_source_template(
            member_glob,
            package_name=package_name,
            normalized_name=normalized_name,
            version=version,
        )
        for member_glob in override.member_globs
    ]
    collected = _archive_license_sources_from_bytes(
        _read_url_bytes(resolved_url),
        member_globs=resolved_member_globs,
        source_label=resolved_url,
    )
    if not collected:
        raise RuntimeError(
            "dependency-management archive license source "
            f"`{resolved_url}` matched no members for `{package_name}`."
        )
    return DependencyLicenseSourceBundle(
        origin_description=f"source archive `{resolved_url}`",
        sources=collected,
    )


def _resolve_dependency_license_bundle(
    *,
    package_name: str,
    version: str,
    dist,
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> DependencyLicenseSourceBundle:
    """Resolve upstream license texts from installed metadata or fallback."""

    installed_sources = _installed_distribution_license_sources(dist)
    if installed_sources:
        return DependencyLicenseSourceBundle(
            origin_description="installed distribution metadata",
            sources=installed_sources,
        )
    normalized_name = _normalize_distribution_name(package_name)
    override = (license_source_overrides or {}).get(normalized_name)
    if override is not None:
        return _archive_license_source_bundle(
            package_name=package_name,
            normalized_name=normalized_name,
            version=version,
            override=override,
        )
    raise RuntimeError(
        "No upstream license files were found in the installed "
        f"distribution metadata for `{package_name}`, and no "
        "`license_source_overrides` entry is configured for "
        f"`{normalized_name}`."
    )


def _render_dependency_license_text(
    *,
    package_name: str,
    version: str,
    origin_description: str,
    sources: list[tuple[str, str]],
) -> str:
    """Render one local aggregate license text for a direct dependency."""
    lines = [
        f"# {package_name} {version}",
        "",
        "This file aggregates the upstream license texts resolved for",
        f"`{package_name}=={version}`.",
        "",
        f"Resolved from: {origin_description}",
        "",
        "Included upstream files:",
    ]
    for source_name, _ in sources:
        lines.append(f"- {source_name}")
    for source_name, source_text in sources:
        lines.extend(
            [
                "",
                f"===== {source_name} =====",
                "",
                source_text.rstrip(),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _inventory_entry_path(
    licenses_dir_path: Path,
    *,
    package_name: str,
    version: str,
) -> Path:
    """Return the managed local license-text path for one dependency."""
    return licenses_dir_path / f"{package_name}-{version}.txt"


def _build_dependency_inventory(
    repo_root: Path,
    *,
    licenses_dir_path: Path,
    resolved_lock_file: str = "requirements.lock",
    direct_dependency_files: Iterable[str] | None = None,
) -> list[dict[str, str]]:
    """Resolve direct dependency inventory with generated license targets."""
    direct_display_names = _direct_dependency_display_names(
        repo_root,
        direct_dependency_files=direct_dependency_files,
    )
    if not direct_display_names:
        return []
    resolved_versions = _resolved_python_lock_versions(
        repo_root,
        resolved_lock_file=resolved_lock_file,
    )
    inventory: list[dict[str, str]] = []
    for normalized_name in sorted(direct_display_names):
        if normalized_name not in resolved_versions:
            continue
        _, version = resolved_versions[normalized_name]
        display_name = direct_display_names[normalized_name]
        try:
            _find_distribution(display_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        inventory.append(
            {
                "normalized_name": normalized_name,
                "package_name": display_name,
                "version": version,
                "relative_path": _inventory_entry_path(
                    licenses_dir_path,
                    package_name=display_name,
                    version=version,
                ).name,
            }
        )
    return inventory


def _render_inventory_section(
    *,
    licenses_dir: str,
    inventory: list[dict[str, str]],
) -> str:
    """Render deterministic dependency inventory section content."""
    lines = [LICENSE_INVENTORY_HEADING]
    for entry in inventory:
        relative_path = _normalized_rel(
            str(Path(licenses_dir) / str(entry["relative_path"]))
        )
        lines.append(
            f"- `{entry['package_name']}=={entry['version']}`: "
            f"`{relative_path}`"
        )
    return "\n".join(lines)


def _render_full_report(
    *,
    licenses_dir: str,
    report_section: str,
    inventory_section: str,
) -> str:
    """Render one deterministic third-party report document."""
    normalized_licenses_dir = _normalized_rel(licenses_dir).strip("/")
    return (
        "# Third-Party Licenses\n\n"
        "This report lists the direct third-party dependencies declared in\n"
        "the tracked dependency manifests and the corresponding license\n"
        f"texts stored under `{normalized_licenses_dir}/`.\n\n"
        f"{report_section}\n\n"
        f"{inventory_section}\n"
    )


def _inventory_paths_from_report(text: str) -> set[str]:
    """Return repo-relative license artifact paths listed in the inventory."""
    section = _extract_section(text, LICENSE_INVENTORY_HEADING)
    paths: set[str] = set()
    for line in section.splitlines():
        match = _DEPENDENCY_INVENTORY_RE.match(line.strip())
        if match is None:
            continue
        paths.add(_normalized_rel(str(match.group("path"))))
    return paths


def _replace_report_section(
    text: str,
    *,
    heading: str,
    replacement: str,
) -> str:
    """Replace one report section or append it when missing."""
    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip().lower() == heading.lower():
            start = index
            break

    if start is None:
        if not text.strip():
            return replacement + "\n"
        return text.rstrip() + "\n\n" + replacement + "\n"

    end = len(lines)
    for index in range(start + 1, len(lines)):
        stripped = lines[index].strip()
        if stripped.startswith("## ") and stripped.lower() != heading.lower():
            end = index
            break

    replacement_lines = replacement.splitlines()
    if end < len(lines) and replacement_lines:
        if replacement_lines[-1].strip():
            replacement_lines.append("")
    updated_lines = lines[:start] + replacement_lines + lines[end:]
    return "\n".join(updated_lines).rstrip() + "\n"


def _sync_dependency_license_files(
    *,
    repo_root: Path,
    licenses_dir_path: Path,
    licenses_dir: str,
    inventory: list[dict[str, str]],
    existing_inventory_paths: set[str],
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> list[Path]:
    """Materialize current dependency license texts and prune stale ones."""
    modified: list[Path] = []
    normalized_licenses_dir = _normalized_rel(licenses_dir).strip("/")
    desired_inventory_paths = {
        _normalized_rel(str(Path(licenses_dir) / entry["relative_path"]))
        for entry in inventory
    }
    for entry in inventory:
        package_name = str(entry["package_name"])
        version = str(entry["version"])
        target_path = licenses_dir_path / str(entry["relative_path"])
        dist = _find_distribution(package_name)
        bundle = _resolve_dependency_license_bundle(
            package_name=package_name,
            version=version,
            dist=dist,
            license_source_overrides=license_source_overrides,
        )
        rendered = _render_dependency_license_text(
            package_name=package_name,
            version=version,
            origin_description=bundle.origin_description,
            sources=bundle.sources,
        )
        if not target_path.exists() or (
            target_path.read_text(encoding="utf-8") != rendered
        ):
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(rendered, encoding="utf-8")
            modified.append(target_path)

    stale_inventory_paths = sorted(
        relative_path
        for relative_path in (
            existing_inventory_paths - desired_inventory_paths
        )
        if relative_path == normalized_licenses_dir
        or relative_path.startswith(f"{normalized_licenses_dir}/")
    )
    for extra_file in sorted(licenses_dir_path.glob("*.txt")):
        relative_path = _normalized_rel(
            str(extra_file.resolve().relative_to(repo_root.resolve()))
        )
        if relative_path not in desired_inventory_paths:
            stale_inventory_paths.append(relative_path)
    stale_inventory_paths = sorted(set(stale_inventory_paths))
    for relative_path in stale_inventory_paths:
        stale_path = repo_root / relative_path
        if stale_path.exists() and stale_path.is_file():
            stale_path.unlink()
            modified.append(stale_path)
    return modified


def _ensure_licenses_readme(
    *,
    licenses_dir_path: Path,
    third_party_file: str,
) -> Path | None:
    """Ensure licenses/README.md exists with generic, metadata-driven text."""
    readme_path = licenses_dir_path / LICENSES_README_NAME
    desired = _render_licenses_readme(third_party_file)
    if readme_path.exists():
        existing = readme_path.read_text(encoding="utf-8")
        if existing == desired:
            return None
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text(desired, encoding="utf-8")
    return readme_path


def _licenses_dir_is_in_sync(
    *,
    repo_root: Path,
    licenses_dir_path: Path,
    licenses_dir: str,
    third_party_file: str,
    inventory: list[dict[str, str]],
    existing_inventory_paths: set[str],
    manage_licenses_readme: bool,
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> bool:
    """Return True when generated license artifacts already match runtime."""
    normalized_licenses_dir = _normalized_rel(licenses_dir).strip("/")
    desired_inventory_paths = {
        _normalized_rel(str(Path(licenses_dir) / entry["relative_path"]))
        for entry in inventory
    }
    for entry in inventory:
        package_name = str(entry["package_name"])
        version = str(entry["version"])
        target_path = licenses_dir_path / str(entry["relative_path"])
        if not target_path.exists():
            return False
        dist = _find_distribution(package_name)
        bundle = _resolve_dependency_license_bundle(
            package_name=package_name,
            version=version,
            dist=dist,
            license_source_overrides=license_source_overrides,
        )
        rendered = _render_dependency_license_text(
            package_name=package_name,
            version=version,
            origin_description=bundle.origin_description,
            sources=bundle.sources,
        )
        if target_path.read_text(encoding="utf-8") != rendered:
            return False

    stale_inventory_paths = sorted(
        relative_path
        for relative_path in (
            existing_inventory_paths - desired_inventory_paths
        )
        if relative_path == normalized_licenses_dir
        or relative_path.startswith(f"{normalized_licenses_dir}/")
    )
    for extra_file in sorted(licenses_dir_path.glob("*.txt")):
        relative_path = _normalized_rel(str(extra_file.relative_to(repo_root)))
        if relative_path not in desired_inventory_paths:
            return False
    for relative_path in stale_inventory_paths:
        stale_path = repo_root / relative_path
        if stale_path.exists() and stale_path.is_file():
            return False

    if not manage_licenses_readme:
        return True
    readme_path = licenses_dir_path / LICENSES_README_NAME
    desired_readme = _render_licenses_readme(third_party_file)
    if not readme_path.exists():
        return False
    return readme_path.read_text(encoding="utf-8") == desired_readme


def _license_artifacts_need_refresh(
    *,
    repo_root: Path,
    changed_dependency_files: Iterable[str],
    third_party_file: str,
    licenses_dir: str,
    report_heading: str,
    resolved_lock_file: str = "requirements.lock",
    direct_dependency_files: Iterable[str] | None = None,
    manage_licenses_readme: bool = True,
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> tuple[bool, bool]:
    """Return whether the report or licenses directory is out of sync."""
    third_party_path, licenses_dir_path = _resolve_artifact_targets(
        repo_root=repo_root,
        third_party_file=third_party_file,
        licenses_dir=licenses_dir,
    )
    third_party_rel = third_party_path.relative_to(
        repo_root.resolve()
    ).as_posix()
    if third_party_path.exists():
        existing = third_party_path.read_text(encoding="utf-8")
    else:
        existing = "# Third-Party Licenses\n"

    inventory = _build_dependency_inventory(
        repo_root,
        licenses_dir_path=licenses_dir_path,
        resolved_lock_file=resolved_lock_file,
        direct_dependency_files=direct_dependency_files,
    )
    report_section = _render_report_section(
        report_heading,
        changed_dependency_files,
        resolved_lock_file=resolved_lock_file,
    )
    existing_inventory_paths = _inventory_paths_from_report(existing)
    inventory_section = _render_inventory_section(
        licenses_dir=licenses_dir,
        inventory=inventory,
    )
    updated_report = _render_full_report(
        licenses_dir=licenses_dir,
        report_section=report_section,
        inventory_section=inventory_section,
    )
    report_needs_refresh = updated_report != existing
    licenses_dir_needs_refresh = not _licenses_dir_is_in_sync(
        repo_root=repo_root,
        licenses_dir_path=licenses_dir_path,
        licenses_dir=licenses_dir,
        third_party_file=third_party_rel,
        inventory=inventory,
        existing_inventory_paths=existing_inventory_paths,
        manage_licenses_readme=manage_licenses_readme,
        license_source_overrides=license_source_overrides,
    )
    return report_needs_refresh, licenses_dir_needs_refresh


def refresh_license_artifacts(
    repo_root: Path,
    *,
    changed_dependency_files: Iterable[str],
    third_party_file: str,
    licenses_dir: str,
    report_heading: str,
    resolved_lock_file: str = "requirements.lock",
    direct_dependency_files: Iterable[str] | None = None,
    manage_licenses_readme: bool = True,
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> List[Path]:
    """Refresh configured report file and licenses marker files."""

    modified: List[Path] = []
    third_party_path, licenses_dir_path = _resolve_artifact_targets(
        repo_root=repo_root,
        third_party_file=third_party_file,
        licenses_dir=licenses_dir,
    )
    third_party_rel = third_party_path.relative_to(
        repo_root.resolve()
    ).as_posix()
    if third_party_path.exists():
        existing = third_party_path.read_text(encoding="utf-8")
    else:
        existing = "# Third-Party Licenses\n"

    inventory = _build_dependency_inventory(
        repo_root,
        licenses_dir_path=licenses_dir_path,
        resolved_lock_file=resolved_lock_file,
        direct_dependency_files=direct_dependency_files,
    )
    report_section = _render_report_section(
        report_heading,
        changed_dependency_files,
        resolved_lock_file=resolved_lock_file,
    )
    existing_inventory_paths = _inventory_paths_from_report(existing)
    inventory_section = _render_inventory_section(
        licenses_dir=licenses_dir,
        inventory=inventory,
    )
    updated_report = _render_full_report(
        licenses_dir=licenses_dir,
        report_section=report_section,
        inventory_section=inventory_section,
    )
    if updated_report != existing:
        third_party_path.parent.mkdir(parents=True, exist_ok=True)
        third_party_path.write_text(updated_report, encoding="utf-8")
        modified.append(third_party_path)

    modified.extend(
        _sync_dependency_license_files(
            repo_root=repo_root,
            licenses_dir_path=licenses_dir_path,
            licenses_dir=licenses_dir,
            inventory=inventory,
            existing_inventory_paths=existing_inventory_paths,
            license_source_overrides=license_source_overrides,
        )
    )

    if manage_licenses_readme:
        readme_path = _ensure_licenses_readme(
            licenses_dir_path=licenses_dir_path,
            third_party_file=third_party_rel,
        )
        if readme_path is not None:
            modified.append(readme_path)
    return modified


def _remediation_suggestion(context: CheckContext) -> str:
    """Return autofix-aware remediation guidance for dependency drift."""
    command = policy_commands_service.canonical_policy_command_invocation(
        "dependency-management",
        "refresh-all",
    )
    if context.autofix_requested:
        return (
            "Autofix is enabled for this run and may invoke "
            f"`{command}` automatically. If the violation persists, run "
            f"`{command}` manually and rerun the gate sequence."
        )
    return (
        f"Run `{command}` manually, or rerun the check path with autofix "
        "enabled so the dependency-management autofixer can invoke the same "
        "runtime action for you."
    )


def _surface_violations(
    *,
    context: CheckContext,
    changed_rel_paths: set[str],
    changed_dependency_files: Iterable[str],
    third_party_file: str,
    licenses_dir: str,
    report_heading: str,
    resolved_lock_file: str = "requirements.lock",
    direct_dependency_files: Iterable[str] | None = None,
    manage_licenses_readme: bool = True,
    license_source_overrides: (
        Mapping[str, DependencyLicenseSourceOverride] | None
    ) = None,
) -> list[Violation]:
    """Return dependency-artifact drift violations for one artifact surface."""

    try:
        third_party_path, license_dir_path = _resolve_artifact_targets(
            repo_root=context.repo_root,
            third_party_file=third_party_file,
            licenses_dir=licenses_dir,
        )
    except ValueError as error:
        return [
            Violation(
                policy_id=DependencyManagementCheck.policy_id,
                severity="error",
                message=str(error),
                can_auto_fix=False,
            )
        ]
    if not report_heading:
        return [
            Violation(
                policy_id=DependencyManagementCheck.policy_id,
                severity="error",
                message=(
                    "dependency-management metadata is missing "
                    "`report_heading`."
                ),
                can_auto_fix=False,
            )
        ]

    repo_root_resolved = context.repo_root.resolve()
    third_party_rel_text = third_party_path.relative_to(
        repo_root_resolved
    ).as_posix()
    licenses_rel_text = license_dir_path.relative_to(
        repo_root_resolved
    ).as_posix()
    normalized_changed_dependency_files = sorted(
        {
            _normalized_rel(entry)
            for entry in changed_dependency_files
            if _normalized_rel(entry)
        }
    )
    if not normalized_changed_dependency_files:
        return []

    context_payload = {
        "changed_dependency_files": normalized_changed_dependency_files,
        "third_party_file": third_party_rel_text,
        "licenses_dir": licenses_rel_text,
        "report_heading": report_heading,
        "resolved_lock_file": _normalized_rel(resolved_lock_file),
        "direct_dependency_files": [
            _normalized_rel(entry)
            for entry in (direct_dependency_files or [])
            if _normalized_rel(entry)
        ],
    }
    report_needs_refresh, licenses_dir_needs_refresh = (
        _license_artifacts_need_refresh(
            repo_root=context.repo_root,
            changed_dependency_files=normalized_changed_dependency_files,
            third_party_file=third_party_rel_text,
            licenses_dir=licenses_rel_text,
            report_heading=report_heading,
            resolved_lock_file=resolved_lock_file,
            direct_dependency_files=direct_dependency_files,
            manage_licenses_readme=manage_licenses_readme,
            license_source_overrides=license_source_overrides,
        )
    )

    violations: list[Violation] = []
    if third_party_rel_text not in changed_rel_paths and report_needs_refresh:
        violations.append(
            Violation(
                policy_id=DependencyManagementCheck.policy_id,
                severity="error",
                file_path=third_party_path,
                message=(
                    "Dependencies changed without updating "
                    f"the license table `{third_party_rel_text}`."
                ),
                suggestion=_remediation_suggestion(context),
                can_auto_fix=True,
                context={**context_payload, "issue": "third_party"},
            )
        )

    normalized_licenses_dir = _normalized_rel(licenses_rel_text).strip("/")
    license_dir_touched = any(
        rel == normalized_licenses_dir
        or rel.startswith(f"{normalized_licenses_dir}/")
        for rel in changed_rel_paths
    )
    if not license_dir_touched and licenses_dir_needs_refresh:
        violations.append(
            Violation(
                policy_id=DependencyManagementCheck.policy_id,
                severity="error",
                file_path=license_dir_path,
                message=(
                    "License files under "
                    f"{licenses_rel_text}/ must be refreshed."
                ),
                suggestion=_remediation_suggestion(context),
                can_auto_fix=True,
                context={**context_payload, "issue": "licenses_dir"},
            )
        )

    if third_party_path.is_file():
        raw_report = third_party_path.read_text(encoding="utf-8")
        report = _extract_license_report(raw_report, report_heading)
    else:
        report = ""

    if not report:
        violations.append(
            Violation(
                policy_id=DependencyManagementCheck.policy_id,
                severity="error",
                file_path=third_party_path,
                message=(
                    f"Add a '{report_heading}' section to "
                    f"`{third_party_rel_text}` that chronicles dependency "
                    "updates."
                ),
                suggestion=_remediation_suggestion(context),
                can_auto_fix=True,
                context={**context_payload, "issue": "missing_report"},
            )
        )
    else:
        missing_references = [
            dep_file
            for dep_file in normalized_changed_dependency_files
            if not _contains_reference(report, dep_file)
        ]
        if missing_references:
            violations.append(
                Violation(
                    policy_id=DependencyManagementCheck.policy_id,
                    severity="error",
                    file_path=third_party_path,
                    message=(
                        "Dependency report is missing changed files: "
                        + ", ".join(missing_references)
                    ),
                    suggestion=_remediation_suggestion(context),
                    can_auto_fix=True,
                    context={
                        **context_payload,
                        "issue": "missing_references",
                        "missing_references": missing_references,
                    },
                )
            )
    return violations


def _surface_vulnerability_violations(
    *,
    context: CheckContext,
    surface: DependencySurface,
    findings: Sequence[object],
) -> list[Violation]:
    """Translate surface vulnerability findings into policy violations."""

    violations: list[Violation] = []
    lock_path = context.repo_root / surface.lock_file
    for finding in findings:
        package_name = str(getattr(finding, "package_name", "")).strip()
        version = str(getattr(finding, "version", "")).strip()
        vulnerability_id = str(
            getattr(finding, "vulnerability_id", "")
        ).strip()
        fix_versions = [
            str(entry).strip()
            for entry in getattr(finding, "fix_versions", ())
            if str(entry).strip()
        ]
        target_ids = [
            str(entry).strip()
            for entry in getattr(finding, "target_ids", ())
            if str(entry).strip()
        ]
        scope_suffix = ""
        if target_ids:
            scope_suffix = (
                " for targets `" + "`, `".join(sorted(target_ids)) + "`"
            )
        if fix_versions:
            listed_versions = "`, `".join(sorted(fix_versions))
            remediation = (
                " Refresh the owning dependency surface so "
                f"`{surface.lock_file}` reaches `{listed_versions}` or newer."
            )
        else:
            remediation = (
                " Refresh the owning dependency surface after an upstream "
                "fix becomes available."
            )
        violations.append(
            Violation(
                policy_id="dependency-management",
                severity="error",
                file_path=lock_path,
                message=(
                    f"Dependency surface `{surface.surface_id}` locks "
                    f"`{package_name}=={version}` with vulnerability "
                    f"`{vulnerability_id}`{scope_suffix}.{remediation}"
                ),
                can_auto_fix=bool(fix_versions),
                context={
                    "changed_dependency_files": [surface.lock_file],
                    "issue": "vulnerability_audit",
                    "surface_id": surface.surface_id,
                    "lock_file": surface.lock_file,
                    "package_name": package_name,
                    "version": version,
                    "vulnerability_id": vulnerability_id,
                    "fix_versions": sorted(fix_versions),
                    "target_ids": sorted(target_ids),
                },
            )
        )
    return violations


class DependencyManagementCheck(PolicyCheck):
    """Ensure dependency changes update lock and compliance artifacts."""

    policy_id = "dependency-management"
    version = "1.0.0"

    def run_runtime_action(
        self,
        action: str,
        *,
        repo_root: Path,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Run declared dependency-management runtime actions."""
        if action != RUNTIME_ACTION_REFRESH_ALL:
            raise ValueError(
                "Unsupported dependency-management runtime action: "
                f"`{action}`."
            )
        from devcovenant.builtin.policies.dependency_management import (
            dependency_lock_runtime,
        )

        return dependency_lock_runtime.refresh_all(
            repo_root,
            payload=payload,
        )

    def check(self, context: CheckContext):
        """Verify dependency changes match the recorded license summary."""
        files = context.changed_files or []

        try:
            license_source_overrides = resolve_license_source_overrides(
                self.get_option("license_source_overrides", [])
            )
            surfaces = resolve_dependency_surfaces(
                repo_root=context.repo_root,
                raw_surfaces=self.get_option("surfaces", []),
                include_inactive=False,
            )
        except ValueError as error:
            return [
                Violation(
                    policy_id=self.policy_id,
                    severity="error",
                    message=str(error),
                    can_auto_fix=False,
                )
            ]
        if not surfaces:
            return []

        changed_rel_paths: set[str] = set()
        for path in files:
            rel_path = _relative_posix(path, context.repo_root)
            if rel_path is None:
                continue
            changed_rel_paths.add(rel_path)

        violations: list[Violation] = []
        from devcovenant.builtin.policies.dependency_management import (
            dependency_lock_runtime,
        )

        audit_cache = context.runtime_cache_bucket(
            "dependency_management_audit"
        )
        for surface in surfaces:
            if surface.audit_service:
                try:
                    findings = (
                        dependency_lock_runtime.audit_surface_vulnerabilities(
                            context.repo_root,
                            surface=surface,
                            runtime_cache=audit_cache,
                        )
                    )
                except RuntimeError as error:
                    violations.append(
                        Violation(
                            policy_id=self.policy_id,
                            severity="error",
                            file_path=context.repo_root / surface.lock_file,
                            message=str(error),
                            can_auto_fix=False,
                        )
                    )
                else:
                    violations.extend(
                        _surface_vulnerability_violations(
                            context=context,
                            surface=surface,
                            findings=findings,
                        )
                    )
            if not surface.direct_dependency_files:
                continue
            surface_changed_files = sorted(
                rel_path
                for rel_path in changed_rel_paths
                if dependency_surface_matches(surface, rel_path)
            )
            if not surface_changed_files:
                continue
            violations.extend(
                _surface_violations(
                    context=context,
                    changed_rel_paths=changed_rel_paths,
                    changed_dependency_files=surface_changed_files,
                    third_party_file=surface.third_party_file,
                    licenses_dir=surface.licenses_dir,
                    report_heading=surface.report_heading,
                    resolved_lock_file=surface.lock_file,
                    direct_dependency_files=surface.direct_dependency_files,
                    manage_licenses_readme=surface.manage_licenses_readme,
                    license_source_overrides=license_source_overrides,
                )
            )
        return violations
