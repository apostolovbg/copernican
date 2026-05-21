"""Full refresh orchestration for DevCovenant repositories."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess  # nosec B404
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

import devcovenant.core.agents_blocks as agents_blocks_lib
import devcovenant.core.asset_materialization as asset_materialization_service
import devcovenant.core.managed_docs as managed_docs_service
import devcovenant.core.policy_metadata as metadata_runtime
import devcovenant.core.policy_runtime_actions as runtime_actions_module
import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.project_governance as project_governance_service
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.repository_validation as manifest_module
from devcovenant.core.execution import (
    merge_active_run_phase_timings,
    print_step,
    runtime_print,
)
from devcovenant.core.policy_contract import CheckContext
from devcovenant.core.policy_metadata import PolicyDefinition
from devcovenant.core.policy_registry import (
    PolicyRegistry,
    iter_script_locations,
    load_policy_descriptor,
    resolve_script_location,
)
from devcovenant.core.repository_paths import display_path
from devcovenant.core.tracked_registry import policy_registry_path

ProjectGovernanceState = project_governance_service.ProjectGovernanceState

USER_GITIGNORE_BEGIN = "# --- User entries (preserved) ---"
USER_GITIGNORE_END = "# --- End user entries ---"


def _utc_today() -> str:
    """Return current UTC date."""
    return datetime.now(timezone.utc).date().isoformat()


def _read_devcovenant_version(repo_root: Path) -> str:
    """Read the DevCovenant package version from devcovenant/VERSION."""
    version_path = repo_root / "devcovenant" / "VERSION"
    if not version_path.exists():
        return "0.0.0"
    version_text = version_path.read_text(encoding="utf-8").strip()
    return version_text or "0.0.0"


def _metadata_string_token(raw: object) -> str:
    """Normalize one metadata value into a single string token."""
    if isinstance(raw, list):
        for entry in raw:
            token = str(entry).strip()
            if token:
                return token
        return ""
    return str(raw or "").strip()


def _install_import_managed_docs(config: dict[str, object]) -> set[str]:
    """Return install-recorded managed-doc import seeds from config."""
    return managed_docs_service.install_import_managed_docs(config)


def _project_version_file_from_config(config: dict[str, object]) -> str:
    """Resolve version-sync.version_file from effective config layers."""
    metadata_layers = (
        config.get("autogen_metadata_overlays"),
        config.get("user_metadata_overlays"),
        config.get("autogen_metadata_overrides"),
        config.get("user_metadata_overrides"),
    )
    resolved = ""
    for layer in metadata_layers:
        if not isinstance(layer, dict):
            continue
        version_sync = layer.get("version-sync")
        if not isinstance(version_sync, dict):
            continue
        token = _metadata_string_token(version_sync.get("version_file"))
        if token:
            resolved = token
    return resolved or "VERSION"


def _read_project_version(
    repo_root: Path,
    config: dict[str, object],
    *,
    required: bool = True,
) -> str:
    """Read the project version using version-sync.version_file."""
    version_file = _project_version_file_from_config(config)
    version_path = _resolve_path_under_root(
        repo_root,
        version_file,
        field_name="version-sync.version_file",
    )
    if not version_path.exists():
        if required:
            raise ValueError(
                f"Missing declared project version file: {version_file}"
            )
        return ""
    try:
        version_text = version_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if required:
            raise ValueError(
                f"Unable to read declared project version file "
                f"{version_file}: {exc}"
            ) from exc
        return ""
    if version_text:
        return version_text
    if required:
        raise ValueError(
            f"Declared project version file is empty: {version_file}"
        )
    return ""


def _read_yaml(path: Path) -> dict[str, object]:
    """Load YAML mapping payload from disk."""
    rendered = display_path(path)
    if not path.exists():
        raise ValueError(f"Missing YAML file: {rendered}")
    try:
        payload = yaml_cache_service.load_yaml(path)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {rendered}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read {rendered}: {exc}") from exc
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"YAML file must contain a mapping: {rendered}")


def _managed_docs_from_config(config: dict[str, object]) -> list[str]:
    """Resolve autogen managed docs from config doc_assets."""
    return managed_docs_service.managed_docs_from_config(config)


def _normalize_doc_name(name: str) -> str:
    """Normalize configured doc names to canonical markdown paths."""
    return managed_docs_service.normalize_doc_name(name)


def _sync_doc(
    repo_root: Path,
    doc_name: str,
    *,
    config_payload: dict[str, object],
    project_version: str,
    devcovenant_version: str,
    project_governance_state: ProjectGovernanceState,
    import_managed_docs: set[str],
) -> bool:
    """Synchronize one managed doc from descriptor content."""
    return managed_docs_service.sync_doc(
        repo_root,
        doc_name,
        config_payload=config_payload,
        project_version=project_version,
        devcovenant_version=devcovenant_version,
        project_governance_state=project_governance_state,
        import_managed_docs=import_managed_docs,
    )


def _active_profiles(config: dict[str, object]) -> list[str]:
    """Resolve active profiles from config, always including global."""
    return profile_registry_service.parse_active_profiles(
        config, include_global=True
    )


def _profile_asset_target(
    repo_root: Path, asset_payload: dict[str, object]
) -> Path | None:
    """Return normalized target path for a profile asset entry."""
    raw_path = str(asset_payload.get("path", "")).strip()
    if not raw_path:
        return None
    return _resolve_path_under_root(
        repo_root,
        raw_path,
        field_name="profile asset target",
    )


def _profile_asset_template(
    repo_root: Path,
    profile_payload: dict[str, object],
    asset_payload: dict[str, object],
) -> Path | None:
    """Return the resolved template path for a profile asset entry."""
    raw_template = str(asset_payload.get("template", "")).strip()
    profile_path = str(profile_payload.get("path", "")).strip()
    if not raw_template or not profile_path:
        return None
    profile_name = str(profile_payload.get("profile", "")).strip() or (
        profile_path
    )
    profile_root = _resolve_path_under_root(
        repo_root,
        profile_path,
        field_name=f"profile root ({profile_name})",
    )
    assets_root = (profile_root / "assets").resolve()
    return _resolve_path_under_root(
        assets_root,
        raw_template,
        field_name=f"profile asset template ({profile_name})",
    )


def _resolve_path_under_root(
    root: Path,
    raw_path: str,
    *,
    field_name: str,
) -> Path:
    """Resolve a relative path and enforce it stays under a root."""
    token = str(raw_path or "").strip()
    if not token:
        raise ValueError(f"{field_name} path cannot be empty.")
    relative_path = Path(token)
    if relative_path.is_absolute():
        raise ValueError(f"{field_name} path must be relative, got '{token}'.")
    root_path = Path(os.path.realpath(root))
    resolved = Path(os.path.realpath(root_path / relative_path))
    common_path = os.path.commonpath([str(root_path), str(resolved)])
    if common_path != str(root_path):
        raise ValueError(
            f"{field_name} path escapes '{root_path}': '{token}'."
        )
    return resolved


def _read_text_if_exists(path: Path) -> str:
    """Read UTF-8 text when file exists, otherwise return empty string."""
    if not path.exists() or not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")


def _write_text_if_changed(target: Path, content: str) -> bool:
    """Write target file only when content changes."""
    current = _read_text_if_exists(target)
    if current == content:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return True


def _record_phase_timing(
    phase_timings: list[dict[str, object]],
    phase_name: str,
    started_at: float,
    *,
    changed: bool | None = None,
    skipped: bool | None = None,
) -> None:
    """Append one normalized phase timing row."""
    entry: dict[str, object] = {
        "phase": phase_name,
        "duration_seconds": round(time.perf_counter() - started_at, 6),
    }
    if changed is not None:
        entry["changed"] = changed
    if skipped is not None:
        entry["skipped"] = skipped
    phase_timings.append(entry)


def _materialize_profile_asset(
    *,
    target: Path,
    template_path: Path | None,
    project_governance_state: ProjectGovernanceState,
) -> bool:
    """Apply one profile asset entry and return True when modified."""
    if template_path is None or not template_path.exists():
        return False

    if target.exists():
        return False

    template_text = template_path.read_text(encoding="utf-8")
    template_text = (
        asset_materialization_service.render_profile_asset_template_text(
            template_text,
            project_governance_state,
        )
    )
    return _write_text_if_changed(target, template_text)


def _refresh_profile_assets(
    repo_root: Path,
    profile_registry: dict[str, dict],
    active_profiles: list[str],
    project_governance_state: ProjectGovernanceState,
) -> list[str]:
    """Materialize active profile assets into the target repository."""
    changed: list[str] = []
    profiles_map = _profile_registry_profiles(profile_registry)
    for profile_name in active_profiles:
        normalized = str(profile_name or "").strip().lower()
        if not normalized:
            continue
        profile_payload = profiles_map.get(normalized, {})
        raw_assets = profile_payload.get("assets")
        if not isinstance(raw_assets, list):
            continue
        for entry in raw_assets:
            if not isinstance(entry, dict):
                continue
            target = _profile_asset_target(repo_root, entry)
            if target is None:
                continue
            template_path = _profile_asset_template(
                repo_root, profile_payload, entry
            )
            if not _materialize_profile_asset(
                target=target,
                template_path=template_path,
                project_governance_state=project_governance_state,
            ):
                continue
            rel_path = _repo_relative_path(repo_root, target)
            changed.append(rel_path)
    return changed


_PROJECT_TOML_SECTION_RE = re.compile(
    r"(?ms)^(?P<header>\[project\][ \t]*\n)(?P<body>.*?)(?=^\[|\Z)"
)


def _replace_or_append_project_toml_field(
    section_body: str,
    *,
    field_name: str,
    toml_value: str,
) -> str:
    """Replace or append one field in the `[project]` TOML section."""
    new_line = f"{field_name} = {toml_value}\n"
    field_re = re.compile(
        rf"(?ms)^{re.escape(field_name)}\s*=.*?" r"(?=^[A-Za-z0-9_.-]+\s*=|\Z)"
    )
    if field_re.search(section_body):
        return field_re.sub(new_line, section_body, count=1)
    separator = "" if not section_body or section_body.endswith("\n") else "\n"
    return f"{section_body}{separator}{new_line}"


def _sync_project_pyproject_identity(
    repo_root: Path,
    project_governance_state: ProjectGovernanceState,
) -> bool:
    """Synchronize package identity fields from project-governance."""
    pyproject_path = repo_root / "pyproject.toml"
    current = _read_text_if_exists(pyproject_path)
    if not current:
        return False
    match = _PROJECT_TOML_SECTION_RE.search(current)
    if match is None:
        return False
    updated_body = _replace_or_append_project_toml_field(
        match.group("body"),
        field_name="name",
        toml_value=json.dumps(project_governance_state.project_name),
    )
    updated_body = _replace_or_append_project_toml_field(
        updated_body,
        field_name="description",
        toml_value=project_governance_service.render_toml_string(
            project_governance_state.project_description
        ),
    )
    updated = (
        current[: match.start("body")]
        + updated_body
        + current[match.end("body") :]
    )
    return _write_text_if_changed(pyproject_path, updated)


def _repo_relative_path(repo_root: Path, target: Path) -> str:
    """Return a target path relative to the repository root across symlink
    aliases."""
    root_path = Path(os.path.realpath(repo_root))
    target_path = Path(os.path.realpath(target))
    relative = os.path.relpath(str(target_path), str(root_path))
    return Path(relative).as_posix()


_CONFIG_AUTOGEN_PATHS: tuple[tuple[str, ...], ...] = (
    ("profiles", "generated", "devcov_core_paths"),
    ("autogen_metadata_overlays",),
    ("autogen_metadata_overrides",),
    ("profiles", "generated"),
)


def _is_autogen_config_path(path: tuple[str, ...]) -> bool:
    """Return True when a config path is owned by autogen refresh."""
    for prefix in _CONFIG_AUTOGEN_PATHS:
        if path[: len(prefix)] == prefix:
            return True
    return False


def _merge_user_config_values(
    base: dict[str, object],
    incoming: dict[str, object],
    *,
    path: tuple[str, ...] = (),
) -> None:
    """Merge user-owned config values while skipping autogen-owned paths."""
    for raw_key, incoming_value in incoming.items():
        key = str(raw_key)
        next_path = path + (key,)
        if _is_autogen_config_path(next_path):
            continue
        current_value = base.get(key)
        if isinstance(current_value, dict) and isinstance(
            incoming_value, dict
        ):
            _merge_user_config_values(
                current_value,
                incoming_value,
                path=next_path,
            )
            continue
        base[key] = copy.deepcopy(incoming_value)


def _config_template_path(repo_root: Path) -> Path:
    """Return global config template path."""
    return (
        repo_root
        / "devcovenant"
        / "builtin"
        / "profiles"
        / "global"
        / "assets"
        / "config.yaml"
    )


def _load_config_template(repo_root: Path) -> dict[str, object]:
    """Load global config template payload."""
    template_payload = _read_yaml(_config_template_path(repo_root))
    if template_payload:
        return template_payload
    raise ValueError(
        "Global config template is empty: "
        f"{_config_template_path(repo_root)}"
    )


def _yaml_block(payload: dict[str, object]) -> str:
    """Dump one YAML block while preserving key order."""
    return yaml.safe_dump(payload, sort_keys=False).rstrip()


def _config_comment_header() -> str:
    """Return static comment header used by rendered config."""
    rule = "# " + ("-" * 67)
    return "\n".join(
        [
            rule,
            "# DevCovenant config template (review-required install baseline)",
            rule,
            (
                "# This file is copied to `devcovenant/config.yaml` by "
                "`devcovenant install`."
            ),
            "#",
            "# Read this file as the repository's operating contract:",
            "# - human-owned keys say what this repository wants;",
            "# - refresh-owned keys show what DevCovenant resolved/generated.",
            "#",
            "# Why install stops before deploy:",
            "# - install gives the repo a safe starting point;",
            "# - deploy turns the config into active managed docs,",
            "#   registries,",
            "#   and generated governance files;",
            "# - `install.config_reviewed: false` keeps deploy blocked until",
            "#   a",
            "#   human has checked that the starting config matches the repo.",
            "#",
            "# Typical first-time flow:",
            "# 1) run `devcovenant install`",
            "# 2) review this file",
            "# 3) set `install.config_reviewed: true`",
            "# 4) run `devcovenant deploy`",
            "# 5) prepare the repository's declared environment",
            "# 6) start normal work with",
            "#    `gate --open` -> `gate --verify` -> `run` -> `gate --close`",
            rule,
        ]
    )


def _config_section_header(title: str) -> str:
    """Return one titled section banner for rendered config blocks."""
    rule = "# " + ("-" * 67)
    return "\n".join([rule, f"# {title}", rule])


def _render_config_yaml(payload: dict[str, object]) -> str:
    """Render config payload with stable comments and key ordering."""
    known_keys = [
        "developer_mode",
        "profiles",
        "paths",
        "doc_assets",
        "install",
        "engine",
        "clean",
        "ci_and_test",
        "pre_commit",
        "integrity",
        "workflow",
        "project-governance",
        "policy_state",
        "ignore",
        "gitignore",
        "autogen_metadata_overlays",
        "user_metadata_overlays",
        "autogen_metadata_overrides",
        "user_metadata_overrides",
    ]
    comments = {
        "scope": _config_section_header("Scope control"),
        "profiles": _config_section_header("Profile activation"),
        "paths": _config_section_header("Canonical paths"),
        "doc_assets": _config_section_header("Managed document controls"),
        "install": _config_section_header("Install/deploy safety"),
        "engine": _config_section_header("Engine behavior"),
        "clean": _config_section_header("Cleanup targets"),
        "ci_and_test": _config_section_header(
            "CI-and-test workflow generation"
        ),
        "pre_commit": _config_section_header("Pre-commit generation"),
        "integrity": _config_section_header("Integrity runtime contract"),
        "workflow": _config_section_header("Workflow runtime contract"),
        "project_governance": _config_section_header("Project governance"),
        "policy": _config_section_header(
            "Policy activation and customization"
        ),
        "ignore": _config_section_header("Global ignore patterns"),
        "gitignore": _config_section_header("Gitignore generation"),
        "metadata": _config_section_header(
            "Metadata layers (resolution order matters)"
        ),
    }

    blocks = [
        _config_comment_header(),
        comments["scope"],
        "# Human-owned section.",
        "\n".join(
            [
                (
                    "# Whether this repository is in DevCovenant developer "
                    "mode."
                ),
                (
                    "# - false: normal user-repository scope (exclude "
                    "DevCovenant implementation internals from ordinary "
                    "repository scope)"
                ),
                (
                    "# - true: DevCovenant developer mode "
                    "(govern the full DevCovenant implementation tree)"
                ),
            ]
        ),
        _yaml_block(
            {
                "developer_mode": bool(payload.get("developer_mode", False)),
            }
        ),
        comments["profiles"],
        "# Mixed-ownership section.",
        "# `profiles.active` is human-owned.",
        "# `profiles.generated` is refresh-owned diagnostic state.",
        "# Ordered profile list. `global` should stay first.",
        (
            "# Add `github` when the repository wants a generated GitHub "
            "Actions workflow."
        ),
        (
            "# Add `python_venv` only when the repository explicitly opts "
            "into the builtin repo-local `.venv` contract."
        ),
        (
            "# Keep inherited values inherited. Add repo-specific behavior in "
            "a custom profile instead of recopied overlays."
        ),
        ('# Here, "inherited" means values from other active profiles.'),
        (
            "# A same-name custom profile is loaded and the builtin profile "
            "is ignored."
        ),
        (
            "# If the repository needs a starting custom profile, copy "
            "`devcovenant/builtin/profiles/userproject/` to "
            "`devcovenant/custom/profiles/userproject/` and edit it there."
        ),
        (
            "# Profiles contribute suffixes, assets, metadata overlays, "
            "and cleanup overlays."
        ),
        "# Generated diagnostics include profile-level core path mappings.",
        (
            "# Profiles do not activate policies. "
            "Policy activation is `policy_state` for normal toggles."
        ),
        (
            "# `severity: critical` policies remain enforced even when "
            "toggled false in `policy_state`."
        ),
        (
            "# Keep GitHub-only CI extensions in a separate optional "
            "GitHub-specific custom profile instead of the repo-identity "
            "profile."
        ),
        _yaml_block({"profiles": payload.get("profiles", {})}),
        comments["paths"],
        "# Human-owned section.",
        "# Every key in `paths` is human-owned.",
        "# Runtime policy source parsed by the engine.",
        "# Generated local policy registry (hashes + diagnostics).",
        (
            "# Runtime evidence files such as gate-status "
            "and workflow-session live here too."
        ),
        _yaml_block({"paths": payload.get("paths", {})}),
        comments["doc_assets"],
        "# Mixed-ownership section.",
        "# `doc_assets.autogen` is human-owned enabled-doc selection.",
        "# `doc_assets.user` is human-owned exclusion state.",
        "\n".join(
            [
                (
                    "# `autogen` entries are managed-doc target paths, not a "
                    "builtin optional-doc class."
                ),
                (
                    "# Available descriptors come from the global asset root "
                    "plus active profile asset roots."
                ),
                (
                    "# Later active profiles override earlier profiles for "
                    "the same target path."
                ),
                (
                    "# Documents in `user` are excluded after `autogen` "
                    "selection."
                ),
                (
                    "# Managed block refresh still applies when markers are "
                    "present."
                ),
            ]
        ),
        _yaml_block({"doc_assets": payload.get("doc_assets", {})}),
        comments["install"],
        "# Mixed-ownership section.",
        "# `install.config_reviewed` is human-owned.",
        "# `install.import_managed_docs` is refresh-owned memory.",
        (
            "# `install` seeds `config_reviewed: false`. Deploy stays blocked "
            "until a human finishes the config review."
        ),
        ("# Set this to true after review to allow `devcovenant deploy`."),
        _yaml_block({"install": payload.get("install", {})}),
        comments["engine"],
        "# Human-owned section.",
        "# Violations at or above fail_threshold fail the run.",
        "# Allowed levels: info, warning, error, critical.",
        "# auto_fix_enabled controls gate-managed policy auto-fix behavior.",
        "# `check` stays read-only.",
        "# logs_keep_last controls run-log retention (`0` keeps all runs).",
        (
            "# pycache_prefix_enabled/pycache_prefix route DevCovenant-"
            "managed Python bytecode caches away from the repo tree "
            "(empty prefix = auto temp path)."
        ),
        "# file_suffixes and ignore_dirs define broad scan boundaries.",
        _yaml_block({"engine": payload.get("engine", {})}),
        comments["clean"],
        "# Human-owned section.",
        (
            "# Additive cleanup targets merged after active profile "
            "clean_overlays."
        ),
        (
            "# Use this for repository-specific build/cache/runtime-registry/"
            "log junk beyond builtin profiles."
        ),
        "# Protected entries are additive safety fences.",
        (
            "# Runtime also protects engine-critical paths such as `.git`, "
            "the tracked registry docs, the logs README, the active clean run "
            "dir, and the managed environment roots resolved from "
            "managed-environment metadata."
        ),
        _yaml_block({"clean": payload.get("clean", {})}),
        comments["ci_and_test"],
        "# Human-owned section.",
        (
            "# Deep-merge patch applied to generated "
            "`.github/workflows/ci.yml`."
        ),
        (
            "# Activate the `github` profile to generate the standard GitHub "
            "Actions CI workflow."
        ),
        (
            "# The builtin github base bootstraps DevCovenant itself from "
            "the shipped `devcovenant/runtime-requirements.lock`, not from "
            "project dependency files."
        ),
        (
            "# Active profiles may also contribute reusable ci_and_test "
            "fragments."
        ),
        "# Keep the github-owned base workflow generic.",
        "# Use config overlays only for repository-local CI adjustments.",
        "# Full replacement payload for generated CI-and-test workflow.",
        (
            "# Use full overrides only when the repository deliberately "
            "takes complete ownership."
        ),
        _yaml_block(
            {
                "ci_and_test": payload.get(
                    "ci_and_test",
                    {},
                )
            }
        ),
        comments["pre_commit"],
        "# Human-owned section.",
        "# `overlays` are merged into generated pre-commit config.",
        "# `overrides` replace generated payload when non-empty.",
        _yaml_block({"pre_commit": payload.get("pre_commit", {})}),
        comments["integrity"],
        "# Human-owned section.",
        (
            "# Integrity watches are optional runtime knobs for the built-in "
            "descriptor/registry/gate checks."
        ),
        (
            "# Use them when specific directories or files should require a "
            "fresh gate-status update."
        ),
        _yaml_block({"integrity": payload.get("integrity", {})}),
        comments["workflow"],
        "# Human-owned section.",
        (
            "# Workflow contract settings such as the canonical pre-commit "
            "command live here."
        ),
        (
            "# These are runtime contract settings, not policy toggles, so "
            "they do not belong under `policy_state`."
        ),
        _yaml_block({"workflow": payload.get("workflow", {})}),
        comments["project_governance"],
        "# Human-owned section.",
        (
            "# Repository lifecycle metadata rendered into managed docs, "
            "registry output, and changelog release flow."
        ),
        (
            "# `project_name` is the canonical public/project identity "
            "string."
        ),
        (
            "# DevCovenant derives normalized path tokens such as "
            "`{{ PROJECT_NAME_PATH }}` where package-safe paths need them."
        ),
        (
            "# Keep distribution/project identity in `project_name`; do not "
            "force Python import-package spelling there just to satisfy path "
            "syntax."
        ),
        (
            "# `project_name` and `project_description` are non-empty "
            "identity strings."
        ),
        (
            "# `stage` must be one of `allowed_stages`; the default set is "
            "prototype, alpha, beta, stable, deprecated, archived."
        ),
        (
            "# `maintenance_stance` must be one of "
            "`allowed_maintenance_stances`; the default set is "
            "active, maintenance, frozen, sunset."
        ),
        (
            "# `compatibility_policy` must be "
            "`backward-compatible`, `breaking-allowed`, "
            "`forward-only`, or `unspecified`."
        ),
        (
            "# Use `compatibility_policy` for compatibility promises, not "
            "for feature notes such as cross-platform support."
        ),
        (
            "# `backward-compatible` preserves the public contract; "
            "`breaking-allowed` makes compatibility optional; "
            "`forward-only` rejects legacy compatibility fallbacks."
        ),
        "# `versioning_mode` must be `versioned` or `unversioned`.",
        (
            "# `codename` and `build_identity` are optional free-form "
            "strings."
        ),
        (
            "# `versioning_mode: unversioned` renders the configured "
            "`unversioned_label` and uses `unreleased_heading`."
        ),
        (
            "# `unversioned_label`, `unreleased_heading`, and "
            "`changelog_file` are free-form strings; `changelog_file` is "
            "a repo-relative path."
        ),
        (
            "# When unversioned, the top visible changelog heading must "
            "match `unreleased_heading` exactly."
        ),
        _yaml_block(
            {
                "project-governance": payload.get(
                    "project-governance",
                    {},
                )
            }
        ),
        comments["policy"],
        "# Human-owned section.",
        "# Human-owned booleans in a refresh-synchronized map.",
        ("# Canonical policy activation map: {policy-id: true|false}."),
        (
            "# Critical-severity policies remain enforced even when set "
            "to false here."
        ),
        (
            "# The shipped install baseline keeps `managed-environment` "
            "false until the repository enables a matching profile such as "
            "`python_venv`."
        ),
        _yaml_block({"policy_state": payload.get("policy_state", {})}),
        comments["ignore"],
        "# Human-owned section.",
        "# Extra glob patterns excluded from CheckContext file collections.",
        _yaml_block({"ignore": payload.get("ignore", {})}),
        comments["gitignore"],
        "# Human-owned section.",
        "# Extra entries appended to generated `.gitignore`.",
        "# Entries are applied before the preserved user block.",
        "# `overrides` replaces generated base/profile/os fragments entirely.",
        _yaml_block({"gitignore": payload.get("gitignore", {})}),
        comments["metadata"],
        "# Mixed-ownership section.",
        "# `autogen_*` blocks below are refresh-owned.",
        "# `user_*` blocks below are human-owned.",
        "# Auto-generated metadata overlays written by refresh.",
        "# Overlay semantics: scalar values replace prior scalars.",
        "# Plain scalar lists append with de-duplication.",
        "# Mapping values merge by subkey.",
        "# Lists of mappings with stable `id` fields merge by `id`.",
        "# Keep one metadata key in one shape across layers.",
        _yaml_block(
            {
                "autogen_metadata_overlays": payload.get(
                    "autogen_metadata_overlays", {}
                )
            }
        ),
        "# Human-owned subsection.",
        "# User-owned overlays applied after autogen overlays.",
        "# Use the same metadata-key shape here as the inherited layer.",
        _yaml_block(
            {
                "user_metadata_overlays": payload.get(
                    "user_metadata_overlays", {}
                )
            }
        ),
        "# Refresh-owned subsection.",
        "# Auto-generated metadata overrides written by refresh.",
        "# Override semantics: full key replacement.",
        "# Do not hand-edit unless you intentionally own this layer.",
        _yaml_block(
            {
                "autogen_metadata_overrides": payload.get(
                    "autogen_metadata_overrides", {}
                )
            }
        ),
        "# Human-owned subsection.",
        "# User-owned overrides applied last (highest precedence).",
        "# Override semantics: full key replacement.",
        "# Shape: {policy-id: {metadata_key: scalar|list|mapping}}",
        "# Do not replace a mapping with ad-hoc sibling flat keys.",
        _yaml_block(
            {
                "user_metadata_overrides": payload.get(
                    "user_metadata_overrides", {}
                )
            }
        ),
    ]

    extras = {
        key: value for key, value in payload.items() if key not in known_keys
    }
    if extras:
        rule = "# " + ("-" * 67)
        blocks.extend(
            [
                rule,
                "# Extra user-defined keys (preserved)",
                rule,
                _yaml_block(extras),
            ]
        )
    return "\n\n".join(blocks).rstrip() + "\n"


def render_config_yaml(payload: dict[str, object]) -> str:
    """Render one config payload with the standard comment contract."""

    return _render_config_yaml(payload)


def render_review_required_config_yaml(
    repo_root: Path,
    *,
    import_managed_docs: list[str] | None = None,
) -> str:
    """Render the review-required install config for one repository."""

    payload = _load_config_template(repo_root)
    install_block = payload.get("install", {})
    if not isinstance(install_block, dict):
        install_block = {}
    install_block["import_managed_docs"] = list(import_managed_docs or [])
    payload["install"] = install_block
    return render_config_yaml(payload)


def _refresh_config_generated(
    repo_root: Path,
    config_path: Path,
    config: dict[str, object],
    user_config: dict[str, object],
    registry: dict[str, dict],
    active_profiles: list[str],
) -> tuple[dict[str, object], bool]:
    """Refresh config with autogen values while preserving user-owned keys."""
    template = _load_config_template(repo_root)
    merged = copy.deepcopy(template)
    _merge_user_config_values(merged, config)
    _apply_profile_aware_engine_defaults(merged, user_config, active_profiles)
    _apply_profile_aware_project_governance_defaults(
        merged,
        user_config,
        registry,
        active_profiles,
    )

    profile_suffixes = profile_registry_service.resolve_profile_suffixes(
        registry, active_profiles
    )
    suffixes = sorted({str(item) for item in profile_suffixes if str(item)})

    profiles_block = merged.get("profiles")
    if not isinstance(profiles_block, dict):
        profiles_block = {}
    generated = profiles_block.get("generated")
    if not isinstance(generated, dict):
        generated = {}
    profiles_block["active"] = list(active_profiles)
    generated["file_suffixes"] = suffixes
    generated["devcov_core_paths"] = _default_core_paths(repo_root)
    profiles_block["generated"] = generated
    merged["profiles"] = profiles_block

    merged.pop("devcov_core_paths", None)
    merged.pop("version", None)
    merged.pop("docs", None)
    merged["autogen_metadata_overlays"] = _config_autogen_metadata_overlays(
        repo_root,
        active_profiles,
        profile_registry=registry,
    )
    merged["autogen_metadata_overrides"] = _config_autogen_metadata_overrides()
    merged["policy_state"] = _materialize_policy_state_map(
        repo_root,
        metadata_runtime.normalize_policy_state(merged.get("policy_state")),
    )

    doc_assets = merged.get("doc_assets")
    if not isinstance(doc_assets, dict):
        doc_assets = {}
    raw_autogen = doc_assets.get("autogen")
    if isinstance(raw_autogen, list):
        autogen = [_normalize_doc_name(item) for item in raw_autogen]
        doc_assets["autogen"] = [doc for doc in autogen if doc]
    else:
        doc_assets["autogen"] = []

    raw_user = doc_assets.get("user")
    if isinstance(raw_user, list):
        user_docs = [_normalize_doc_name(item) for item in raw_user]
        doc_assets["user"] = [doc for doc in user_docs if doc]
    else:
        doc_assets["user"] = []
    merged["doc_assets"] = doc_assets

    rendered = _render_config_yaml(merged)
    current = _read_text_if_exists(config_path)
    if current == rendered:
        return merged, False
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(rendered, encoding="utf-8")
    return merged, True


def _apply_profile_aware_engine_defaults(
    merged: dict[str, object],
    user_config: dict[str, object],
    active_profiles: list[str],
) -> None:
    """Apply profile-aware config defaults when the user left keys unset."""
    user_engine = user_config.get("engine")
    user_engine_map = user_engine if isinstance(user_engine, dict) else {}

    engine_block = merged.get("engine")
    if not isinstance(engine_block, dict):
        engine_block = {}
        merged["engine"] = engine_block

    if bool(merged.get("developer_mode", False)):
        if "auto_fix_enabled" not in user_engine_map:
            engine_block["auto_fix_enabled"] = True
        if "pycache_prefix_enabled" not in user_engine_map:
            engine_block["pycache_prefix_enabled"] = True


def _normalize_project_governance_mapping(
    raw_value: object,
) -> dict[str, object]:
    """Normalize one project-governance config block into a mapping."""
    if not isinstance(raw_value, dict):
        return {}
    return {
        str(key).strip(): copy.deepcopy(value)
        for key, value in raw_value.items()
        if str(key).strip()
    }


def _profile_project_governance_defaults(
    profile_registry: dict[str, dict],
    active_profiles: list[str],
) -> dict[str, object]:
    """Return merged project-governance defaults from active profiles."""
    profiles_map = _profile_registry_profiles(profile_registry)
    resolved: dict[str, object] = {}
    for profile_name in active_profiles:
        normalized_name = str(profile_name or "").strip().lower()
        if not normalized_name:
            continue
        payload = profiles_map.get(normalized_name, {})
        fragment = _normalize_project_governance_mapping(
            payload.get("project-governance")
        )
        if not fragment:
            continue
        resolved.update(fragment)
    return resolved


def _apply_profile_aware_project_governance_defaults(
    merged: dict[str, object],
    user_config: dict[str, object],
    profile_registry: dict[str, dict],
    active_profiles: list[str],
) -> None:
    """Apply active-profile project-governance defaults when unset by user."""
    current = _normalize_project_governance_mapping(
        merged.get("project-governance")
    )
    user_owned = _normalize_project_governance_mapping(
        user_config.get("project-governance")
    )
    profile_defaults = _profile_project_governance_defaults(
        profile_registry,
        active_profiles,
    )
    for key, value in profile_defaults.items():
        if key in user_owned:
            continue
        current[key] = copy.deepcopy(value)
    merged["project-governance"] = current


def _materialize_policy_state_map(
    repo_root: Path, current_state: Dict[str, bool]
) -> Dict[str, bool]:
    """Return full alphabetical policy_state map from live policy sources."""
    discovered = _discover_policy_sources(repo_root)
    resolved: Dict[str, bool] = {}
    for policy_id in sorted(discovered):
        if policy_id in current_state:
            resolved[policy_id] = current_state[policy_id]
            continue
        default_enabled = True
        descriptor = load_policy_descriptor(repo_root, policy_id)
        if descriptor is not None:
            raw_enabled = descriptor.metadata.get("enabled")
            if isinstance(raw_enabled, bool):
                default_enabled = raw_enabled
            elif raw_enabled is not None:
                token = str(raw_enabled).strip().lower()
                if token in {"true", "1", "yes", "y", "on"}:
                    default_enabled = True
                elif token in {"false", "0", "no", "n", "off"}:
                    default_enabled = False
        resolved[policy_id] = default_enabled
    return resolved


def _normalize_string_list(raw_value: object) -> list[str]:
    """Normalize raw config values into a clean string list."""
    if isinstance(raw_value, str):
        items = [raw_value]
    elif isinstance(raw_value, list):
        items = raw_value
    else:
        return []

    cleaned: list[str] = []
    for raw_entry in items:
        token = str(raw_entry or "").strip()
        if token:
            cleaned.append(token)
    return cleaned


def _default_core_paths(repo_root: Path) -> list[str]:
    """Return canonical devcov core paths from manifest inventory."""
    del repo_root
    return manifest_module.default_scan_excluded_core_paths()


def _config_autogen_metadata_overlays(
    repo_root: Path,
    active_profiles: list[str],
    *,
    profile_registry: dict[str, dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    """Build deterministic profile-derived autogen metadata overlays."""
    overlays = metadata_runtime.collect_profile_overlays(
        repo_root,
        active_profiles,
        profile_registry=profile_registry,
    )
    normalized: Dict[str, Dict[str, object]] = {}
    for policy_id in sorted(overlays.keys()):
        policy_map = overlays[policy_id]
        key_map: Dict[str, object] = {}
        for key_name in sorted(policy_map.keys()):
            value = copy.deepcopy(policy_map[key_name])
            if value in ("", [], {}):
                continue
            key_map[key_name] = value
        if key_map:
            normalized[policy_id] = key_map
    return normalized


def _config_autogen_metadata_overrides() -> Dict[str, Dict[str, object]]:
    """Return generated metadata overrides owned by refresh runtime."""
    return {}


def _profile_registry_profiles(
    registry: dict[str, dict],
) -> dict[str, dict[str, object]]:
    """Return normalized profile map from a profile registry payload."""
    raw_profiles = registry.get("profiles")
    if not isinstance(raw_profiles, dict):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for name, payload in raw_profiles.items():
        if not isinstance(payload, dict):
            continue
        normalized[str(name).strip().lower()] = payload
    return normalized


def _merge_mapping_fragment(
    base_payload: dict[str, object],
    fragment: dict[str, object],
) -> dict[str, object]:
    """Merge mapping fragments recursively with append-dedupe lists."""
    merged = copy.deepcopy(base_payload)
    for metadata_key, incoming_value in fragment.items():
        existing = merged.get(metadata_key)
        if isinstance(existing, dict) and isinstance(incoming_value, dict):
            merged[metadata_key] = _merge_mapping_fragment(
                existing,
                incoming_value,
            )
            continue
        if isinstance(existing, list) and isinstance(incoming_value, list):
            combined = copy.deepcopy(existing)
            for item in incoming_value:
                candidate = copy.deepcopy(item)
                if candidate not in combined:
                    combined.append(candidate)
            merged[metadata_key] = combined
            continue
        merged[metadata_key] = copy.deepcopy(incoming_value)
    return merged


def _load_active_ci_and_test_template(
    repo_root: Path,
    profiles_map: dict[str, dict[str, object]],
    active_profiles: list[str],
) -> tuple[str, dict[str, object]] | None:
    """Load the one active CI-and-test workflow template, if any."""
    active_owners: list[str] = []
    for profile_name in active_profiles:
        normalized = str(profile_name or "").strip().lower()
        if not normalized:
            continue
        profile_payload = profiles_map.get(normalized, {})
        template_name = str(
            profile_payload.get("ci_and_test_template", "")
        ).strip()
        if template_name:
            active_owners.append(normalized)

    if not active_owners:
        return None
    if len(active_owners) > 1:
        owners = ", ".join(active_owners)
        raise ValueError(
            "Multiple active profiles define ci_and_test_template: "
            f"{owners}."
        )

    owner_name = active_owners[0]
    owner_profile = profiles_map.get(owner_name, {})
    template_name = str(owner_profile.get("ci_and_test_template", "")).strip()
    if not template_name:
        raise ValueError(
            f"Active profile '{owner_name}' is missing ci_and_test_template."
        )

    profile_path = str(owner_profile.get("path", "")).strip()
    if not profile_path:
        raise ValueError(f"Active profile '{owner_name}' path is unavailable.")
    profile_root = _resolve_path_under_root(
        repo_root,
        profile_path,
        field_name=f"profile root ({owner_name})",
    )

    template_path = _resolve_path_under_root(
        profile_root / "assets",
        template_name,
        field_name=f"CI-and-test template ({owner_name})",
    )
    payload = _read_yaml(template_path)
    if not isinstance(payload, dict):
        raise ValueError("CI-and-test template must contain a YAML mapping.")
    return owner_name, payload


def _config_ci_and_test_adjustments(
    config: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    """Resolve config overlays/overrides for CI-and-test generation."""
    ci_and_test_block = config.get("ci_and_test")
    if not isinstance(ci_and_test_block, dict):
        return {}, {}
    overlays = ci_and_test_block.get("overlays")
    if not isinstance(overlays, dict):
        overlays = {}
    overrides = ci_and_test_block.get("overrides")
    if not isinstance(overrides, dict):
        overrides = {}
    return overlays, overrides


def _normalize_governance_trigger_key(
    payload: dict[str, object],
) -> dict[str, object]:
    """Normalize CI-and-test trigger key to literal ``on``."""
    normalized = copy.deepcopy(payload)
    if "on" in normalized:
        normalized.pop(True, None)
    elif True in normalized:
        normalized["on"] = normalized.pop(True)

    if "on" not in normalized:
        return normalized

    ordered: dict[str, object] = {}
    if "name" in normalized:
        ordered["name"] = normalized["name"]
    ordered["on"] = normalized["on"]
    for key, value in normalized.items():
        if key in {"name", "on"}:
            continue
        ordered[key] = value
    return ordered


def _render_governance_workflow_yaml(payload: dict[str, object]) -> str:
    """Render CI-and-test workflow YAML in canonical GitHub syntax."""

    class _WorkflowYamlDumper(yaml.SafeDumper):
        """Preserve literal blocks for multiline workflow command strings."""

    def _represent_workflow_string(
        dumper: yaml.SafeDumper, text_value: str
    ) -> yaml.nodes.ScalarNode:
        """Render multiline workflow strings as literal blocks."""
        style = "|" if "\n" in text_value else None
        return dumper.represent_scalar(
            "tag:yaml.org,2002:str",
            text_value,
            style=style,
        )

    _WorkflowYamlDumper.add_representer(str, _represent_workflow_string)
    rendered = yaml.dump(
        payload,
        Dumper=_WorkflowYamlDumper,
        sort_keys=False,
    )
    lines = rendered.splitlines()
    normalized_lines: list[str] = []
    in_on_block = False
    null_event_pattern = re.compile(r"^(\s+[A-Za-z0-9_-]+): null$")

    for line in lines:
        if line in {"'on':", '"on":'}:
            normalized_lines.append("on:")
            in_on_block = True
            continue

        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if indent == 0 and stripped and stripped != "on:":
            in_on_block = False

        if in_on_block:
            null_event_match = null_event_pattern.match(line)
            if null_event_match:
                normalized_lines.append(f"{null_event_match.group(1)}:")
                continue

        normalized_lines.append(line)

    normalized = "\n".join(normalized_lines)
    if rendered.endswith("\n"):
        normalized += "\n"
    return normalized


def _refresh_ci_and_test(
    repo_root: Path,
    config: dict[str, object],
    profile_registry: dict[str, dict],
    active_profiles: list[str],
) -> bool:
    """Regenerate the CI workflow from template and fragments."""
    profiles_map = _profile_registry_profiles(profile_registry)
    overlays, overrides = _config_ci_and_test_adjustments(config)
    loaded = _load_active_ci_and_test_template(
        repo_root,
        profiles_map,
        active_profiles,
    )
    if loaded is None:
        if overrides:
            payload = copy.deepcopy(overrides)
        elif overlays:
            raise ValueError(
                "config.ci_and_test.overlays requires an active profile "
                "that defines ci_and_test_template, such as `github`."
            )
        else:
            return False
    else:
        _, payload = loaded

    for profile_name in active_profiles:
        normalized = str(profile_name or "").strip().lower()
        if not normalized:
            continue
        profile_payload = profiles_map.get(normalized, {})
        fragment = profile_payload.get("ci_and_test")
        if isinstance(fragment, dict):
            payload = _merge_mapping_fragment(payload, fragment)
    if overlays:
        payload = _merge_mapping_fragment(payload, overlays)
    if overrides:
        payload = copy.deepcopy(overrides)
    payload = _normalize_governance_trigger_key(payload)

    target_path = repo_root / ".github" / "workflows" / "ci.yml"
    rendered = _render_governance_workflow_yaml(payload)
    changed = True
    if target_path.exists():
        changed = target_path.read_text(encoding="utf-8") != rendered
    if changed:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(rendered, encoding="utf-8")
    return changed


def _merge_repo_hooks(
    base_hooks: list[object], incoming_hooks: list[object]
) -> list[object]:
    """Merge pre-commit hook lists by hook id while preserving order."""
    merged = copy.deepcopy(base_hooks)
    hook_indexes: dict[str, int] = {}
    for index, hook in enumerate(merged):
        if not isinstance(hook, dict):
            continue
        hook_id = str(hook.get("id", "")).strip()
        if hook_id and hook_id not in hook_indexes:
            hook_indexes[hook_id] = index

    for hook in incoming_hooks:
        if not isinstance(hook, dict):
            merged.append(copy.deepcopy(hook))
            continue
        hook_id = str(hook.get("id", "")).strip()
        if not hook_id or hook_id not in hook_indexes:
            merged.append(copy.deepcopy(hook))
            if hook_id:
                hook_indexes[hook_id] = len(merged) - 1
            continue
        existing = merged[hook_indexes[hook_id]]
        if isinstance(existing, dict):
            updated = copy.deepcopy(existing)
            updated.update(copy.deepcopy(hook))
            merged[hook_indexes[hook_id]] = updated
            continue
        merged[hook_indexes[hook_id]] = copy.deepcopy(hook)
    return merged


def _merge_repo_entries(
    base_repos: list[object], incoming_repos: list[object]
) -> list[object]:
    """Merge pre-commit repo entries by repo identifier."""
    merged = copy.deepcopy(base_repos)
    repo_indexes: dict[str, int] = {}
    for index, repo_entry in enumerate(merged):
        if not isinstance(repo_entry, dict):
            continue
        repo_name = str(repo_entry.get("repo", "")).strip()
        if repo_name and repo_name not in repo_indexes:
            repo_indexes[repo_name] = index

    for repo_entry in incoming_repos:
        if not isinstance(repo_entry, dict):
            merged.append(copy.deepcopy(repo_entry))
            continue
        repo_name = str(repo_entry.get("repo", "")).strip()
        if not repo_name or repo_name not in repo_indexes:
            merged.append(copy.deepcopy(repo_entry))
            if repo_name:
                repo_indexes[repo_name] = len(merged) - 1
            continue

        existing = merged[repo_indexes[repo_name]]
        if not isinstance(existing, dict):
            merged[repo_indexes[repo_name]] = copy.deepcopy(repo_entry)
            continue

        updated = copy.deepcopy(existing)
        for metadata_key, incoming_value in repo_entry.items():
            if metadata_key == "hooks" and isinstance(incoming_value, list):
                current_hooks = updated.get("hooks")
                if isinstance(current_hooks, list):
                    updated["hooks"] = _merge_repo_hooks(
                        current_hooks, incoming_value
                    )
                else:
                    updated["hooks"] = copy.deepcopy(incoming_value)
                continue
            updated[metadata_key] = copy.deepcopy(incoming_value)
        merged[repo_indexes[repo_name]] = updated
    return merged


def _merge_pre_commit_fragment(
    base_payload: dict[str, object], fragment: dict[str, object]
) -> dict[str, object]:
    """Merge one pre-commit fragment into a base payload."""
    merged = copy.deepcopy(base_payload)
    for metadata_key, incoming_value in fragment.items():
        if metadata_key == "repos" and isinstance(incoming_value, list):
            current_repos = merged.get("repos")
            if isinstance(current_repos, list):
                merged["repos"] = _merge_repo_entries(
                    current_repos, incoming_value
                )
            else:
                merged["repos"] = copy.deepcopy(incoming_value)
            continue
        existing = merged.get(metadata_key)
        if isinstance(existing, dict) and isinstance(incoming_value, dict):
            updated = copy.deepcopy(existing)
            updated.update(copy.deepcopy(incoming_value))
            merged[metadata_key] = updated
            continue
        merged[metadata_key] = copy.deepcopy(incoming_value)
    return merged


def _normalize_ignore_dir(raw: object) -> str:
    """Normalize ignore directory values for pre-commit exclude generation."""
    token = str(raw or "").strip().strip("/")
    if not token:
        return ""
    return token


def _build_pre_commit_exclude(ignore_dirs: list[str]) -> str:
    """Build a shared pre-commit exclude regex from ignore directories."""
    pattern_parts = [re.escape(entry) for entry in ignore_dirs if entry]
    pattern_parts.append(r"[^/]+\.egg-info")
    if not pattern_parts:
        return ""
    body = "\n".join(
        [
            "(?x)",
            "(^|/)",
            "(",
            "  " + "\n  | ".join(pattern_parts),
            ")",
            "(/|$)",
        ]
    )
    return body


def _resolved_pre_commit_hooks(payload: dict[str, object]) -> list[str]:
    """Return stable list of resolved hook identifiers."""
    hooks: list[str] = []
    repos_value = payload.get("repos")
    if not isinstance(repos_value, list):
        return hooks
    for repo_entry in repos_value:
        if not isinstance(repo_entry, dict):
            continue
        repo_name = str(repo_entry.get("repo", "")).strip()
        if not repo_name:
            continue
        hooks_value = repo_entry.get("hooks")
        if not isinstance(hooks_value, list):
            continue
        for hook_entry in hooks_value:
            if not isinstance(hook_entry, dict):
                continue
            hook_id = str(hook_entry.get("id", "")).strip()
            if not hook_id:
                continue
            hooks.append(f"{repo_name}:{hook_id}")
    return hooks


_EXCLUDE_PLACEHOLDER = "__DEVCOVENANT_EXCLUDE_PLACEHOLDER__"


def _render_pre_commit_yaml(payload: dict[str, object]) -> str:
    """Render pre-commit payload while preserving readable exclude blocks."""
    exclude_value = payload.get("exclude")
    if not isinstance(exclude_value, str) or "\n" not in exclude_value:
        return yaml.safe_dump(payload, sort_keys=False)

    serialized = copy.deepcopy(payload)
    serialized["exclude"] = _EXCLUDE_PLACEHOLDER
    rendered = yaml.safe_dump(serialized, sort_keys=False)
    literal_block = "\n".join(
        f"  {line}" for line in exclude_value.splitlines()
    )
    marker = f"exclude: {_EXCLUDE_PLACEHOLDER}\n"
    replacement = "exclude: |-\n" + literal_block + "\n"
    return rendered.replace(marker, replacement, 1)


def _record_pre_commit_manifest(
    repo_root: Path,
    active_profiles: list[str],
    pre_commit_payload: dict[str, object],
) -> None:
    """Persist resolved pre-commit metadata into tracked inventory."""
    manifest = manifest_module.ensure_manifest(repo_root)
    if not isinstance(manifest, dict):
        return

    profiles_block = manifest.get("profiles")
    if not isinstance(profiles_block, dict):
        profiles_block = {}

    resolved_hooks = _resolved_pre_commit_hooks(pre_commit_payload)
    changed = False
    if profiles_block.get("active") != active_profiles:
        profiles_block["active"] = list(active_profiles)
        changed = True
    if profiles_block.get("resolved_pre_commit_hooks") != resolved_hooks:
        profiles_block["resolved_pre_commit_hooks"] = resolved_hooks
        changed = True
    if not changed:
        return

    manifest["profiles"] = profiles_block
    manifest_module.write_manifest(repo_root, manifest)


def _ensure_devcovenant_hook_last(payload: dict[str, object]) -> None:
    """Move the local devcovenant hook to the end of pre-commit repos."""
    repos_value = payload.get("repos")
    if not isinstance(repos_value, list):
        return

    target_index = -1
    for index, repo_entry in enumerate(repos_value):
        if not isinstance(repo_entry, dict):
            continue
        if str(repo_entry.get("repo", "")).strip() != "local":
            continue
        hooks_value = repo_entry.get("hooks")
        if not isinstance(hooks_value, list):
            continue
        has_devcovenant = any(
            isinstance(hook_entry, dict)
            and str(hook_entry.get("id", "")).strip() == "devcovenant"
            for hook_entry in hooks_value
        )
        if has_devcovenant:
            target_index = index

    if target_index < 0 or target_index == len(repos_value) - 1:
        return

    target_entry = repos_value.pop(target_index)
    repos_value.append(target_entry)
    payload["repos"] = repos_value


def _find_devcovenant_hook(
    payload: dict[str, object],
) -> dict[str, object] | None:
    """Return a copy of the devcovenant local hook when present."""
    repos_value = payload.get("repos")
    if not isinstance(repos_value, list):
        return None
    for repo_entry in repos_value:
        if not isinstance(repo_entry, dict):
            continue
        if str(repo_entry.get("repo", "")).strip() != "local":
            continue
        hooks_value = repo_entry.get("hooks")
        if not isinstance(hooks_value, list):
            continue
        for hook_entry in hooks_value:
            if not isinstance(hook_entry, dict):
                continue
            if str(hook_entry.get("id", "")).strip() == "devcovenant":
                return copy.deepcopy(hook_entry)
    return None


def _ensure_devcovenant_hook_present(
    payload: dict[str, object],
    default_hook: dict[str, object] | None,
) -> None:
    """Ensure generated pre-commit payload contains the devcovenant hook."""
    if not default_hook:
        return
    repos_value = payload.get("repos")
    if not isinstance(repos_value, list):
        repos_value = []

    for index, repo_entry in enumerate(repos_value):
        if not isinstance(repo_entry, dict):
            continue
        if str(repo_entry.get("repo", "")).strip() != "local":
            continue
        hooks_value = repo_entry.get("hooks")
        if not isinstance(hooks_value, list):
            hooks_value = []
            repo_entry["hooks"] = hooks_value
        has_devcovenant = any(
            isinstance(hook_entry, dict)
            and str(hook_entry.get("id", "")).strip() == "devcovenant"
            for hook_entry in hooks_value
        )
        if not has_devcovenant:
            hooks_value.append(copy.deepcopy(default_hook))
        payload["repos"] = repos_value
        return

    repos_value.append(
        {
            "repo": "local",
            "hooks": [copy.deepcopy(default_hook)],
        }
    )
    payload["repos"] = repos_value


def _refresh_pre_commit_config(
    repo_root: Path,
    config: dict[str, object],
    profile_registry: dict[str, dict],
    active_profiles: list[str],
) -> bool:
    """Regenerate .pre-commit-config.yaml from fragments and overrides."""
    profiles_map = _profile_registry_profiles(profile_registry)
    payload: dict[str, object] = {}

    global_fragment = profiles_map.get("global", {}).get("pre_commit")
    if isinstance(global_fragment, dict):
        payload = _merge_pre_commit_fragment(payload, global_fragment)

    for profile_name in active_profiles:
        normalized = str(profile_name or "").strip().lower()
        if not normalized or normalized == "global":
            continue
        fragment = profiles_map.get(normalized, {}).get("pre_commit")
        if not isinstance(fragment, dict):
            continue
        payload = _merge_pre_commit_fragment(payload, fragment)

    ignore_dirs: list[str] = []
    profile_ignores = profile_registry_service.resolve_profile_ignore_dirs(
        profile_registry, active_profiles
    )
    for entry in profile_ignores:
        token = _normalize_ignore_dir(entry)
        if token and token not in ignore_dirs:
            ignore_dirs.append(token)

    engine_block = config.get("engine")
    if isinstance(engine_block, dict):
        raw_engine_ignores = engine_block.get("ignore_dirs")
        if isinstance(raw_engine_ignores, list):
            for entry in raw_engine_ignores:
                token = _normalize_ignore_dir(entry)
                if token and token not in ignore_dirs:
                    ignore_dirs.append(token)

    if ignore_dirs:
        payload["exclude"] = _build_pre_commit_exclude(ignore_dirs)

    devcovenant_hook = _find_devcovenant_hook(payload)

    pre_commit_block = config.get("pre_commit")
    if isinstance(pre_commit_block, dict):
        overlays = pre_commit_block.get("overlays")
        if isinstance(overlays, dict):
            payload = _merge_pre_commit_fragment(payload, overlays)
        overrides = pre_commit_block.get("overrides")
        if isinstance(overrides, dict) and overrides:
            payload = copy.deepcopy(overrides)

    if "repos" not in payload or not isinstance(payload.get("repos"), list):
        payload["repos"] = []

    _ensure_devcovenant_hook_present(payload, devcovenant_hook)
    _ensure_devcovenant_hook_last(payload)

    target_path = repo_root / ".pre-commit-config.yaml"
    rendered = _render_pre_commit_yaml(payload)
    changed = True
    if target_path.exists():
        changed = target_path.read_text(encoding="utf-8") != rendered
    if changed:
        target_path.write_text(rendered, encoding="utf-8")

    _record_pre_commit_manifest(repo_root, active_profiles, payload)
    return changed


def _read_text(path: Path) -> str:
    """Read UTF-8 text from a path, returning empty string when missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _normalize_gitignore_entries(raw_value: object) -> list[str]:
    """Normalize configured gitignore fragment entries."""
    if not isinstance(raw_value, list):
        return []
    entries: list[str] = []
    for raw_entry in raw_value:
        token = str(raw_entry or "").strip()
        if not token:
            continue
        if token in entries:
            continue
        entries.append(token)
    return entries


def _profile_gitignore_entries(
    profile_payload: dict[str, object],
) -> list[str]:
    """Resolve one profile's gitignore entries from manifest metadata."""
    explicit_entries = _normalize_gitignore_entries(
        profile_payload.get("gitignore_fragments")
    )
    if explicit_entries:
        return explicit_entries
    return _normalize_gitignore_entries(profile_payload.get("ignore_dirs"))


def _config_gitignore_adjustments(
    config: dict[str, object],
) -> tuple[list[str], list[str]]:
    """Resolve user-configured gitignore overlays and overrides."""
    gitignore_block = config.get("gitignore")
    if not isinstance(gitignore_block, dict):
        return [], []
    overlays = _normalize_gitignore_entries(gitignore_block.get("overlays"))
    overrides = _normalize_gitignore_entries(gitignore_block.get("overrides"))
    return overlays, overrides


def _load_global_gitignore_template(
    repo_root: Path,
    profile_registry: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Load global gitignore base/os entries from configured YAML template."""
    profiles_map = _profile_registry_profiles(profile_registry)
    global_profile = profiles_map.get("global", {})
    template_name = str(global_profile.get("gitignore_template", "")).strip()
    if not template_name:
        raise ValueError("Global profile is missing gitignore_template.")
    profile_path = str(global_profile.get("path", "")).strip()
    if not profile_path:
        raise ValueError("Global profile path is missing in registry.")
    profile_root = _resolve_path_under_root(
        repo_root,
        profile_path,
        field_name="global profile root",
    )
    template_path = _resolve_path_under_root(
        profile_root / "assets",
        template_name,
        field_name="global gitignore template",
    )
    template_payload = _read_yaml(template_path)
    base_entries = _normalize_gitignore_entries(
        template_payload.get("base_fragments")
    )
    os_entries = _normalize_gitignore_entries(
        template_payload.get("os_fragments")
    )
    return base_entries, os_entries


def _extract_user_gitignore_entries(existing_text: str) -> list[str]:
    """Extract preserved user entries from an existing .gitignore body."""
    begin_index = existing_text.find(USER_GITIGNORE_BEGIN)
    end_index = existing_text.find(USER_GITIGNORE_END)
    if begin_index < 0 or end_index < 0 or end_index < begin_index:
        return [line.rstrip() for line in existing_text.splitlines() if line]

    body_start = begin_index + len(USER_GITIGNORE_BEGIN)
    body_text = existing_text[body_start:end_index]
    user_lines = [line.rstrip() for line in body_text.splitlines()]
    while user_lines and not user_lines[0].strip():
        user_lines.pop(0)
    while user_lines and not user_lines[-1].strip():
        user_lines.pop()
    return user_lines


def _render_gitignore(
    base_entries: list[str],
    os_entries: list[str],
    profile_sections: list[tuple[str, list[str]]],
    config_overlays: list[str],
    config_overrides: list[str],
    user_entries: list[str],
) -> str:
    """Render full .gitignore with generated and preserved user sections."""
    sections: list[str] = []
    if config_overrides:
        sections.append(
            "\n".join(["# Config gitignore overrides", *config_overrides])
        )
    else:
        if base_entries:
            sections.append(
                "\n".join(["# DevCovenant base ignores", *base_entries])
            )

        for profile_name, fragment_entries in profile_sections:
            if not fragment_entries:
                continue
            section_header = f"# Profile: {profile_name}"
            section_body = "\n".join(fragment_entries)
            sections.append("\n".join([section_header, section_body]))

        if os_entries:
            sections.append(
                "\n".join(["# OS-specific ignores (DevCovenant)", *os_entries])
            )

        if config_overlays:
            sections.append(
                "\n".join(["# Config gitignore overlays", *config_overlays])
            )

    user_block_lines = [USER_GITIGNORE_BEGIN, ""]
    user_block_lines.extend(user_entries)
    user_block_lines.extend(["", USER_GITIGNORE_END])
    sections.append("\n".join(user_block_lines))

    return (
        "\n\n".join(section for section in sections if section).rstrip() + "\n"
    )


def _refresh_gitignore(
    repo_root: Path,
    config: dict[str, object],
    profile_registry: dict[str, dict],
    active_profiles: list[str],
) -> bool:
    """Regenerate .gitignore from template, profiles, and config metadata."""
    profiles_map = _profile_registry_profiles(profile_registry)
    profile_sections: list[tuple[str, list[str]]] = []
    for profile_name in active_profiles:
        normalized_name = str(profile_name or "").strip().lower()
        if not normalized_name:
            continue
        profile_payload = profiles_map.get(normalized_name, {})
        fragment_entries = _profile_gitignore_entries(profile_payload)
        profile_sections.append((normalized_name, fragment_entries))

    base_entries, os_entries = _load_global_gitignore_template(
        repo_root,
        profile_registry,
    )
    config_overlays, config_overrides = _config_gitignore_adjustments(config)
    gitignore_path = repo_root / ".gitignore"
    current_text = _read_text(gitignore_path)
    user_entries = _extract_user_gitignore_entries(current_text)
    rendered = _render_gitignore(
        base_entries,
        os_entries,
        profile_sections,
        config_overlays,
        config_overrides,
        user_entries,
    )
    if current_text == rendered:
        return False
    gitignore_path.write_text(rendered, encoding="utf-8")
    return True


def _refresh_dependency_artifacts(repo_root: Path) -> list[str]:
    """Refresh dependency-management outputs during full refresh."""

    try:
        payload = runtime_actions_module.run_policy_runtime_action(
            repo_root,
            policy_id="dependency-management",
            action="refresh-all",
            payload={},
        )
    except (
        ValueError,
        RuntimeError,
        subprocess.CalledProcessError,
    ) as error:
        raise ValueError(
            "Dependency-management refresh failed: " f"{error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            "Dependency-management refresh returned an invalid payload."
        )
    changed: list[str] = []
    raw_lock_results = payload.get("lock_results")
    if isinstance(raw_lock_results, list):
        for entry in raw_lock_results:
            if not isinstance(entry, dict) or not entry.get("changed"):
                continue
            token = str(entry.get("lock_file") or "").strip()
            if token:
                changed.append(token)
    raw_artifacts = payload.get("refreshed_artifacts")
    if isinstance(raw_artifacts, list):
        for entry in raw_artifacts:
            token = str(entry or "").strip()
            if token:
                changed.append(token)
    return list(dict.fromkeys(changed))


def refresh_repo(repo_root: Path) -> int:
    """Run full refresh for the repository."""
    phase_timings: list[dict[str, object]] = []
    config_path = repo_root / "devcovenant" / "config.yaml"
    try:
        bootstrap_started = time.perf_counter()
        try:
            config = _load_config_template(repo_root)
            user_config = (
                _read_yaml(config_path) if config_path.exists() else {}
            )
            _merge_user_config_values(config, user_config)
            active_profiles = _active_profiles(config)
            profile_registry = profile_registry_service.build_profile_registry(
                repo_root,
                active_profiles,
            )
            _apply_profile_aware_project_governance_defaults(
                config,
                user_config,
                profile_registry,
                active_profiles,
            )
            config["autogen_metadata_overlays"] = (
                _config_autogen_metadata_overlays(
                    repo_root,
                    active_profiles,
                    profile_registry=profile_registry,
                )
            )
            project_governance_state = (
                project_governance_service.resolve_runtime_state(
                    repo_root,
                    config_payload=config,
                )
            )
            declared_project_version = _read_project_version(
                repo_root,
                config,
                required=not project_governance_state.is_unversioned,
            )
            project_version = (
                project_governance_state.displayed_project_version(
                    declared_project_version
                )
            )
            devcovenant_version = _read_devcovenant_version(repo_root)
            import_managed_docs = _install_import_managed_docs(config)
            seeded_agents = _sync_doc(
                repo_root,
                "AGENTS.md",
                config_payload=config,
                project_version=project_version,
                devcovenant_version=devcovenant_version,
                project_governance_state=project_governance_state,
                import_managed_docs=import_managed_docs,
            )
        except ValueError as error:
            print_step(f"Refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "bootstrap",
            bootstrap_started,
            changed=seeded_agents,
        )

        assets_started = time.perf_counter()
        try:
            refreshed_assets = _refresh_profile_assets(
                repo_root,
                profile_registry,
                active_profiles,
                project_governance_state,
            )
        except ValueError as error:
            print_step(f"Profile asset refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "profile_assets",
            assets_started,
            changed=bool(refreshed_assets),
        )
        if refreshed_assets:
            print_step(
                "Materialized profile assets: " + ", ".join(refreshed_assets),
                "✅",
            )

        config_started = time.perf_counter()
        try:
            config, config_changed = _refresh_config_generated(
                repo_root,
                config_path,
                config,
                user_config,
                profile_registry,
                active_profiles,
            )
        except ValueError as error:
            print_step(f"Config refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "config",
            config_started,
            changed=config_changed,
        )
        if config_changed:
            print_step("Refreshed config generated profile metadata", "✅")

        profile_registry_started = time.perf_counter()
        profile_registry_service.write_profile_registry(
            repo_root, profile_registry
        )
        _record_phase_timing(
            phase_timings,
            "profile_registry",
            profile_registry_started,
        )

        policy_registry_started = time.perf_counter()
        registry_result = refresh_policy_registry(
            repo_root,
            config_payload=config,
            profile_registry=profile_registry,
        )
        _record_phase_timing(
            phase_timings,
            "policy_registry",
            policy_registry_started,
            changed=registry_result == 0,
        )
        if registry_result != 0:
            return registry_result

        agents_block_started = time.perf_counter()
        agents_path = repo_root / "AGENTS.md"
        try:
            refresh_agents_policy_block(agents_path, None, repo_root=repo_root)
        except ValueError as error:
            print_step(f"AGENTS block refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "agents_block",
            agents_block_started,
        )

        dependency_started = time.perf_counter()
        try:
            refreshed_dependency_artifacts = _refresh_dependency_artifacts(
                repo_root
            )
        except ValueError as error:
            print_step(str(error), "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "dependency_artifacts",
            dependency_started,
            changed=bool(refreshed_dependency_artifacts),
        )
        if refreshed_dependency_artifacts:
            print_step(
                "Refreshed dependency artifacts: "
                + ", ".join(refreshed_dependency_artifacts),
                "✅",
            )

        ci_started = time.perf_counter()
        try:
            ci_and_test_changed = _refresh_ci_and_test(
                repo_root,
                config,
                profile_registry,
                active_profiles,
            )
        except ValueError as error:
            print_step(f"CI-and-test workflow refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "ci_and_test",
            ci_started,
            changed=ci_and_test_changed,
        )
        if ci_and_test_changed:
            print_step("Regenerated CI workflow", "✅")

        pre_commit_timer_started = time.perf_counter()
        try:
            pre_commit_changed = _refresh_pre_commit_config(
                repo_root,
                config,
                profile_registry,
                active_profiles,
            )
        except ValueError as error:
            print_step(f"Pre-commit config refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "pre_commit",
            pre_commit_timer_started,
            changed=pre_commit_changed,
        )
        if pre_commit_changed:
            print_step(
                "Regenerated .pre-commit-config.yaml from profile fragments",
                "✅",
            )

        gitignore_started = time.perf_counter()
        try:
            gitignore_changed = _refresh_gitignore(
                repo_root,
                config,
                profile_registry,
                active_profiles,
            )
        except ValueError as error:
            print_step(f"Gitignore refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "gitignore",
            gitignore_started,
            changed=gitignore_changed,
        )
        if gitignore_changed:
            print_step("Regenerated .gitignore from profile fragments", "✅")

        managed_docs_started = time.perf_counter()
        try:
            docs = _managed_docs_from_config(config)
        except ValueError as error:
            print_step(f"Managed doc routing refresh failed: {error}", "🚫")
            return 1
        try:
            synced = [
                doc
                for doc in docs
                if _sync_doc(
                    repo_root,
                    doc,
                    config_payload=config,
                    project_version=project_version,
                    devcovenant_version=devcovenant_version,
                    project_governance_state=project_governance_state,
                    import_managed_docs=import_managed_docs,
                )
            ]
        except ValueError as error:
            print_step(f"Managed doc refresh failed: {error}", "🚫")
            return 1
        _record_phase_timing(
            phase_timings,
            "managed_docs",
            managed_docs_started,
            changed=bool(synced),
        )
        if synced:
            print_step(f"Synchronized managed docs: {', '.join(synced)}", "✅")

        pyproject_started = time.perf_counter()
        pyproject_changed = _sync_project_pyproject_identity(
            repo_root,
            project_governance_state,
        )
        _record_phase_timing(
            phase_timings,
            "pyproject_identity",
            pyproject_started,
            changed=pyproject_changed,
        )
        if pyproject_changed:
            print_step(
                "Synchronized pyproject package identity "
                "from project governance",
                "✅",
            )

        return 0
    finally:
        merge_active_run_phase_timings(
            "refresh",
            phase_timings,
        )


# ---- AGENTS policy block refresh ----


RefreshResult = agents_blocks_lib.PolicyBlockRefreshResult
refresh_agents_policy_block = agents_blocks_lib.refresh_agents_policy_block


# ---- Local policy registry refresh ----
def _ensure_trailing_newline(path: Path) -> bool:
    """Ensure the given file ends with a newline."""
    if not path.exists():
        return False
    contents = path.read_bytes()
    if not contents:
        path.write_text("\n", encoding="utf-8")
        return True
    if contents.endswith(b"\n"):
        return False
    path.write_bytes(contents + b"\n")
    return True


def _discover_policy_sources(repo_root: Path) -> Dict[str, Dict[str, bool]]:
    """Return policy ids and whether builtin/custom scripts exist."""

    discovered: Dict[str, Dict[str, bool]] = {}
    for source in ("builtin", "custom"):
        source_root = repo_root / "devcovenant" / source / "policies"
        if not source_root.exists():
            continue
        for entry in sorted(
            source_root.iterdir(),
            key=lambda candidate: candidate.name.lower(),
        ):
            if not entry.is_dir():
                continue
            script = entry / f"{entry.name}.py"
            if not script.exists():
                continue
            policy_id = entry.name.replace("_", "-").strip()
            record = discovered.setdefault(
                policy_id, {"builtin": False, "custom": False}
            )
            if source == "builtin":
                record["builtin"] = True
            else:
                record["custom"] = True
    return discovered


def _as_bool(raw_value: object, *, default: bool) -> bool:
    """Interpret a resolved metadata value as a boolean."""

    if raw_value is None:
        return default
    if isinstance(raw_value, list):
        for entry in raw_value:
            token = str(entry or "").strip().lower()
            if token:
                break
        else:
            return default
    elif isinstance(raw_value, dict):
        return default
    else:
        token = str(raw_value).strip().lower()
    if token in {"true", "1", "yes", "on"}:
        return True
    if token in {"false", "0", "no", "off"}:
        return False
    return default


def _resolve_policy_sources(
    repo_root: Path, policy_id: str
) -> tuple[object | None, bool, bool]:
    """Resolve active script location and source availability flags."""
    location = resolve_script_location(repo_root, policy_id)
    available = {
        loc.kind
        for loc in iter_script_locations(repo_root, policy_id)
        if loc.path.exists()
    }
    return location, "builtin" in available, "custom" in available


def _sha256_bytes(payload: bytes) -> str:
    """Return one stable SHA-256 digest string."""

    return hashlib.sha256(payload).hexdigest()


def _hash_file_or_missing(path: Path) -> str:
    """Return one stable file digest or a placeholder for missing files."""

    if not path.exists():
        return "__missing__"
    return _sha256_bytes(path.read_bytes())


def _policy_registry_input_fingerprint(
    repo_root: Path,
    *,
    config_payload: dict[str, object],
    discovered: set[str],
) -> str:
    """Return one stable fingerprint for policy-registry refresh inputs."""

    policy_sources: list[dict[str, object]] = []
    for policy_id in sorted(discovered):
        (
            location,
            builtin_available,
            custom_available,
        ) = _resolve_policy_sources(repo_root, policy_id)
        if location is None:
            continue
        descriptor_path = location.path.with_suffix(".yaml")
        policy_sources.append(
            {
                "id": policy_id,
                "origin": location.kind,
                "builtin_available": builtin_available,
                "custom_available": custom_available,
                "script_path": str(location.path.relative_to(repo_root)),
                "script_hash": _hash_file_or_missing(location.path),
                "descriptor_path": str(descriptor_path.relative_to(repo_root)),
                "descriptor_hash": _hash_file_or_missing(descriptor_path),
            }
        )
    serialized = json.dumps(
        {
            "config_payload": config_payload,
            "policies": policy_sources,
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return _sha256_bytes(serialized)


def refresh_policy_registry(
    repo_root: Path | None = None,
    *,
    config_payload: dict[str, object] | None = None,
    profile_registry: dict[str, dict] | None = None,
) -> int:
    """Refresh policy hashes.

    Writes devcovenant/registry/registry.yaml.
    """

    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    agents_md_path = repo_root / "AGENTS.md"
    registry_path = policy_registry_path(repo_root)

    if not agents_md_path.exists():
        runtime_print(
            f"Error: AGENTS.md not found at {agents_md_path}",
            file=sys.stderr,
        )
        return 1

    if config_payload is None:
        try:
            config_payload = _read_yaml(
                repo_root / "devcovenant" / "config.yaml"
            )
        except ValueError as error:
            runtime_print(f"Error: {error}", file=sys.stderr)
            return 1
    try:
        context = metadata_runtime.build_metadata_context_from_payload(
            repo_root,
            dict(config_payload),
        )
    except ValueError as error:
        runtime_print(f"Error: {error}", file=sys.stderr)
        return 1
    config_context = CheckContext(repo_root=repo_root, config=config_payload)
    discovered = _discover_policy_sources(repo_root)

    registry = PolicyRegistry(registry_path, repo_root)
    input_fingerprint = _policy_registry_input_fingerprint(
        repo_root,
        config_payload=dict(config_payload),
        discovered=discovered,
    )
    if registry_path.exists() and (
        registry.get_registry_metadata_value("policy_registry_input_hash")
        == input_fingerprint
    ):
        runtime_print("Policy registry already current.", verbose_only=True)
        return 0

    updated = 0
    policies: List[PolicyDefinition] = []
    seen_policy_ids: set[str] = set()
    metadata_warning_targets: List[str] = []
    for policy_id in sorted(discovered):
        location, _builtin_available, _custom_available = (
            _resolve_policy_sources(repo_root, policy_id)
        )
        if location is None:
            runtime_print(
                f"Error: Policy script missing for {policy_id}.",
                file=sys.stderr,
            )
            return 1
        else:
            updated += 1
        try:
            descriptor = load_policy_descriptor(repo_root, policy_id)
        except ValueError as error:
            runtime_print(f"Error: {error}", file=sys.stderr)
            return 1
        if descriptor is None:
            runtime_print(
                (f"Error: Descriptor missing for {policy_id}."),
                file=sys.stderr,
            )
            return 1
        policy_text = str(descriptor.text or "").strip()
        if not policy_text:
            runtime_print(
                (f"Error: Descriptor text missing for {policy_id}."),
                file=sys.stderr,
            )
            return 1

        current_order = list(descriptor.metadata.keys())
        current_values = {
            key: copy.deepcopy(descriptor.metadata.get(key))
            for key in current_order
        }
        bundle = metadata_runtime.resolve_policy_metadata_bundle(
            policy_id,
            current_order,
            current_values,
            descriptor,
            context,
            custom_policy=bool(location and location.kind == "custom"),
        )
        resolved_order = bundle.order
        resolved_metadata = bundle.raw_map
        ordered_metadata = {
            key: resolved_metadata.get(key, "") for key in resolved_order
        }
        runtime_option_views = (
            runtime_actions_module.build_runtime_policy_option_views(
                bundle.decode_options(),
                config_context.get_policy_config(policy_id),
            )
        )
        severity = ordered_metadata.get("severity") or "warning"
        enabled = _as_bool(ordered_metadata.get("enabled"), default=True)
        custom = _as_bool(ordered_metadata.get("custom"), default=False)
        auto_fix = _as_bool(ordered_metadata.get("auto_fix"), default=False)
        policy_name = policy_id.replace("-", " ").title()
        policy = PolicyDefinition(
            policy_id=policy_id,
            name=policy_name,
            severity=severity,
            auto_fix=auto_fix,
            enabled=enabled,
            custom=custom,
            description=policy_text,
            raw_metadata=dict(ordered_metadata),
        )
        seen_policy_ids.add(policy_id)
        policies.append(policy)
        registry.update_policy_entry(
            policy,
            location,
            descriptor,
            resolved_metadata=ordered_metadata,
            metadata_resolution=bundle.resolution_trace,
            metadata_warnings=bundle.warnings,
            runtime_option_views=runtime_option_views,
            save=False,
        )
        for warning in bundle.warning_messages():
            metadata_warning_targets.append(f"{policy_id}: {warning}")
        script_name = (
            location.path.name if location is not None else "<missing>"
        )
        runtime_print(
            f"Recorded {policy_id}: {script_name}",
            verbose_only=True,
        )

    stale_ids = registry.prune_policies(seen_policy_ids, save=False)
    for stale_id in stale_ids:
        runtime_print(
            f"Removed stale policy entry: {stale_id}",
            verbose_only=True,
        )

    registry.update_registry_metadata_value(
        "policy_registry_input_hash",
        input_fingerprint,
        save=False,
    )
    registry.save()

    if updated == 0:
        runtime_print("All policy hashes are up to date.", verbose_only=True)
    else:
        runtime_print(
            "\nUpdated " f"{updated} policy hash(es) in {registry_path}",
            verbose_only=True,
        )

    if _ensure_trailing_newline(registry_path):
        runtime_print(
            f"Ensured trailing newline in {registry_path}.",
            verbose_only=True,
        )
    if metadata_warning_targets:
        runtime_print(
            "Recorded metadata replacement warnings in registry.yaml "
            f"for {len(metadata_warning_targets)} key(s).",
            verbose_only=True,
        )

    try:
        project_governance_state = (
            project_governance_service.resolve_runtime_state(
                repo_root,
                config_payload=config_payload,
            )
        )
        declared_project_version = _read_project_version(
            repo_root,
            config_payload,
            required=not project_governance_state.is_unversioned,
        )
        registry.update_project_governance(
            project_governance_state.registry_payload(declared_project_version)
        )
        registry.update_managed_docs(
            managed_docs_service.managed_docs_registry_payload(
                repo_root,
                config_payload=config_payload,
            )
        )
        if profile_registry is None:
            profile_registry = profile_registry_service.build_profile_registry(
                repo_root,
                _active_profiles(config_payload),
            )
        registry.update_workflow_contract(
            profile_registry.get("workflow_contract", {})
        )
    except ValueError as error:
        runtime_print(f"Error: {error}", file=sys.stderr)
        return 1

    return 0
