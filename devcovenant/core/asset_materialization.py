"""Shared asset-command helpers for profile assets and managed docs."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import devcovenant.core.managed_docs as managed_docs_service
import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.project_governance as project_governance_service
import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.repository_paths import display_path


@dataclass(frozen=True)
class MaterializableAssetCandidate:
    """One materializable asset or managed-doc candidate."""

    kind: str
    profile_name: str
    active: bool
    profile_path: str
    target_path: str
    template_path: Path | None = None
    descriptor_path: Path | None = None

    @property
    def basename(self) -> str:
        """Return the target basename for selection and output defaults."""
        return Path(self.target_path).name


@dataclass(frozen=True)
class MaterializedAssetResult:
    """Return value for one completed asset materialization."""

    candidate: MaterializableAssetCandidate
    output_path: Path
    modified: bool


def _read_config_payload(repo_root: Path) -> dict[str, object]:
    """Load `devcovenant/config.yaml` as a mapping."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    rendered = display_path(config_path, repo_root=repo_root)
    if not config_path.exists():
        raise ValueError(f"Missing config file: {rendered}")
    payload = yaml_cache_service.load_yaml(config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {rendered}")
    return payload


def _metadata_string_token(raw: object) -> str:
    """Normalize one metadata value into a single string token."""
    if isinstance(raw, list):
        for entry in raw:
            token = str(entry or "").strip()
            if token:
                return token
        return ""
    return str(raw or "").strip()


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


def _resolve_path_under_root(
    root: Path,
    raw_path: str,
    *,
    field_name: str,
) -> Path:
    """Resolve a relative path and ensure it stays under *root*."""
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
            raise ValueError(f"Missing project version file: {version_path}")
        return ""
    token = version_path.read_text(encoding="utf-8").strip()
    if token:
        return token
    if required:
        raise ValueError(f"Project version file is empty: {version_path}")
    return ""


def render_profile_asset_template_text(
    template_text: str,
    project_governance_state: (
        project_governance_service.ProjectGovernanceState
    ),
) -> str:
    """Render project-identity placeholders for one raw profile template."""
    rendered = str(template_text or "")
    for placeholder, value in (
        (
            '"{{ PROJECT_NAME }}"',
            json.dumps(project_governance_state.project_name),
        ),
        (
            '"{{ PROJECT_DESCRIPTION }}"',
            json.dumps(project_governance_state.project_description),
        ),
        (
            "'{{ PROJECT_NAME }}'",
            json.dumps(project_governance_state.project_name),
        ),
        (
            "'{{ PROJECT_DESCRIPTION }}'",
            json.dumps(project_governance_state.project_description),
        ),
    ):
        rendered = rendered.replace(placeholder, value)
    return project_governance_service.render_identity_placeholders(
        rendered,
        project_governance_state,
    )


def _normalize_query(token: str) -> str:
    """Normalize a requested asset token for exact-path matching."""
    normalized = str(token or "").strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _discover_managed_doc_candidates_for_profile(
    repo_root: Path,
    *,
    profile_name: str,
    profile_path: str,
    active: bool,
) -> list[MaterializableAssetCandidate]:
    """Return managed-doc descriptor candidates from one assets root."""
    profile_root = _resolve_path_under_root(
        repo_root,
        profile_path,
        field_name=f"profile root ({profile_name})",
    )
    assets_root = profile_root / "assets"
    if not assets_root.exists():
        return []
    candidates: list[MaterializableAssetCandidate] = []
    for (
        entry
    ) in managed_docs_service.managed_doc_descriptor_entries_from_roots(
        [assets_root]
    ):
        target_path = str(entry.get("doc", "")).strip()
        descriptor_path = entry.get("descriptor_path")
        if not target_path or not isinstance(descriptor_path, Path):
            continue
        candidates.append(
            MaterializableAssetCandidate(
                kind="managed_doc",
                profile_name=profile_name,
                active=active,
                profile_path=profile_path,
                target_path=target_path,
                descriptor_path=descriptor_path,
            )
        )
    return candidates


def _discover_profile_asset_candidates_for_profile(
    repo_root: Path,
    *,
    profile_name: str,
    profile_payload: dict[str, object],
    active: bool,
) -> list[MaterializableAssetCandidate]:
    """Return manifest-declared template candidates from one profile."""
    profile_path = str(profile_payload.get("path") or "").strip()
    if not profile_path:
        return []
    raw_assets = profile_payload.get("assets")
    if not isinstance(raw_assets, list):
        return []
    profile_root = _resolve_path_under_root(
        repo_root,
        profile_path,
        field_name=f"profile root ({profile_name})",
    )
    assets_root = profile_root / "assets"
    candidates: list[MaterializableAssetCandidate] = []
    for entry in raw_assets:
        if not isinstance(entry, dict):
            continue
        target_path = str(entry.get("path") or "").strip()
        template_token = str(entry.get("template") or "").strip()
        if not target_path or not template_token:
            continue
        template_path = _resolve_path_under_root(
            assets_root,
            template_token,
            field_name=f"profile asset template ({profile_name})",
        )
        candidates.append(
            MaterializableAssetCandidate(
                kind="profile_asset",
                profile_name=profile_name,
                active=active,
                profile_path=profile_path,
                target_path=target_path,
                template_path=template_path,
            )
        )
    return candidates


def _profile_priority_order(
    registry: dict[str, dict[str, object]],
    active_profiles: list[str],
) -> list[str]:
    """Return profile names in asset-selection priority order."""
    ordered: list[str] = []
    for name in active_profiles:
        normalized = str(name or "").strip().lower()
        if normalized and normalized in registry and normalized not in ordered:
            ordered.append(normalized)
    inactive = sorted(name for name in registry if name not in ordered)
    return ordered + inactive


def _discover_all_candidates(
    repo_root: Path,
) -> tuple[list[str], list[MaterializableAssetCandidate], dict[str, object]]:
    """Return profile order, all candidates, and loaded config payload."""
    config = _read_config_payload(repo_root)
    registry_payload = profile_registry_service.discover_profiles(repo_root)
    active_profiles = profile_registry_service.parse_active_profiles(
        config,
        include_global=True,
    )
    profile_order = _profile_priority_order(registry_payload, active_profiles)
    active_set = set(active_profiles)
    candidates: list[MaterializableAssetCandidate] = []
    for profile_name in profile_order:
        profile_payload = registry_payload.get(profile_name, {})
        profile_path = str(profile_payload.get("path") or "").strip()
        active = profile_name in active_set
        candidates.extend(
            _discover_profile_asset_candidates_for_profile(
                repo_root,
                profile_name=profile_name,
                profile_payload=profile_payload,
                active=active,
            )
        )
        if profile_path:
            candidates.extend(
                _discover_managed_doc_candidates_for_profile(
                    repo_root,
                    profile_name=profile_name,
                    profile_path=profile_path,
                    active=active,
                )
            )
    return profile_order, candidates, config


def resolve_materializable_asset(
    repo_root: Path,
    asset_name: str,
) -> MaterializableAssetCandidate:
    """Resolve one materializable asset/doc by exact path or basename."""
    requested = _normalize_query(asset_name)
    if not requested:
        raise ValueError("Asset name cannot be empty.")
    profile_order, candidates, _config = _discover_all_candidates(repo_root)
    exact_matches = [
        candidate
        for candidate in candidates
        if _normalize_query(candidate.target_path) == requested
    ]
    matches = exact_matches
    if not matches:
        basename = Path(requested).name
        matches = [
            candidate
            for candidate in candidates
            if candidate.basename == basename
        ]
    if not matches:
        raise ValueError(
            f"No materializable asset or managed doc matches `{asset_name}`."
        )

    profile_rank = {name: index for index, name in enumerate(profile_order)}
    matches.sort(
        key=lambda candidate: (
            profile_rank.get(candidate.profile_name, len(profile_rank)),
            candidate.target_path,
        )
    )
    winning = matches[0]
    same_profile_matches = [
        candidate
        for candidate in matches
        if candidate.profile_name == winning.profile_name
    ]
    if len(same_profile_matches) > 1 and not exact_matches:
        targets = ", ".join(
            sorted(candidate.target_path for candidate in same_profile_matches)
        )
        raise ValueError(
            "Asset name is ambiguous within the winning profile "
            f"`{winning.profile_name}`. Use an exact asset target path. "
            f"Choices: {targets}"
        )
    return winning


def _render_candidate_content(
    repo_root: Path,
    candidate: MaterializableAssetCandidate,
    *,
    config: dict[str, object],
) -> str:
    """Render one selected candidate into final text."""
    project_governance_state = (
        project_governance_service.resolve_runtime_state(
            repo_root,
            config_payload=config,
        )
    )
    if candidate.kind == "profile_asset":
        template_path = candidate.template_path
        if template_path is None or not template_path.exists():
            raise ValueError(
                f"Missing template source for `{candidate.target_path}`."
            )
        template_text = template_path.read_text(encoding="utf-8")
        return render_profile_asset_template_text(
            template_text,
            project_governance_state,
        )
    if candidate.kind == "managed_doc":
        declared_project_version = _read_project_version(
            repo_root,
            config,
            required=not project_governance_state.is_unversioned,
        )
        devcovenant_version = (
            (repo_root / "devcovenant" / "VERSION")
            .read_text(encoding="utf-8")
            .strip()
        )
        return managed_docs_service.render_doc(
            repo_root,
            candidate.target_path,
            project_version=project_governance_state.displayed_project_version(
                declared_project_version
            ),
            devcovenant_version=devcovenant_version or "0.0.0",
            project_governance_state=project_governance_state,
            config_payload=config,
        )
    raise ValueError(f"Unsupported asset kind `{candidate.kind}`.")


def resolve_desktop_directory() -> Path:
    """Resolve the user's Desktop path across supported OS families."""
    home = Path.home()
    if os.name == "nt":
        user_profile = str(os.environ.get("USERPROFILE", "")).strip()
        if user_profile:
            return Path(user_profile).expanduser() / "Desktop"
        return home / "Desktop"
    xdg_desktop = str(os.environ.get("XDG_DESKTOP_DIR", "")).strip()
    if xdg_desktop:
        return Path(
            xdg_desktop.replace("$HOME", str(home)).replace(
                "${HOME}", str(home)
            )
        ).expanduser()
    user_dirs = home / ".config" / "user-dirs.dirs"
    if user_dirs.exists():
        try:
            for line in user_dirs.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped.startswith("XDG_DESKTOP_DIR="):
                    continue
                _, raw_value = stripped.split("=", 1)
                token = raw_value.strip().strip('"').strip("'")
                if token:
                    return Path(
                        token.replace("$HOME", str(home)).replace(
                            "${HOME}", str(home)
                        )
                    ).expanduser()
        except OSError:
            pass
    return home / "Desktop"


def resolve_asset_output_path(
    candidate: MaterializableAssetCandidate,
    output_name: str | None,
) -> Path:
    """Resolve one Desktop-only output path with optional filename override."""
    default_dir = resolve_desktop_directory()
    default_name = candidate.basename
    token = str(output_name or "").strip()
    if not token:
        return default_dir / default_name
    if "/" in token or "\\" in token:
        raise ValueError(
            "Asset output name must be a plain filename, not a path."
        )
    requested = Path(token)
    if requested.is_absolute():
        raise ValueError("Asset output name must not be an absolute path.")
    if any(part in {"", ".", ".."} for part in requested.parts):
        raise ValueError(
            "Asset output name must be a plain filename, not a path."
        )
    if len(requested.parts) != 1:
        raise ValueError(
            "Asset output name must be a plain filename, not a path."
        )
    return default_dir / requested.name


def materialize_named_asset(
    repo_root: Path,
    asset_name: str,
    *,
    output_name: str | None = None,
    overwrite: bool = False,
) -> MaterializedAssetResult:
    """Render one selected asset/doc and write it to the resolved target."""
    candidate = resolve_materializable_asset(repo_root, asset_name)
    _profile_order, _candidates, config = _discover_all_candidates(repo_root)
    output_path = resolve_asset_output_path(candidate, output_name)
    if output_path.exists() and not overwrite:
        raise ValueError(
            f"Destination already exists: {output_path}. Use `--overwrite`."
        )
    content = _render_candidate_content(
        repo_root,
        candidate,
        config=config,
    )
    current = ""
    if output_path.exists() and output_path.is_file():
        current = output_path.read_text(encoding="utf-8")
    modified = current != content
    if modified or overwrite or not output_path.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content, encoding="utf-8")
        modified = True
    return MaterializedAssetResult(
        candidate=candidate,
        output_path=output_path,
        modified=modified,
    )
