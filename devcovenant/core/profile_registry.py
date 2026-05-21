"""Profile discovery and tracked profile-registry helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml

from devcovenant.core.repository_paths import (
    display_path,
    load_yaml,
    require_repo_relative_path,
)
from devcovenant.core.tracked_registry import (
    load_registry_document,
    write_registry_document,
)

REGISTRY_PROFILE = Path("devcovenant/registry/registry.yaml")
BUILTIN_PROFILE_ROOT = Path("devcovenant/builtin/profiles")
CUSTOM_PROFILE_ROOT = Path("devcovenant/custom/profiles")
LANGUAGE_CATEGORY = "language"
_CLEAN_OVERLAY_KEYS = (
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


def _load_yaml(path: Path) -> dict[str, object]:
    """Load YAML mapping content from path."""
    rendered = display_path(path)
    if not path.exists():
        raise ValueError(f"Missing YAML file: {rendered}")
    try:
        payload = load_yaml(path)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {rendered}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read {rendered}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"YAML file must contain a mapping: {rendered}")
    return payload


def _normalize_profile_name(raw: str) -> str:
    """Normalize a profile name for matching."""
    return str(raw or "").strip().lower()


def _active_profile_names(raw_active: object) -> list[str]:
    """Normalize active profile names from config-like payloads."""
    if isinstance(raw_active, str):
        candidates = [raw_active]
    elif isinstance(raw_active, list):
        candidates = raw_active
    else:
        candidates = [raw_active] if raw_active else []

    names: list[str] = []
    for entry in candidates:
        name = _normalize_profile_name(str(entry or ""))
        if not name or name in names:
            continue
        names.append(name)
    return names


def parse_active_profiles(
    config: dict[str, object], *, include_global: bool = True
) -> list[str]:
    """Resolve active profiles from config."""
    profiles_block = config.get("profiles")
    if isinstance(profiles_block, dict):
        names = _active_profile_names(profiles_block.get("active"))
    else:
        names = []
    if include_global:
        names = [name for name in names if name != "global"]
        names.insert(0, "global")
    return names


def _iter_profile_dirs(root: Path) -> list[Path]:
    """Return profile directories beneath a root."""
    if not root.exists():
        return []
    return sorted(
        [
            entry
            for entry in root.iterdir()
            if entry.is_dir()
            and not entry.name.startswith("_")
            and not entry.name.startswith(".")
        ],
        key=lambda entry: entry.name.lower(),
    )


def _relative_path(path: Path, base: Path) -> str:
    """Return a relative path when possible."""
    return require_repo_relative_path(base, path, label="profile path")


def _profile_assets(profile_dir: Path, repo_root: Path) -> list[str]:
    """List asset files under a profile directory."""
    assets: list[str] = []
    assets_root = profile_dir / "assets"
    scan_root = assets_root if assets_root.exists() else profile_dir
    for entry in scan_root.rglob("*"):
        if not entry.is_file():
            continue
        if "__pycache__" in entry.parts:
            continue
        if entry.suffix in {".pyc", ".pyo", ".pyd"}:
            continue
        if entry.name == f"{profile_dir.name}.yaml":
            continue
        assets.append(_relative_path(entry, repo_root))
    return sorted(assets)


def _manifest_path(profile_dir: Path) -> Path:
    """Return the preferred manifest path for a profile directory."""
    return profile_dir / f"{profile_dir.name}.yaml"


def _validate_profile_template_reference(
    profile_dir: Path,
    *,
    profile_name: str,
    source_label: str,
    field_name: str,
    template_value: object,
) -> None:
    """Validate one profile template reference under `assets/`."""
    template_text = str(template_value or "").strip()
    if not template_text:
        raise ValueError(
            f"{source_label} profile '{profile_name}' defines {field_name} "
            "with an empty template reference."
        )
    template_rel = Path(template_text)
    if template_rel.is_absolute() or ".." in template_rel.parts:
        raise ValueError(
            f"{source_label} profile '{profile_name}' defines {field_name} "
            f"with an invalid template path '{template_text}'."
        )
    template_path = profile_dir / "assets" / template_rel
    if not template_path.exists() or not template_path.is_file():
        raise ValueError(
            f"{source_label} profile '{profile_name}' references missing "
            f"template '{template_text}' in field {field_name}."
        )


def _validate_profile_asset_manifest(
    profile_dir: Path,
    profile_name: str,
    profile_meta: dict[str, object],
    *,
    source_label: str,
) -> None:
    """Validate manifest-declared template references for one profile."""
    for field_name in ("gitignore_template", "ci_and_test_template"):
        if field_name not in profile_meta:
            continue
        _validate_profile_template_reference(
            profile_dir,
            profile_name=profile_name,
            source_label=source_label,
            field_name=field_name,
            template_value=profile_meta.get(field_name),
        )

    raw_assets = profile_meta.get("assets")
    if raw_assets is None:
        return
    if not isinstance(raw_assets, list):
        raise ValueError(
            f"{source_label} profile '{profile_name}' must define assets as "
            "a list."
        )
    for index, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise ValueError(
                f"{source_label} profile '{profile_name}' has non-mapping "
                "asset entries."
            )
        asset_path = str(raw_asset.get("path") or "").strip()
        if not asset_path:
            raise ValueError(
                f"{source_label} profile '{profile_name}' asset entry "
                f"#{index + 1} is missing path."
            )
        if "template" not in raw_asset:
            continue
        _validate_profile_template_reference(
            profile_dir,
            profile_name=profile_name,
            source_label=source_label,
            field_name=f"assets[{index}].template",
            template_value=raw_asset.get("template"),
        )


def _normalize_translator_extensions(
    raw_value: object,
    *,
    profile_name: str,
    translator_id: str,
    source_label: str,
) -> list[str]:
    """Normalize translator extension lists."""
    if not isinstance(raw_value, list):
        raise ValueError(
            f"{source_label} translator '{translator_id}' in profile "
            f"'{profile_name}' must define extensions as a list."
        )
    normalized: list[str] = []
    for raw_entry in raw_value:
        token = str(raw_entry or "").strip().lower()
        if not token:
            continue
        if not token.startswith("."):
            token = f".{token}"
        if token not in normalized:
            normalized.append(token)
    if not normalized:
        raise ValueError(
            f"{source_label} translator '{translator_id}' in profile "
            f"'{profile_name}' must declare at least one extension."
        )
    return normalized


def _normalize_translator_strategy(
    raw_value: object,
    *,
    profile_name: str,
    translator_id: str,
    section: str,
    source_label: str,
) -> dict[str, object]:
    """Normalize a translator strategy block."""
    if not isinstance(raw_value, dict):
        raise ValueError(
            f"{source_label} translator '{translator_id}' in profile "
            f"'{profile_name}' must define {section} as a mapping."
        )
    normalized = dict(raw_value)
    strategy = str(normalized.get("strategy", "")).strip().lower()
    entrypoint = str(normalized.get("entrypoint", "")).strip()
    if not strategy:
        raise ValueError(
            f"{source_label} translator '{translator_id}' in profile "
            f"'{profile_name}' must define {section}.strategy."
        )
    if not entrypoint:
        raise ValueError(
            f"{source_label} translator '{translator_id}' in profile "
            f"'{profile_name}' must define {section}.entrypoint."
        )
    normalized["strategy"] = strategy
    normalized["entrypoint"] = entrypoint
    return normalized


def _normalize_profile_translators(
    profile_name: str,
    profile_meta: dict[str, object],
    *,
    source_label: str,
) -> None:
    """Normalize translator declarations for one profile."""
    category = str(profile_meta.get("category", "")).strip().lower()
    raw_translators = profile_meta.get("translators")
    if raw_translators is None:
        return
    if category != LANGUAGE_CATEGORY:
        raise ValueError(
            f"{source_label} profile '{profile_name}' declares translators "
            "but is not category: language."
        )
    if not isinstance(raw_translators, list):
        raise ValueError(
            f"{source_label} profile '{profile_name}' must define "
            "translators as a list."
        )
    normalized_entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_entry in raw_translators:
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"{source_label} profile '{profile_name}' has non-mapping "
                "translator entries."
            )
        entry = dict(raw_entry)
        translator_id = str(entry.get("id", "")).strip().lower()
        if not translator_id:
            raise ValueError(
                f"{source_label} profile '{profile_name}' has a translator "
                "without id."
            )
        if translator_id in seen_ids:
            raise ValueError(
                f"{source_label} profile '{profile_name}' has duplicate "
                f"translator id '{translator_id}'."
            )
        seen_ids.add(translator_id)
        entry["id"] = translator_id
        entry["extensions"] = _normalize_translator_extensions(
            entry.get("extensions"),
            profile_name=profile_name,
            translator_id=translator_id,
            source_label=source_label,
        )
        entry["can_handle"] = _normalize_translator_strategy(
            entry.get("can_handle"),
            profile_name=profile_name,
            translator_id=translator_id,
            section="can_handle",
            source_label=source_label,
        )
        entry["translate"] = _normalize_translator_strategy(
            entry.get("translate"),
            profile_name=profile_name,
            translator_id=translator_id,
            section="translate",
            source_label=source_label,
        )
        normalized_entries.append(entry)
    profile_meta["translators"] = normalized_entries


def _normalize_profile_run_events(
    profile_name: str,
    profile_meta: dict[str, object],
    *,
    source_label: str,
) -> None:
    """Normalize run-event adapter declarations for one profile."""
    raw_entries = profile_meta.get("run_events")
    if "test_events" in profile_meta:
        raise ValueError(
            f"{source_label} profile '{profile_name}' must define "
            "`run_events`, not legacy `test_events`."
        )
    if raw_entries is None:
        return
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"{source_label} profile '{profile_name}' must define "
            "run_events as a list."
        )
    normalized_entries: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError(
                f"{source_label} profile '{profile_name}' has "
                "non-mapping run_event entries."
            )
        entry_id = str(raw_entry.get("id") or "").strip().lower()
        if not entry_id:
            raise ValueError(
                f"{source_label} profile '{profile_name}' has a run_event "
                "declaration without id."
            )
        if entry_id in seen_ids:
            raise ValueError(
                f"{source_label} profile '{profile_name}' has duplicate "
                f"run_event id '{entry_id}'."
            )
        entrypoint = str(raw_entry.get("entrypoint") or "").strip()
        if not entrypoint:
            raise ValueError(
                f"{source_label} profile '{profile_name}' run_event "
                f"'{entry_id}' must define an entrypoint."
            )
        config_value = raw_entry.get("config")
        if config_value is not None and not isinstance(config_value, dict):
            raise ValueError(
                f"{source_label} profile '{profile_name}' run_event "
                f"'{entry_id}' config must be a mapping."
            )
        normalized_entry: dict[str, object] = {
            "id": entry_id,
            "entrypoint": entrypoint,
        }
        if config_value is not None:
            normalized_entry["config"] = dict(config_value)
        seen_ids.add(entry_id)
        normalized_entries.append(normalized_entry)
    profile_meta["run_events"] = normalized_entries


def _normalize_profile_clean_overlays(
    profile_name: str,
    profile_meta: dict[str, object],
    *,
    source_label: str,
) -> None:
    """Normalize cleanup overlays declared by one profile manifest."""
    raw_overlays = profile_meta.get("clean_overlays")
    if raw_overlays is None:
        return
    if not isinstance(raw_overlays, dict):
        raise ValueError(
            f"{source_label} profile '{profile_name}' must define "
            "clean_overlays as a mapping."
        )
    normalized: dict[str, list[str]] = {}
    for key in _CLEAN_OVERLAY_KEYS:
        if key not in raw_overlays:
            continue
        raw_value = raw_overlays.get(key)
        if isinstance(raw_value, str):
            entries = [raw_value]
        elif isinstance(raw_value, list):
            entries = raw_value
        else:
            raise ValueError(
                f"{source_label} profile '{profile_name}' clean_overlays."
                f"{key} must be a string or list."
            )
        values: list[str] = []
        for raw_entry in entries:
            token = str(raw_entry or "").strip()
            if token and token not in values:
                values.append(token)
        normalized[key] = values
    profile_meta["clean_overlays"] = normalized


def _normalize_registry_profiles(
    registry: Dict[str, Dict], *, source_label: str
) -> Dict[str, Dict]:
    """Validate and normalize profile registry entries in place."""
    for profile_name, raw_meta in list(registry.items()):
        if not isinstance(raw_meta, dict):
            registry[profile_name] = {}
            continue
        meta = dict(raw_meta)
        category = str(meta.get("category", "")).strip().lower()
        if not category:
            category = "unknown"
        meta["category"] = category
        _normalize_profile_translators(
            profile_name,
            meta,
            source_label=source_label,
        )
        _normalize_profile_run_events(
            profile_name,
            meta,
            source_label=source_label,
        )
        _normalize_profile_clean_overlays(
            profile_name,
            meta,
            source_label=source_label,
        )
        registry[profile_name] = meta
    return registry


def load_profile(manifest_path: Path) -> dict:
    """Load a single profile manifest from disk."""
    return _load_yaml(manifest_path)


def discover_profiles(
    repo_root: Path,
    *,
    builtin_root: Path | None = None,
    custom_root: Path | None = None,
) -> Dict[str, Dict]:
    """Discover profiles from builtin/custom roots."""
    registry: Dict[str, Dict] = {}
    builtin_root = builtin_root or (repo_root / BUILTIN_PROFILE_ROOT)
    custom_root = custom_root or (repo_root / CUSTOM_PROFILE_ROOT)

    for source, root in (
        ("builtin", builtin_root),
        ("custom", custom_root),
    ):
        for entry in _iter_profile_dirs(root):
            manifest_path = _manifest_path(entry)
            if not manifest_path.exists():
                raise ValueError(
                    f"Profile manifest is missing: {manifest_path}"
                )
            manifest = _load_yaml(manifest_path)
            meta = dict(manifest)
            name = _normalize_profile_name(meta.get("profile") or entry.name)
            if not name:
                continue
            meta.setdefault("profile", name)
            if "category" not in meta:
                meta["category"] = (
                    "custom" if source == "custom" else "unknown"
                )
            _validate_profile_asset_manifest(
                entry,
                name,
                meta,
                source_label=str(manifest_path),
            )
            _normalize_profile_translators(
                name,
                meta,
                source_label=str(manifest_path),
            )
            _normalize_profile_run_events(
                name,
                meta,
                source_label=str(manifest_path),
            )
            _normalize_profile_clean_overlays(
                name,
                meta,
                source_label=str(manifest_path),
            )
            meta["source"] = source
            meta["path"] = _relative_path(entry, repo_root)
            meta["assets_available"] = _profile_assets(entry, repo_root)
            registry[name] = meta
    return registry


def build_profile_registry(
    repo_root: Path,
    active_profiles: list[str] | None = None,
    *,
    builtin_root: Path | None = None,
    custom_root: Path | None = None,
) -> Dict[str, Dict]:
    """Build a profile registry payload for registry output."""
    registry = discover_profiles(
        repo_root,
        builtin_root=builtin_root,
        custom_root=custom_root,
    )
    active_names = _active_profile_names(active_profiles or [])
    active = {name for name in active_names}
    for name, meta in registry.items():
        meta["active"] = name in active
    import devcovenant.core.workflow_support as workflow_runtime

    workflow_contract = workflow_runtime.build_workflow_contract(
        repo_root,
        registry,
        active_names,
    )
    return {"profiles": registry, "workflow_contract": workflow_contract}


def write_profile_registry(repo_root: Path, registry: Dict[str, Dict]) -> Path:
    """Write the profile registry into the tracked registry document."""
    path = repo_root / REGISTRY_PROFILE
    payload = load_registry_document(path)
    profiles = registry.get("profiles")
    normalized_profiles = dict(profiles) if isinstance(profiles, dict) else {}
    workflow_contract = registry.get("workflow_contract")
    normalized_workflow_contract = (
        dict(workflow_contract) if isinstance(workflow_contract, dict) else {}
    )
    if path.exists():
        if (
            payload.get("profiles") == normalized_profiles
            and payload.get("workflow_contract")
            == normalized_workflow_contract
        ):
            return path
    payload["profiles"] = normalized_profiles
    payload["workflow_contract"] = normalized_workflow_contract
    write_registry_document(path, payload)
    return path


def refresh_profile_registry(
    repo_root: Path, active_profiles: list[str] | None = None
) -> Dict[str, Dict]:
    """Rebuild and persist profile registry, then return it."""
    registry = build_profile_registry(repo_root, active_profiles or [])
    write_profile_registry(repo_root, registry)
    return registry


def _normalize_registry(registry: Dict[str, Dict]) -> Dict[str, Dict]:
    """Normalize profile registries that include a top-level profiles key."""
    if "profiles" in registry and isinstance(registry["profiles"], dict):
        return _normalize_registry_profiles(
            registry["profiles"], source_label=str(REGISTRY_PROFILE)
        )
    return _normalize_registry_profiles(
        registry, source_label="profile-registry"
    )


def load_profile_registry(repo_root: Path) -> Dict[str, Dict]:
    """Load the merged profile registry from registry or profile roots."""
    registry_path = repo_root / REGISTRY_PROFILE
    if registry_path.exists():
        registry_data = load_registry_document(registry_path)
        if isinstance(registry_data, dict) and registry_data:
            normalized = _normalize_registry(registry_data)
            if normalized:
                return normalized
    return discover_profiles(repo_root)


def list_profiles(registry: Dict[str, Dict]) -> List[str]:
    """Return a sorted list of profile names."""
    normalized = _normalize_registry(registry)
    return sorted(name for name in normalized.keys() if name)


def resolve_profile_suffixes(
    registry: Dict[str, Dict], active_profiles: List[str]
) -> List[str]:
    """Return file suffixes associated with active profiles."""
    normalized_registry = _normalize_registry(registry)
    suffixes: List[str] = []
    for name in _active_profile_names(active_profiles):
        meta = normalized_registry.get(name, {})
        raw = meta.get("suffixes") or []
        for entry in raw:
            suffix_value = str(entry).strip()
            if not suffix_value:
                continue
            suffixes.append(suffix_value)
    return suffixes


def resolve_profile_ignore_dirs(
    registry: Dict[str, Dict], active_profiles: List[str]
) -> List[str]:
    """Return ignored directory names from active profiles."""
    normalized_registry = _normalize_registry(registry)
    ignored: List[str] = []
    for name in _active_profile_names(active_profiles):
        meta = normalized_registry.get(name, {})
        raw = meta.get("ignore_dirs") or []
        for entry in raw:
            dir_value = str(entry).strip()
            if not dir_value:
                continue
            ignored.append(dir_value)
    return ignored


def resolve_profile_clean_overlays(
    registry: Dict[str, Dict], active_profiles: List[str]
) -> Dict[str, List[str]]:
    """Return additive cleanup overlays contributed by active profiles."""
    normalized_registry = _normalize_registry(registry)
    resolved = {key: [] for key in _CLEAN_OVERLAY_KEYS}
    for name in _active_profile_names(active_profiles):
        meta = normalized_registry.get(name, {})
        raw = meta.get("clean_overlays")
        if not isinstance(raw, dict):
            continue
        for key in _CLEAN_OVERLAY_KEYS:
            values = raw.get(key)
            if not isinstance(values, list):
                continue
            for entry in values:
                token = str(entry or "").strip()
                if token and token not in resolved[key]:
                    resolved[key].append(token)
    return resolved
