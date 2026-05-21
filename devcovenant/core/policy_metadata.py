"""Policy metadata parsing and resolution."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import yaml

import devcovenant.core.profile_registry as profile_runtime
import devcovenant.core.repository_paths as project_runtime
from devcovenant.core.selectors import _normalize_globs

if TYPE_CHECKING:
    from devcovenant.core.policy_registry import PolicyDescriptor


def _normalize_policy_metadata_value(raw_value: Any) -> Any:
    """Normalize parsed metadata values to the runtime storage shape."""
    if isinstance(raw_value, dict):
        return {
            str(key).strip(): _normalize_policy_metadata_value(value)
            for key, value in raw_value.items()
            if str(key).strip()
        }
    if isinstance(raw_value, list):
        normalized_list = [
            _normalize_policy_metadata_value(entry) for entry in raw_value
        ]
        if any(isinstance(entry, (dict, list)) for entry in normalized_list):
            return [
                entry for entry in normalized_list if entry not in ("", [], {})
            ]
        return [
            str(entry).strip()
            for entry in normalized_list
            if str(entry).strip()
        ]
    if isinstance(raw_value, bool):
        return "true" if raw_value else "false"
    if raw_value is None:
        return ""
    return str(raw_value).strip()


def parse_policy_metadata_block(
    block: str,
) -> tuple[list[str], dict[str, Any]]:
    """Return ordered keys and typed YAML values from one metadata block."""
    lines = block.splitlines()
    for index, line in enumerate(lines[:-1]):
        match = re.match(r"^(\s*)([A-Za-z0-9_-]+):\s*(.+?)\s*$", line)
        if not match:
            continue
        value = match.group(3).strip()
        if value in {"|", ">", "|-", ">-", "|+", ">+"}:
            continue
        current_indent = len(match.group(1))
        next_line = lines[index + 1]
        if not next_line.strip():
            continue
        next_indent = len(next_line) - len(next_line.lstrip(" "))
        if next_indent > current_indent:
            raise ValueError(
                "Invalid policy metadata YAML: unsupported continuation "
                "shortcut syntax."
            )
    try:
        payload = project_runtime.load_yaml_text(block)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid policy metadata YAML: {exc}") from exc
    if payload is None:
        return [], {}
    if not isinstance(payload, dict):
        raise ValueError("Policy metadata block must be a YAML mapping.")
    order = [str(key).strip() for key in payload.keys() if str(key).strip()]
    values = {
        str(key).strip(): _normalize_policy_metadata_value(value)
        for key, value in payload.items()
        if str(key).strip()
    }
    return order, values


@dataclass
class PolicyDefinition:
    """A policy definition parsed from AGENTS.md."""

    policy_id: str
    name: str
    severity: str
    auto_fix: bool
    enabled: bool
    custom: bool
    description: str
    hash_from_file: Optional[str] = None
    raw_metadata: Dict[str, Any] = field(default_factory=dict)


class PolicyParser:
    """Parse policy definitions from the managed AGENTS policy block."""

    def __init__(self, agents_md_path: Path):
        """Store the AGENTS path used for managed policy parsing."""
        self.agents_md_path = agents_md_path

    def parse_agents_md(self) -> List[PolicyDefinition]:
        """Return policy definitions discovered in AGENTS.md."""
        with open(self.agents_md_path, "r", encoding="utf-8") as file_obj:
            content = file_obj.read()

        policy_block = self._policy_block(content)
        if not policy_block.strip():
            return []

        policies: list[PolicyDefinition] = []
        policy_pattern = re.compile(
            r"##\s+Policy:\s+([^\n]+)\n\n```policy-def\n(.*?)\n```\n\n"
            r"(.*?)(?=\n---\n|\n##|\n<!-- DEVCOV-POLICIES:END -->|\Z)",
            re.DOTALL,
        )
        for match in policy_pattern.finditer(policy_block):
            _, metadata = parse_policy_metadata_block(match.group(2).strip())
            policy_id = self._required_metadata(metadata, "id")
            severity = self._required_metadata(metadata, "severity")
            auto_fix = self._parse_bool_metadata(
                metadata,
                "auto_fix",
                policy_id=policy_id,
            )
            enabled = self._parse_bool_metadata(
                metadata,
                "enabled",
                policy_id=policy_id,
            )
            custom = self._parse_bool_metadata(
                metadata,
                "custom",
                policy_id=policy_id,
            )
            policy = PolicyDefinition(
                policy_id=policy_id,
                name=match.group(1).strip(),
                severity=severity,
                auto_fix=auto_fix,
                enabled=enabled,
                custom=custom,
                description=match.group(3).strip(),
                hash_from_file=metadata.get("hash"),
                raw_metadata=metadata,
            )
            policies.append(policy)
        return policies

    @staticmethod
    def _policy_block(content: str) -> str:
        """Return the text inside the managed AGENTS policy block."""
        begin_marker = "<!-- DEVCOV-POLICIES:BEGIN -->"
        end_marker = "<!-- DEVCOV-POLICIES:END -->"
        try:
            begin = content.index(begin_marker) + len(begin_marker)
            end = content.index(end_marker, begin)
        except ValueError:
            return ""
        return content[begin:end]

    @staticmethod
    def _required_metadata(metadata: Dict[str, Any], key: str) -> str:
        """Return required metadata key value or raise parse error."""
        raw_value = metadata.get(key, "")
        if isinstance(raw_value, list):
            for entry in raw_value:
                raw = str(entry or "").strip()
                if raw:
                    return raw
            raise ValueError(f"Missing required metadata key `{key}`.")
        raw = str(raw_value).strip()
        if raw:
            return raw
        raise ValueError(f"Missing required metadata key `{key}`.")

    @staticmethod
    def _parse_bool_metadata(
        metadata: Dict[str, Any],
        key: str,
        *,
        policy_id: str,
    ) -> bool:
        """Parse strict bool metadata values from policy-def blocks."""
        token = PolicyParser._required_metadata(metadata, key).lower()
        if token == "true":
            return True
        if token == "false":
            return False
        raise ValueError(
            f"Invalid boolean `{key}: {token}` in policy `{policy_id}`."
        )


# fmt: off
_COMMON_KEYS = [
    "id",
    "severity",
    "auto_fix",
    "enforcement",
    "enabled",
    "custom",
]
_COMMON_DEFAULTS: Dict[str, str] = {
    "severity": "warning",
    "auto_fix": "false",
    "enforcement": "active",
    "enabled": "true",
    "custom": "false",
}
_ROLE_SUFFIXES: Tuple[str, ...] = ("globs", "files", "dirs")
_GLOB_SUFFIXES: Tuple[str, ...] = ("prefixes", "suffixes")
_SELECTOR_ROLE_TARGETS = {
    "include_globs": ("include", "globs"),
    "exclude_globs": ("exclude", "globs"),
    "force_include_globs": ("force_include", "globs"),
    "include_files": ("include", "files"),
    "exclude_files": ("exclude", "files"),
    "force_include_files": ("force_include", "files"),
    "include_dirs": ("include", "dirs"),
    "exclude_dirs": ("exclude", "dirs"),
    "force_include_dirs": ("force_include", "dirs"),
    "watch_globs": ("watch", "globs"),
    "watch_files_files": ("watch_files", "files"),
    "watch_files_globs": ("watch_files", "globs"),
    "watch_files_dirs": ("watch_files", "dirs"),
    "tests_watch_globs": ("tests_watch", "globs"),
    "tests_watch_files": ("tests_watch", "files"),
    "tests_watch_dirs": ("tests_watch", "dirs"),
}
_DERIVED_VALUE_KEYS = {"updated"}
_ORDER_EXCLUDE_KEYS = {"updated"}
_TRACE_LAYER_RUNTIME_DEFAULTS = "runtime_defaults"
_TRACE_LAYER_DESCRIPTOR = "descriptor"
_TRACE_LAYER_PROFILE_OVERLAYS = "profile_overlays"
_TRACE_LAYER_AUTOGEN_OVERLAYS = "autogen_overlays"
_TRACE_LAYER_USER_OVERLAYS = "user_overlays"
_TRACE_LAYER_AUTOGEN_OVERRIDES = "autogen_overrides"
_TRACE_LAYER_USER_OVERRIDES = "user_overrides"
_TRACE_LAYER_POLICY_STATE = "policy_state"
_TRACE_LAYER_DERIVED_SELECTORS = "derived_selectors"
_TRACE_LAYER_RUNTIME_IDENTITY = "runtime_identity"
_TRACE_LAYER_RUNTIME_CUSTOM = "runtime_custom_source"


def metadata_value_list(raw_value: object) -> List[str]:
    """Return metadata values as a string list."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        values: List[str] = []
        for item in raw_value:
            if isinstance(item, (dict, list)):
                dumped = yaml.safe_dump(
                    item,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=False,
                ).strip()
                if dumped:
                    values.append(dumped)
                continue
            if str(item):
                values.append(str(item))
        return values
    if isinstance(raw_value, dict):
        dumped = yaml.safe_dump(
            raw_value,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ).strip()
        return [dumped] if dumped else []
    return [str(raw_value)]


def _normalize_metadata_value(raw_value: object) -> Any:
    """Normalize one raw metadata value while preserving structure."""
    if raw_value is None:
        return ""
    if isinstance(raw_value, dict):
        normalized: Dict[str, Any] = {}
        for key, value in raw_value.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            normalized[key_name] = _normalize_metadata_value(value)
        return normalized
    if isinstance(raw_value, list):
        normalized_list: List[Any] = []
        for value in raw_value:
            normalized = _normalize_metadata_value(value)
            if normalized == "" or normalized == [] or normalized == {}:
                continue
            normalized_list.append(normalized)
        return normalized_list
    if isinstance(raw_value, (bool, int, float)):
        return (
            str(raw_value).lower()
            if isinstance(raw_value, bool)
            else str(raw_value)
        )
    return str(raw_value).strip()


def _normalize_layer_value(raw_value: object) -> Any:
    """Normalize overlay/override values while preserving list semantics."""
    return _normalize_metadata_value(raw_value)


def _uses_sequence_semantics(
    inherited_value: Any,
    incoming_value: Any,
) -> bool:
    """Return True when one metadata update should keep list semantics."""

    if isinstance(inherited_value, list):
        return True
    if not isinstance(incoming_value, list):
        return False
    if any(isinstance(entry, (dict, list)) for entry in incoming_value):
        return True
    return len(incoming_value) != 1


@dataclass(frozen=True)
class PolicyControl:
    """Config-driven policy control flags."""

    policy_state: Dict[str, bool]


@dataclass(frozen=True)
class MetadataContext:
    """Resolved metadata context for policy normalization."""

    control: PolicyControl
    profile_overlays: Dict[str, Dict[str, Any]]
    autogen_overlays: Dict[str, Dict[str, Any]]
    user_overlays: Dict[str, Dict[str, Any]]
    autogen_overrides: Dict[str, Dict[str, Any]]
    user_overrides: Dict[str, Dict[str, Any]]


@dataclass(frozen=True)
class ResolvedPolicyMetadata:
    """Canonical resolved metadata views for one policy."""

    order: List[str]
    raw_map: Dict[str, Any]
    list_map: Dict[str, List[str]]
    string_map: Dict[str, str]
    resolution_trace: Dict[str, Dict[str, Any]]
    warnings: List[Dict[str, Any]]

    def decode_options(
        self,
        *,
        reserved_keys: Iterable[str] = (),
    ) -> Dict[str, Any]:
        """Return a typed view of metadata suitable for policy options."""
        return decode_metadata_options_map(
            self.raw_map,
            reserved_keys=reserved_keys,
        )

    def warning_messages(self) -> List[str]:
        """Return human-readable metadata warning messages."""
        messages: List[str] = []
        for warning in self.warnings:
            message = str(warning.get("message", "")).strip()
            if message:
                messages.append(message)
        return messages


def _load_config_payload(repo_root: Path) -> Dict[str, object]:
    """Load config.yaml into a dictionary."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    rendered = project_runtime.display_path(config_path, repo_root=repo_root)
    if not config_path.exists():
        raise ValueError(f"Missing config file: {rendered}")
    try:
        payload = project_runtime.load_yaml(config_path)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {rendered}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Unable to read {rendered}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a YAML mapping: {rendered}")
    return payload


def _load_active_profiles(payload: Dict[str, object]) -> List[str]:
    """Return active profiles from config payload."""
    return profile_runtime.parse_active_profiles(payload, include_global=True)


def _normalize_metadata_values(raw_value: object) -> List[str]:
    """Normalize a metadata value into a list of strings."""
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        text = raw_value.strip()
        if not text:
            return []
        return [text]
    if isinstance(raw_value, list):
        cleaned: List[str] = []
        for entry in raw_value:
            token = str(entry or "").strip()
            if token:
                cleaned.append(token)
        return cleaned
    text = str(raw_value).strip()
    if not text:
        return []
    return [text]


def _normalize_override_map(
    raw_value: object,
) -> Dict[str, Dict[str, Any]]:
    """Normalize policy override maps into typed metadata entries."""
    if not isinstance(raw_value, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for policy_id, mapping in raw_value.items():
        if not isinstance(mapping, dict):
            continue
        policy_key = str(policy_id).strip()
        if not policy_key:
            continue
        entries: Dict[str, Any] = {}
        for key, metadata_value in mapping.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            entries[key_name] = _normalize_layer_value(metadata_value)
        if entries:
            normalized[policy_key] = entries
    return normalized


def _normalize_overlay_map(
    raw_value: object,
) -> Dict[str, Dict[str, Any]]:
    """Normalize metadata overlays into merge/replace-aware entries."""
    if not isinstance(raw_value, dict):
        return {}
    normalized: Dict[str, Dict[str, Any]] = {}
    for policy_id, mapping in raw_value.items():
        if not isinstance(mapping, dict):
            continue
        policy_key = str(policy_id).strip()
        if not policy_key:
            continue
        entries: Dict[str, Any] = {}
        for key, metadata_value in mapping.items():
            key_name = str(key).strip()
            if not key_name:
                continue
            entries[key_name] = _normalize_layer_value(metadata_value)
        if entries:
            normalized[policy_key] = entries
    return normalized


def _load_metadata_layers(
    payload: Dict[str, object],
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
    Dict[str, Dict[str, Any]],
]:
    """Return autogen/user metadata overlays and overrides from config."""
    autogen_overlays = _normalize_overlay_map(
        payload.get("autogen_metadata_overlays")
    )
    user_overlays = _normalize_overlay_map(
        payload.get("user_metadata_overlays")
    )
    autogen = _normalize_override_map(
        payload.get("autogen_metadata_overrides")
    )
    user = _normalize_override_map(payload.get("user_metadata_overrides"))
    return autogen_overlays, user_overlays, autogen, user


def _merge_values(existing: List[str], incoming: List[str]) -> List[str]:
    """Merge values with de-duplication preserving order."""
    existing_values = (
        existing
        if isinstance(existing, list)
        else [str(existing)] if str(existing or "").strip() else []
    )
    incoming_values = (
        incoming
        if isinstance(incoming, list)
        else [str(incoming)] if str(incoming or "").strip() else []
    )
    return _dedupe(existing_values + incoming_values)


def _list_supports_merge_by_id(value: Any) -> bool:
    """Return True when a list contains mapping entries keyed by stable ids."""

    if not isinstance(value, list) or not value:
        return False
    for entry in value:
        normalized = _normalize_metadata_value(entry)
        if not isinstance(normalized, dict):
            return False
        if not str(normalized.get("id", "")).strip():
            return False
    return True


def _merge_mapping_lists_by_id(
    existing: Sequence[object],
    incoming: Sequence[object],
) -> List[Any]:
    """Merge structured mapping lists by stable `id` while preserving order."""
    merged: List[Any] = []
    index_by_id: Dict[str, int] = {}
    for entry in list(existing) + list(incoming):
        normalized = _normalize_metadata_value(entry)
        if not isinstance(normalized, dict):
            if normalized not in merged:
                merged.append(normalized)
            continue
        surface_id = str(normalized.get("id", "")).strip()
        if not surface_id:
            merged.append(normalized)
            continue
        if surface_id not in index_by_id:
            index_by_id[surface_id] = len(merged)
            merged.append(normalized)
            continue
        current = merged[index_by_id[surface_id]]
        if not isinstance(current, dict):
            merged[index_by_id[surface_id]] = normalized
            continue
        updated = dict(current)
        for key, value in normalized.items():
            if key == "id":
                updated[key] = value
                continue
            if isinstance(updated.get(key), list) and isinstance(value, list):
                updated[key] = _merge_metadata_values(
                    key,
                    updated.get(key, []),
                    value,
                )
                continue
            updated[key] = value
        merged[index_by_id[surface_id]] = updated
    return merged


def _merge_metadata_values(key: str, existing: Any, incoming: Any) -> Any:
    """Merge metadata values using key-aware behavior."""
    del key
    existing_normalized = _normalize_metadata_value(existing)
    incoming_normalized = _normalize_metadata_value(incoming)
    if _list_supports_merge_by_id(incoming_normalized):
        if not isinstance(existing_normalized, list):
            return list(incoming_normalized)
        if existing_normalized and not _list_supports_merge_by_id(
            existing_normalized
        ):
            return list(incoming_normalized)
        return _merge_mapping_lists_by_id(
            existing_normalized,
            incoming_normalized,
        )
    if _list_supports_merge_by_id(existing_normalized):
        return existing_normalized
    if isinstance(incoming, list):
        existing_list = (
            existing
            if isinstance(existing, list)
            else [str(existing)] if str(existing or "").strip() else []
        )
        return _merge_values(
            [str(entry) for entry in existing_list],
            [str(entry) for entry in incoming],
        )
    return incoming


def _collect_profile_overlays(
    repo_root: Path,
    active_profiles: List[str],
    *,
    profile_registry: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Collect policy overlays from the profile registry."""
    raw_registry = (
        profile_registry
        if isinstance(profile_registry, dict)
        else profile_runtime.load_profile_registry(repo_root)
    )
    if not isinstance(raw_registry, dict):
        return {}
    registry = (
        profile_runtime._normalize_registry(raw_registry)
        if "profiles" in raw_registry
        else raw_registry
    )
    overlays: Dict[str, Dict[str, Any]] = {}
    for profile in active_profiles:
        meta = registry.get(profile)
        if not isinstance(meta, dict):
            continue
        for section_name in ("policy_overlays",):
            raw_overlays = meta.get(section_name) or {}
            if not isinstance(raw_overlays, dict):
                continue
            for policy_id, overlay in raw_overlays.items():
                if not isinstance(overlay, dict):
                    continue
                policy_key = str(policy_id).strip()
                if not policy_key:
                    continue
                policy_map = overlays.setdefault(policy_key, {})
                for key, raw_value in overlay.items():
                    key_name = str(key).strip()
                    if not key_name:
                        continue
                    value = _normalize_layer_value(raw_value)
                    if key_name in policy_map and _uses_sequence_semantics(
                        policy_map.get(key_name, []),
                        value,
                    ):
                        policy_map[key_name] = _merge_metadata_values(
                            key_name,
                            policy_map.get(key_name, []),
                            value,
                        )
                        continue
                    policy_map[key_name] = value
    return overlays


def collect_profile_overlays(
    repo_root: Path,
    active_profiles: List[str],
    *,
    profile_registry: Dict[str, Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """Public wrapper for resolved profile policy overlays."""
    return _collect_profile_overlays(
        repo_root,
        active_profiles,
        profile_registry=profile_registry,
    )


def _normalize_policy_state(raw_value: object) -> Dict[str, bool]:
    """Normalize policy_state config into a boolean map."""
    if not isinstance(raw_value, dict):
        return {}
    normalized: Dict[str, bool] = {}
    for policy_id, enabled_value in raw_value.items():
        key = str(policy_id or "").strip()
        if not key:
            continue
        if isinstance(enabled_value, bool):
            normalized[key] = enabled_value
            continue
        token = str(enabled_value).strip().lower()
        if token in {"true", "1", "yes", "y", "on"}:
            normalized[key] = True
        elif token in {"false", "0", "no", "n", "off"}:
            normalized[key] = False
    return normalized


def normalize_policy_state(raw_value: object) -> Dict[str, bool]:
    """Public wrapper for policy_state normalization."""
    return _normalize_policy_state(raw_value)


def load_policy_control_config(payload: Dict[str, object]) -> PolicyControl:
    """Load policy control values for policies."""
    policy_state = _normalize_policy_state(payload.get("policy_state"))
    return PolicyControl(policy_state)


def build_metadata_context(repo_root: Path) -> MetadataContext:
    """Return the metadata resolution context for a repo."""
    payload = _load_config_payload(repo_root)
    return build_metadata_context_from_payload(repo_root, payload)


def build_metadata_context_from_payload(
    repo_root: Path,
    payload: Dict[str, object],
) -> MetadataContext:
    """Return the metadata resolution context for an in-memory config."""
    active_profiles = _load_active_profiles(payload)
    profile_overlays = _collect_profile_overlays(repo_root, active_profiles)
    (
        autogen_overlays,
        user_overlays,
        autogen_overrides,
        user_overrides,
    ) = _load_metadata_layers(payload)
    control = load_policy_control_config(payload)
    return MetadataContext(
        control=control,
        profile_overlays=profile_overlays,
        autogen_overlays=autogen_overlays,
        user_overlays=user_overlays,
        autogen_overrides=autogen_overrides,
        user_overrides=user_overrides,
    )


def _ensure_metadata_key(
    order: List[str],
    values: Dict[str, Any],
    key: str,
) -> None:
    """Ensure a metadata key exists in order and values."""
    if key not in values:
        values[key] = []
    if key not in order:
        order.append(key)


def _first_metadata_token(
    values: Dict[str, Any],
    key: str,
) -> str:
    """Return the first normalized metadata token for a key."""
    raw_value = values.get(key, [])
    if isinstance(raw_value, list):
        if not raw_value:
            return ""
        return str(raw_value[0] or "").strip().lower()
    if isinstance(raw_value, (dict, tuple, set)):
        return ""
    return str(raw_value or "").strip().lower()


def apply_policy_control(
    order: List[str],
    values: Dict[str, Any],
    policy_id: str,
    control: PolicyControl,
) -> Tuple[List[str], Dict[str, Any]]:
    """Apply enabled controls to metadata values."""
    if policy_id in control.policy_state:
        requested_enabled = bool(control.policy_state[policy_id])
        severity_token = _first_metadata_token(values, "severity")
        if severity_token == "critical" and not requested_enabled:
            requested_enabled = True
        _ensure_metadata_key(order, values, "enabled")
        values["enabled"] = "true" if requested_enabled else "false"
    return order, values


def descriptor_metadata_order_values(
    descriptor: PolicyDescriptor,
) -> Tuple[List[str], Dict[str, Any]]:
    """Return ordered keys and normalized values from a descriptor."""
    order = list(descriptor.metadata.keys())
    values: Dict[str, Any] = {}
    for key in order:
        values[key] = _normalize_metadata_value(descriptor.metadata.get(key))
    return order, values


def _split_values(raw_values: Sequence[str]) -> List[str]:
    """Return a flattened list of comma-separated values."""
    items: List[str] = []
    for entry in raw_values:
        for part in entry.split(","):
            token = part.strip()
            if token:
                items.append(token)
    return items


def split_metadata_values(raw_values: Sequence[str]) -> List[str]:
    """Public wrapper for metadata value splitting."""
    return _split_values(raw_values)


def decode_metadata_option_value(raw_value: object) -> Any:
    """Decode one metadata value into a common scalar/list representation."""
    if raw_value is None:
        return ""
    if isinstance(raw_value, Mapping):
        decoded: Dict[str, Any] = {}
        for raw_key, nested_value in raw_value.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            decoded[key] = decode_metadata_option_value(nested_value)
        return decoded
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        return raw_value
    if isinstance(raw_value, (list, tuple, set)):
        items: List[Any] = []
        for entry in raw_value:
            decoded = decode_metadata_option_value(entry)
            if decoded == "" or decoded == [] or decoded == {}:
                continue
            if isinstance(decoded, list):
                items.extend(decoded)
                continue
            if isinstance(decoded, str) and "," in decoded:
                items.extend(_split_values([decoded]))
                continue
            items.append(decoded)
        return items

    text = str(raw_value).strip()
    if not text:
        return ""

    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"

    if "," in text:
        return _split_values([text])

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        pass

    return text


def decode_metadata_options_map(
    raw_metadata: Mapping[str, object] | None,
    *,
    reserved_keys: Iterable[str] = (),
) -> Dict[str, Any]:
    """Decode a metadata map into typed policy/runtime options."""
    if not isinstance(raw_metadata, Mapping):
        return {}
    reserved = {
        str(key).strip().lower() for key in reserved_keys if str(key).strip()
    }
    decoded: Dict[str, Any] = {}
    for raw_key, raw_value in raw_metadata.items():
        key = str(raw_key or "").strip()
        if not key:
            continue
        if key.lower() in reserved:
            continue
        decoded[key] = decode_metadata_option_value(raw_value)
    return decoded


def _dedupe(values: Iterable[str]) -> List[str]:
    """Return unique values while preserving order."""
    seen: set[str] = set()
    ordered: List[str] = []
    for entry in values:
        if entry in seen:
            continue
        seen.add(entry)
        ordered.append(entry)
    return ordered


def _trace_bucket(
    trace: Dict[str, Dict[str, Any]],
    key: str,
) -> Dict[str, Any]:
    """Return the trace bucket for one metadata key."""
    return trace.setdefault(key, {})


def _trace_value_list(raw_value: object) -> List[str]:
    """Render one metadata value into stable trace strings."""
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        if raw_value and all(
            not isinstance(item, (dict, list)) for item in raw_value
        ):
            return [str(entry) for entry in raw_value if str(entry)]
        dumped = yaml.safe_dump(
            raw_value,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ).strip()
        return [dumped] if dumped else []
    if isinstance(raw_value, dict):
        dumped = yaml.safe_dump(
            raw_value,
            sort_keys=False,
            default_flow_style=False,
            allow_unicode=False,
        ).strip()
        return [dumped] if dumped else []
    return [str(raw_value)] if str(raw_value) else []


def _record_trace_layer(
    trace: Dict[str, Dict[str, Any]],
    key: str,
    *,
    layer: str,
    values: object,
    behavior: str,
    replaced_inherited_values: object = (),
    note: str = "",
) -> None:
    """Record one resolution-layer contribution for a metadata key."""
    bucket = _trace_bucket(trace, key)
    payload: Dict[str, Any] = {
        "values": _trace_value_list(values),
        "behavior": behavior,
    }
    replaced = _trace_value_list(replaced_inherited_values)
    if replaced:
        payload["replaced_inherited_values"] = replaced
    if note.strip():
        payload["note"] = note.strip()
    bucket[layer] = payload


def _record_effective_trace(
    trace: Dict[str, Dict[str, Any]],
    key: str,
    values: object,
) -> None:
    """Record the final effective values for one metadata key."""
    bucket = _trace_bucket(trace, key)
    bucket["effective"] = {"values": _trace_value_list(values)}


def _build_override_warning(
    policy_id: str,
    key: str,
    *,
    layer: str,
    inherited_values: object,
    replacement_values: object,
) -> Dict[str, Any]:
    """Build one structured override-replacement warning payload."""
    inherited = _trace_value_list(inherited_values)
    replacement = _trace_value_list(replacement_values)
    return {
        "policy_id": policy_id,
        "key": key,
        "layer": layer,
        "inherited_values": inherited,
        "replacement_values": replacement,
        "message": (
            f"{layer} replaces inherited metadata for "
            f"`{policy_id}.{key}`; use overlays if you intended additive "
            "behavior."
        ),
    }


def _convert_prefixes(prefixes: Iterable[str]) -> List[str]:
    """Convert prefixes into glob patterns."""
    globs: List[str] = []
    for prefix in prefixes:
        cleaned = prefix.strip().strip("/")
        if cleaned:
            globs.append(f"{cleaned}/**")
    return globs


def _convert_suffixes(suffixes: Iterable[str]) -> List[str]:
    """Convert suffixes into glob patterns."""
    globs: List[str] = []
    for suffix in suffixes:
        cleaned = suffix.strip()
        if not cleaned:
            continue
        if cleaned.startswith("."):
            globs.append(f"*{cleaned}")
            continue
        globs.append(f"*.{cleaned}")
    return globs


def _role_from_key(key: str) -> Tuple[str, str] | None:
    """Return (role, target) for selector-ish metadata keys."""
    if key in _SELECTOR_ROLE_TARGETS:
        return _SELECTOR_ROLE_TARGETS[key]
    for suffix in _ROLE_SUFFIXES:
        marker = f"_{suffix}"
        if key.endswith(marker):
            return key[: -len(marker)], suffix
    for suffix in _GLOB_SUFFIXES:
        marker = f"_{suffix}"
        if key.endswith(marker):
            return key[: -len(marker)], "globs"
    return None


def _apply_selector_roles(
    order: List[str],
    values: Dict[str, Any],
) -> Tuple[List[str], Dict[str, Any]]:
    """Insert selector role keys and normalize selector values."""
    roles: List[str] = []
    if "selector_roles" in values:
        selector_roles = values["selector_roles"]
        if isinstance(selector_roles, list):
            roles = _split_values([str(entry) for entry in selector_roles])
        elif selector_roles:
            roles = _split_values([str(selector_roles)])
    role_values: Dict[str, Dict[str, List[str]]] = {}
    for key, raw_values in values.items():
        if key == "selector_roles":
            continue
        role_info = _role_from_key(key)
        if not role_info:
            continue
        if isinstance(raw_values, (dict, list)) and (
            isinstance(raw_values, dict)
            or any(isinstance(item, (dict, list)) for item in raw_values)
        ):
            continue
        role, target = role_info
        if role not in roles:
            roles.append(role)
        bucket = role_values.setdefault(
            role, {"globs": [], "files": [], "dirs": []}
        )
        if isinstance(raw_values, list):
            items = _split_values([str(entry) for entry in raw_values])
        else:
            items = _split_values([str(raw_values)])
        if key.endswith("_prefixes"):
            items = _convert_prefixes(items)
        elif key.endswith("_suffixes"):
            items = _convert_suffixes(items)
        if key in _SELECTOR_ROLE_TARGETS:
            items = _normalize_globs(items)
        bucket[target] = _merge_values(bucket[target], items)
    if roles and "selector_roles" not in values:
        values["selector_roles"] = [",".join(roles)]
        order.append("selector_roles")
    new_order = list(order)
    if "selector_roles" in new_order:
        insert_at = new_order.index("selector_roles") + 1
    else:
        insert_at = len(new_order)
    for role in roles:
        for suffix in _ROLE_SUFFIXES:
            key = f"{role}_{suffix}"
            if key not in values:
                values[key] = []
            if role in role_values:
                values[key] = _merge_values(
                    values[key], role_values[role][suffix]
                )
            if key not in new_order:
                new_order.insert(insert_at, key)
                insert_at += 1
    return new_order, values


def _apply_overrides_replace(
    values: Dict[str, Any],
    overrides: Dict[str, Any],
    *,
    policy_id: str,
    layer_name: str,
    trace: Dict[str, Dict[str, Any]],
    warnings: List[Dict[str, Any]],
) -> None:
    """Apply override values by replacing existing entries."""
    for key, override_values in overrides.items():
        inherited_values = values.get(key, [])
        values[key] = override_values
        _record_trace_layer(
            trace,
            key,
            layer=layer_name,
            values=values[key],
            behavior="replace",
            replaced_inherited_values=inherited_values,
        )
        if inherited_values and inherited_values != values[key]:
            warnings.append(
                _build_override_warning(
                    policy_id,
                    key,
                    layer=layer_name,
                    inherited_values=inherited_values,
                    replacement_values=values[key],
                )
            )


def _apply_profile_overlays(
    values: Dict[str, Any],
    overlays: Dict[str, Any],
    *,
    layer_name: str,
    trace: Dict[str, Dict[str, Any]],
) -> None:
    """Apply profile overlays, merging list values and replacing scalars."""
    for key, overlay_values in overlays.items():
        inherited_values = values.get(key, [])
        if _uses_sequence_semantics(
            inherited_values,
            overlay_values,
        ):
            values[key] = _merge_metadata_values(
                key,
                values.get(key, []),
                overlay_values,
            )
            _record_trace_layer(
                trace,
                key,
                layer=layer_name,
                values=overlay_values,
                behavior="append",
            )
            continue
        values[key] = overlay_values
        _record_trace_layer(
            trace,
            key,
            layer=layer_name,
            values=values[key],
            behavior="replace",
            replaced_inherited_values=inherited_values,
        )


def _strip_derived_values(values: Dict[str, Any]) -> None:
    """Remove derived metadata values before recomputing."""
    for key in _DERIVED_VALUE_KEYS:
        values.pop(key, None)


def _resolve_metadata(
    policy_id: str,
    current_order: List[str],
    current_values: Dict[str, Any],
    descriptor: PolicyDescriptor | None,
    context: MetadataContext,
    *,
    custom_policy: bool = False,
) -> Tuple[
    List[str],
    Dict[str, Any],
    Dict[str, Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Resolve metadata using defaults, overlays, and config overrides."""
    trace: Dict[str, Dict[str, Any]] = {}
    warnings: List[Dict[str, Any]] = []
    if descriptor:
        base_order, base_values = descriptor_metadata_order_values(descriptor)
        base_order = [
            key for key in base_order if key not in _ORDER_EXCLUDE_KEYS
        ]
        values = {key: base_values.get(key) for key in base_values}
        for key in base_order:
            _record_trace_layer(
                trace,
                key,
                layer=_TRACE_LAYER_DESCRIPTOR,
                values=values.get(key, []),
                behavior="base",
            )
    else:
        base_order = [
            key for key in current_order if key not in _ORDER_EXCLUDE_KEYS
        ]
        values = {key: current_values.get(key) for key in current_values}
        for key in base_order:
            _record_trace_layer(
                trace,
                key,
                layer=_TRACE_LAYER_DESCRIPTOR,
                values=values.get(key, []),
                behavior="base",
            )
    if not descriptor:
        for key in current_order:
            if key in _ORDER_EXCLUDE_KEYS:
                continue
            values.setdefault(key, current_values.get(key, []))

    overlays = context.profile_overlays.get(policy_id, {})
    _apply_profile_overlays(
        values,
        overlays,
        layer_name=_TRACE_LAYER_PROFILE_OVERLAYS,
        trace=trace,
    )
    autogen_overlays = context.autogen_overlays.get(policy_id, {})
    _apply_profile_overlays(
        values,
        autogen_overlays,
        layer_name=_TRACE_LAYER_AUTOGEN_OVERLAYS,
        trace=trace,
    )
    user_overlays = context.user_overlays.get(policy_id, {})
    _apply_profile_overlays(
        values,
        user_overlays,
        layer_name=_TRACE_LAYER_USER_OVERLAYS,
        trace=trace,
    )
    autogen_overrides = context.autogen_overrides.get(policy_id, {})
    _apply_overrides_replace(
        values,
        autogen_overrides,
        policy_id=policy_id,
        layer_name=_TRACE_LAYER_AUTOGEN_OVERRIDES,
        trace=trace,
        warnings=warnings,
    )
    user_overrides = context.user_overrides.get(policy_id, {})
    _apply_overrides_replace(
        values,
        user_overrides,
        policy_id=policy_id,
        layer_name=_TRACE_LAYER_USER_OVERRIDES,
        trace=trace,
        warnings=warnings,
    )
    _strip_derived_values(values)

    ordered_keys: List[str] = []
    for key in _COMMON_KEYS:
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in base_order:
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in overlays.keys():
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in autogen_overlays.keys():
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in user_overlays.keys():
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in autogen_overrides.keys():
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    for key in user_overrides.keys():
        if key in _ORDER_EXCLUDE_KEYS:
            continue
        _ensure_metadata_key(ordered_keys, values, key)
    if not descriptor:
        for key in current_order:
            if key in _ORDER_EXCLUDE_KEYS:
                continue
            _ensure_metadata_key(ordered_keys, values, key)

    values["id"] = policy_id
    _record_trace_layer(
        trace,
        "id",
        layer=_TRACE_LAYER_RUNTIME_IDENTITY,
        values=policy_id,
        behavior="replace",
    )
    if custom_policy:
        values["custom"] = "true"
        _record_trace_layer(
            trace,
            "custom",
            layer=_TRACE_LAYER_RUNTIME_CUSTOM,
            values="true",
            behavior="replace",
            note="Resolved from active custom policy script.",
        )

    for key in ordered_keys:
        current = values.get(key, [])
        if current:
            if isinstance(current, list):
                if _list_supports_merge_by_id(current):
                    values[key] = _merge_metadata_values(key, [], current)
                elif all(
                    not isinstance(entry, (dict, list)) for entry in current
                ):
                    values[key] = _dedupe([str(entry) for entry in current])
                else:
                    values[key] = list(current)
            else:
                values[key] = current
            continue
        if key in _COMMON_DEFAULTS:
            values[key] = _COMMON_DEFAULTS[key]
            _record_trace_layer(
                trace,
                key,
                layer=_TRACE_LAYER_RUNTIME_DEFAULTS,
                values=values[key],
                behavior="default",
            )
            continue
        values[key] = []

    control_requested = context.control.policy_state.get(policy_id)
    pre_control_enabled = values.get("enabled", [])
    severity_token = _first_metadata_token(values, "severity")
    ordered_keys, values = apply_policy_control(
        ordered_keys,
        values,
        policy_id,
        context.control,
    )
    if control_requested is not None:
        control_note = ""
        if severity_token == "critical" and not bool(control_requested):
            control_note = (
                "Critical policy disable attempt preserved enforcement."
            )
        _record_trace_layer(
            trace,
            "enabled",
            layer=_TRACE_LAYER_POLICY_STATE,
            values=["true" if bool(control_requested) else "false"],
            behavior="replace",
            replaced_inherited_values=pre_control_enabled,
            note=control_note,
        )

    pre_selector_values = {
        key: (
            list(entries)
            if isinstance(entries, list)
            else dict(entries) if isinstance(entries, dict) else entries
        )
        for key, entries in values.items()
    }
    ordered_keys, values = _apply_selector_roles(ordered_keys, values)
    for key, resolved_values in values.items():
        previous_values = pre_selector_values.get(key, [])
        if resolved_values == previous_values:
            continue
        if isinstance(resolved_values, list) and isinstance(
            previous_values,
            list,
        ):
            derived_values = [
                entry
                for entry in resolved_values
                if entry not in previous_values
            ]
        else:
            derived_values = resolved_values
        _record_trace_layer(
            trace,
            key,
            layer=_TRACE_LAYER_DERIVED_SELECTORS,
            values=derived_values or resolved_values,
            behavior="derive",
        )
    for key in ordered_keys:
        _record_effective_trace(trace, key, values.get(key, []))
    return ordered_keys, values, trace, warnings


def resolve_policy_metadata_map(
    policy_id: str,
    current_order: List[str],
    current_values: Dict[str, Any],
    descriptor: PolicyDescriptor | None,
    context: MetadataContext,
    *,
    custom_policy: bool = False,
) -> Tuple[List[str], Dict[str, str]]:
    """Return resolved metadata order and string map for a policy."""
    bundle = resolve_policy_metadata_bundle(
        policy_id,
        current_order,
        current_values,
        descriptor,
        context,
        custom_policy=custom_policy,
    )
    return bundle.order, bundle.string_map


def resolve_policy_metadata_bundle(
    policy_id: str,
    current_order: List[str],
    current_values: Dict[str, Any],
    descriptor: PolicyDescriptor | None,
    context: MetadataContext,
    *,
    custom_policy: bool = False,
) -> ResolvedPolicyMetadata:
    """Return resolved metadata in list and string forms."""
    order, values, trace, warnings = _resolve_metadata(
        policy_id,
        current_order,
        current_values,
        descriptor,
        context,
        custom_policy=custom_policy,
    )
    raw_map: Dict[str, Any] = {}
    list_map: Dict[str, List[str]] = {}
    string_map: Dict[str, str] = {}
    for key in order:
        raw_value = values.get(key, [])
        raw_map[key] = raw_value
        entries = metadata_value_list(raw_value)
        list_map[key] = entries
        if isinstance(raw_value, (dict, list)) and (
            isinstance(raw_value, dict)
            or any(isinstance(item, (dict, list)) for item in raw_value)
        ):
            string_map[key] = yaml.safe_dump(
                raw_value,
                sort_keys=False,
                default_flow_style=False,
                allow_unicode=False,
            ).strip()
        else:
            string_map[key] = ", ".join(entry for entry in entries if entry)
    return ResolvedPolicyMetadata(
        order=list(order),
        raw_map=raw_map,
        list_map=list_map,
        string_map=string_map,
        resolution_trace=trace,
        warnings=warnings,
    )


def render_metadata_block(keys: Iterable[str], values: Dict[str, Any]) -> str:
    """Render a policy-def block from ordered keys and values."""
    ordered: Dict[str, Any] = {}
    for key in keys:
        ordered[key] = values.get(key, [])
    return yaml.safe_dump(
        ordered,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=False,
    ).rstrip()
# fmt: on
