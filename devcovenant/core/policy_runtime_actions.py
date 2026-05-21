"""Policy-owned runtime action execution helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml

import devcovenant.core.policy_commands as policy_commands
import devcovenant.core.policy_registry as policy_registry
import devcovenant.core.repository_paths as repository_paths
import devcovenant.core.tracked_registry as tracked_registry
from devcovenant.core.policy_contract import CheckContext, PolicyCheck
from devcovenant.core.policy_metadata import decode_metadata_options_map


def load_policy_check_instance(
    repo_root: Path, policy_id: str
) -> PolicyCheck | None:
    """Load one policy script and return its `PolicyCheck` instance."""
    repo_root = Path(repo_root).resolve()
    location = policy_registry.resolve_script_location(repo_root, policy_id)
    if location is None:
        return None

    spec = importlib.util.spec_from_file_location(
        location.module, location.path
    )
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, PolicyCheck)
            and attr is not PolicyCheck
        ):
            return attr()
    return None


def runtime_policy_config_overrides(
    repo_root: Path, policy_id: str
) -> dict[str, Any]:
    """Return merged config overrides for one policy runtime action."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = repository_paths.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(payload, dict):
        return {}
    context = CheckContext(repo_root=repo_root, config=payload)
    return context.get_policy_config(policy_id)


def _option_value_is_empty(candidate: Any) -> bool:
    """Return True when a runtime option value is an empty placeholder."""
    if candidate is None:
        return True
    if isinstance(candidate, str):
        return candidate.strip() == ""
    if isinstance(candidate, dict):
        return not candidate
    if isinstance(candidate, (list, tuple, set)):
        if not candidate:
            return True
        return all(not str(item).strip() for item in candidate)
    return False


def build_runtime_policy_option_views(
    metadata_options: Mapping[str, Any] | None,
    config_overrides: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Build typed runtime option views for registry/debug inspection."""
    metadata = dict(metadata_options or {})
    overrides = dict(config_overrides or {})
    effective: dict[str, Any] = {}
    for key in list(metadata.keys()) + list(overrides.keys()):
        if key in effective:
            continue
        if key in overrides and not _option_value_is_empty(overrides[key]):
            effective[key] = overrides[key]
            continue
        if key in metadata and not _option_value_is_empty(metadata[key]):
            effective[key] = metadata[key]
    return {
        "runtime_metadata_options": metadata,
        "runtime_config_overrides": overrides,
        "runtime_effective_options": effective,
    }


def runtime_policy_metadata_options(
    repo_root: Path,
    policy_id: str,
    *,
    descriptor_loader: Callable[[Path, str], object | None] = (
        policy_registry.load_policy_descriptor
    ),
    registry_path_resolver: Callable[
        [Path], Path
    ] = tracked_registry.policy_registry_path,
) -> dict[str, Any]:
    """Return decoded runtime metadata options for one policy action."""
    registry_path = registry_path_resolver(repo_root)
    if registry_path.exists():
        try:
            registry_payload = repository_paths.load_yaml(registry_path)
        except (OSError, yaml.YAMLError):
            registry_payload = None
        if isinstance(registry_payload, dict):
            policies = registry_payload.get("policies")
            if isinstance(policies, dict):
                entry = policies.get(policy_id)
                if isinstance(entry, dict):
                    typed_metadata = entry.get("runtime_metadata_options")
                    if isinstance(typed_metadata, dict):
                        return dict(typed_metadata)
                    metadata = entry.get("metadata")
                    if isinstance(metadata, dict):
                        return decode_metadata_options_map(metadata)
    descriptor = descriptor_loader(repo_root, policy_id)
    descriptor_metadata = getattr(descriptor, "metadata", None)
    if isinstance(descriptor_metadata, dict):
        return decode_metadata_options_map(descriptor_metadata)
    return {}


def run_policy_runtime_action(
    repo_root: Path,
    *,
    policy_id: str,
    action: str,
    payload: dict[str, Any] | None = None,
    checker_loader: Callable[[Path, str], PolicyCheck | None] = (
        load_policy_check_instance
    ),
    metadata_loader: Callable[[Path, str], dict[str, Any]] = (
        runtime_policy_metadata_options
    ),
    config_loader: Callable[[Path, str], dict[str, Any]] = (
        runtime_policy_config_overrides
    ),
    action_validator: Callable[..., None] = (
        policy_commands.validate_runtime_action_declared
    ),
) -> Any:
    """Run one policy-owned runtime action through the policy contract."""
    repo_root = Path(repo_root).resolve()
    action_validator(
        repo_root,
        policy_id=policy_id,
        action=action,
    )
    checker = checker_loader(repo_root, policy_id)
    if checker is None:
        raise ValueError(
            f"Policy script not found for runtime action: `{policy_id}`."
        )
    checker.set_options(
        metadata_loader(repo_root, policy_id),
        config_loader(repo_root, policy_id),
    )
    return checker.run_runtime_action(
        action,
        repo_root=repo_root,
        payload=payload or {},
    )
