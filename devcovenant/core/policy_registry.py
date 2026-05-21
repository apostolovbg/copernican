"""Policy descriptor and tracked policy registry services."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import yaml

import devcovenant.core.repository_paths as repository_paths
import devcovenant.core.tracked_registry as tracked_registry
from devcovenant.core.repository_paths import require_repo_relative_path

if TYPE_CHECKING:
    from devcovenant.core.policy_metadata import PolicyDefinition


DEV_COVENANT_DIR = tracked_registry.DEV_COVENANT_DIR
POLICY_BLOCK_RE = re.compile(
    r"(##\s+Policy:\s+[^\n]+\n\n)```policy-def\n(.*?)\n```\n\n"
    r"(.*?)(?=\n---\n|\n##|\Z)",
    re.DOTALL,
)


@dataclass(frozen=True)
class PolicyScriptLocation:
    """Resolved policy script location."""

    kind: str
    path: Path
    module: str


@dataclass
class PolicyDescriptor:
    """Metadata descriptor shipped with a policy."""

    policy_id: str
    text: str
    metadata: Dict[str, object]
    runtime_actions: List[object] | None = None
    commands: List[object] | None = None


def _script_name(policy_id: str) -> str:
    """Return the Python module name for a policy id."""
    return policy_id.replace("-", "_")


def iter_script_locations(
    repo_root: Path,
    policy_id: str,
) -> Iterable[PolicyScriptLocation]:
    """Yield candidate policy script locations in priority order."""
    script_name = _script_name(policy_id)
    devcov_dir = repo_root / DEV_COVENANT_DIR
    candidates = [
        (
            "custom",
            devcov_dir
            / "custom"
            / "policies"
            / script_name
            / f"{script_name}.py",
            f"devcovenant.custom.policies.{script_name}.{script_name}",
        ),
        (
            "builtin",
            devcov_dir
            / "builtin"
            / "policies"
            / script_name
            / f"{script_name}.py",
            f"devcovenant.builtin.policies.{script_name}.{script_name}",
        ),
    ]
    for kind, path, module in candidates:
        yield PolicyScriptLocation(kind=kind, path=path, module=module)


def resolve_script_location(
    repo_root: Path, policy_id: str
) -> PolicyScriptLocation | None:
    """Return the first existing policy script location, if any."""
    for location in iter_script_locations(repo_root, policy_id):
        if location.path.exists():
            return location
    return None


def load_policy_descriptor(
    repo_root: Path, policy_id: str
) -> Optional[PolicyDescriptor]:
    """Return the descriptor for a policy if it exists."""
    for location in iter_script_locations(repo_root, policy_id):
        descriptor_path = location.path.with_suffix(".yaml")
        if not descriptor_path.exists():
            continue
        rendered = repository_paths.display_path(
            descriptor_path,
            repo_root=repo_root,
        )
        try:
            contents = repository_paths.load_yaml(descriptor_path)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML in policy descriptor {rendered}: {exc}"
            ) from exc
        if not isinstance(contents, dict):
            raise ValueError(
                "Policy descriptor must contain a YAML mapping: " f"{rendered}"
            )
        descriptor_id = contents.get("id", policy_id)
        text = contents.get("text", "")
        metadata = contents.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
        runtime_actions = contents.get("runtime_actions", [])
        if not isinstance(runtime_actions, list):
            runtime_actions = []
        commands = contents.get("commands", [])
        if not isinstance(commands, list):
            commands = []
        return PolicyDescriptor(
            policy_id=descriptor_id,
            text=text,
            metadata=metadata,
            runtime_actions=runtime_actions,
            commands=commands,
        )
    return None


@dataclass
class PolicySyncIssue:
    """One tracked policy/script synchronization issue."""

    policy_id: str
    policy_text: str
    policy_hash: str
    script_path: Path
    script_exists: bool
    issue_type: str
    current_hash: Optional[str] = None


class PolicyRegistry:
    """Manage the tracked policy registry."""

    def __init__(self, registry_path: Path, repo_root: Path):
        """Initialize the tracked policy registry."""
        self.registry_path = registry_path
        self.repo_root = repo_root
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self):
        """Load the registry from disk."""
        self._data = tracked_registry.load_registry_document(
            self.registry_path
        )

    def _normalize_registry_hashes(self) -> None:
        """Normalize stored hashes to string form."""
        policies = self._data.get("policies", {})
        for policy_data in policies.values():
            raw_hash = policy_data.get("hash")
            normalized = self._normalize_hash_value(raw_hash)
            if normalized:
                policy_data["hash"] = normalized

    def save(self):
        """Save the registry to disk."""
        self._normalize_registry_hashes()
        tracked_registry.write_registry_document(
            self.registry_path, self._data
        )

    def get_registry_metadata_value(
        self,
        key: str,
        default: Any | None = None,
    ) -> Any | None:
        """Return one stored top-level registry metadata value."""

        metadata = self._data.get("metadata", {})
        if not isinstance(metadata, dict):
            return default
        return metadata.get(key, default)

    def update_registry_metadata_value(
        self,
        key: str,
        value: Any,
        *,
        save: bool = True,
    ) -> None:
        """Update one top-level registry metadata value."""

        metadata = self._data.setdefault("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}
            self._data["metadata"] = metadata
        metadata[str(key).strip()] = value
        if save:
            self.save()

    def update_project_governance(
        self,
        payload: Dict[str, Any],
        *,
        save: bool = True,
    ) -> None:
        """Update the tracked project-governance registry section."""
        self._data["project-governance"] = dict(payload)
        if save:
            self.save()

    def update_managed_docs(
        self,
        payload: Dict[str, Any],
        *,
        save: bool = True,
    ) -> None:
        """Update the tracked managed-doc registry section."""
        self._data["managed-docs"] = dict(payload)
        if save:
            self.save()

    def update_workflow_contract(
        self,
        payload: Dict[str, Any],
        *,
        save: bool = True,
    ) -> None:
        """Update the tracked workflow-contract registry section."""
        self._data["workflow_contract"] = dict(payload)
        if save:
            self.save()

    def policy_ids(self) -> set[str]:
        """Return policy ids currently stored in the registry."""
        policies = self._data.get("policies", {})
        if not isinstance(policies, dict):
            return set()
        return {str(policy_id) for policy_id in policies.keys()}

    def prune_policies(
        self,
        keep_ids: set[str],
        *,
        save: bool = True,
    ) -> list[str]:
        """Remove policy entries not present in keep_ids."""
        policies = self._data.get("policies", {})
        if not isinstance(policies, dict):
            self._data["policies"] = {}
            return []
        removed = sorted(
            policy_id for policy_id in policies if policy_id not in keep_ids
        )
        for policy_id in removed:
            policies.pop(policy_id, None)
        if removed and save:
            self.save()
        return removed

    def _normalize_hash_value(self, hash_value: object) -> str | None:
        """Normalize stored hash values to a string."""
        if isinstance(hash_value, list):
            return "".join(str(part) for part in hash_value)
        if isinstance(hash_value, str):
            return hash_value
        return None

    def calculate_full_hash(
        self, policy_text: str, script_content: str
    ) -> str:
        """Calculate one combined policy-text plus script-content hash."""
        normalized_policy = policy_text.strip()
        normalized_script = script_content.strip()
        combined = f"{normalized_policy}\n---\n{normalized_script}"
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()

    def check_policy_sync(
        self, policies: List["PolicyDefinition"]
    ) -> List[PolicySyncIssue]:
        """Return synchronization issues for enabled tracked policies."""
        issues = []
        for policy in policies:
            if not policy.enabled:
                continue
            location = resolve_script_location(
                self.repo_root, policy.policy_id
            )
            script_path = location.path if location else Path()
            script_exists = location is not None and script_path.exists()
            current_hash = None
            if policy.policy_id in self._data.get("policies", {}):
                raw_hash = self._data["policies"][policy.policy_id].get("hash")
                current_hash = self._normalize_hash_value(raw_hash)
            if not script_exists:
                issues.append(
                    PolicySyncIssue(
                        policy_id=policy.policy_id,
                        policy_text=policy.description,
                        policy_hash="",
                        script_path=script_path,
                        script_exists=script_exists,
                        issue_type="script_missing",
                        current_hash=current_hash,
                    )
                )
                continue
            script_content = script_path.read_text(encoding="utf-8")
            calculated_hash = self.calculate_full_hash(
                policy.description, script_content
            )
            if current_hash and calculated_hash != current_hash:
                issues.append(
                    PolicySyncIssue(
                        policy_id=policy.policy_id,
                        policy_text=policy.description,
                        policy_hash=calculated_hash,
                        script_path=script_path,
                        script_exists=script_exists,
                        issue_type="hash_mismatch",
                        current_hash=current_hash,
                    )
                )
        return issues

    def _compact_script_path(self, script_path: Path) -> str:
        """Return a shorter script path for registry storage."""
        devcov_root = self.repo_root / "devcovenant"
        try:
            relative = script_path.relative_to(devcov_root)
        except ValueError:
            return require_repo_relative_path(
                self.repo_root,
                script_path,
                label="policy script path",
            )

        parts = relative.parts
        if len(parts) >= 4 and parts[1] == "policies":
            scope = parts[0]
            policy_name = parts[2]
            if relative.name == f"{policy_name}.py":
                return f"{scope}/{policy_name}.py"
        return str(relative)

    def _split_metadata_values(self, raw_value: object) -> List[str]:
        """Split nested metadata values into flat readable tokens."""
        items: List[str] = []

        def _collect(value: object) -> None:
            """Collect leaf values from nested metadata."""
            if value is None:
                return
            if isinstance(value, dict):
                for nested in value.values():
                    _collect(nested)
                return
            if isinstance(value, list):
                for nested in value:
                    _collect(nested)
                return
            text = str(value)
            for part in text.replace("\n", ",").split(","):
                normalized = part.strip()
                if normalized:
                    items.append(normalized)

        _collect(raw_value)
        return items

    def _extract_asset_values(self, metadata: Dict[str, Any]) -> List[str]:
        """Return metadata values that look like asset paths."""
        candidates: List[str] = []
        for metadata_value in metadata.values():
            for token in self._split_metadata_values(metadata_value):
                normalized = token.strip()
                lowered = normalized.lower()
                if "/" in normalized or lowered.endswith(
                    (".md", ".yaml", ".yml", ".json", ".zip")
                ):
                    candidates.append(normalized)
        return sorted(dict.fromkeys(candidates))

    def update_policy_entry(
        self,
        policy: "PolicyDefinition",
        script_location,
        descriptor: PolicyDescriptor | None = None,
        *,
        resolved_metadata: Dict[str, Any] | None = None,
        metadata_resolution: Dict[str, Dict[str, Any]] | None = None,
        metadata_warnings: List[Dict[str, Any]] | None = None,
        runtime_option_views: Dict[str, Dict[str, Any]] | None = None,
        save: bool = True,
    ):
        """Update one policy entry in the tracked registry."""
        entry = self._data["policies"].setdefault(policy.policy_id, {})
        previous_hash = entry.get("hash")
        previous_runtime_state = entry.get("runtime_state")
        entry.clear()
        entry["enabled"] = policy.enabled
        entry["custom"] = policy.custom
        entry["description"] = policy.name
        entry["policy_text"] = policy.description
        metadata_map = dict(resolved_metadata or policy.raw_metadata)
        entry["metadata"] = dict(metadata_map)
        entry["metadata_resolution"] = dict(metadata_resolution or {})
        entry["metadata_warnings"] = list(metadata_warnings or [])
        entry["runtime_actions"] = list(
            getattr(descriptor, "runtime_actions", None) or []
        )
        entry["commands"] = list(getattr(descriptor, "commands", None) or [])
        views = dict(runtime_option_views or {})
        entry["runtime_metadata_options"] = dict(
            views.get("runtime_metadata_options", {})
        )
        entry["runtime_config_overrides"] = dict(
            views.get("runtime_config_overrides", {})
        )
        entry["runtime_effective_options"] = dict(
            views.get("runtime_effective_options", {})
        )
        entry["assets"] = self._extract_asset_values(metadata_map)
        entry["origin"] = None
        entry["script_exists"] = False

        if script_location and script_location.path.exists():
            script_path = script_location.path
            script_content = script_path.read_text(encoding="utf-8")
            entry["hash"] = self.calculate_full_hash(
                policy.description, script_content
            )
            entry["script_path"] = self._compact_script_path(script_path)
            entry["script_exists"] = True
            entry["origin"] = script_location.kind
        else:
            entry["hash"] = previous_hash
            entry["script_path"] = None

        if isinstance(previous_runtime_state, dict) and previous_runtime_state:
            entry["runtime_state"] = dict(previous_runtime_state)

        if save:
            self.save()

    def get_policy_runtime_state(self, policy_id: str) -> Dict[str, Any]:
        """Return one stored per-policy runtime-state mapping."""

        entry = self._data.get("policies", {}).get(policy_id, {})
        if not isinstance(entry, dict):
            return {}
        runtime_state = entry.get("runtime_state", {})
        if not isinstance(runtime_state, dict):
            return {}
        return dict(runtime_state)

    def update_policy_runtime_state(
        self,
        policy_id: str,
        payload: Dict[str, Any],
        *,
        save: bool = True,
    ) -> None:
        """Persist one per-policy runtime-state mapping."""

        policies = self._data.setdefault("policies", {})
        if not isinstance(policies, dict):
            policies = {}
            self._data["policies"] = policies
        entry = policies.setdefault(policy_id, {})
        if not isinstance(entry, dict):
            entry = {}
            policies[policy_id] = entry
        entry["runtime_state"] = dict(payload)
        if save:
            self.save()

    def get_policy_hash(self, policy_id: str) -> Optional[str]:
        """Get the stored hash for one policy."""
        raw_hash = (
            self._data.get("policies", {}).get(policy_id, {}).get("hash")
        )
        return self._normalize_hash_value(raw_hash)

    def get_policy_metadata_map(self, policy_id: str) -> Dict[str, Any]:
        """Return a copy of the stored metadata map for one policy."""
        entry = self._data.get("policies", {}).get(policy_id, {})
        if not isinstance(entry, dict):
            return {}
        raw_metadata = entry.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            return {}
        metadata_map: Dict[str, Any] = {}
        for raw_key, raw_value in raw_metadata.items():
            key = str(raw_key or "").strip()
            if not key:
                continue
            metadata_map[key] = raw_value
        return metadata_map

    def get_policy_metadata_typed(self, policy_id: str) -> Dict[str, Any]:
        """Return a typed metadata view decoded from stored metadata."""
        from devcovenant.core.policy_metadata import (
            decode_metadata_options_map,
        )

        return decode_metadata_options_map(
            self.get_policy_metadata_map(policy_id)
        )
