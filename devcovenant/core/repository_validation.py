"""Tracked manifest inventory plus integrity and structure validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

import devcovenant.core.gate_runtime as status_validation
import devcovenant.core.managed_docs as managed_docs_service
import devcovenant.core.repository_paths as yaml_cache_service
import devcovenant.core.tracked_registry as tracked_registry
import devcovenant.core.workflow_support as workflow_support_module
from devcovenant.core.policy_contract import CheckContext, Violation
from devcovenant.core.policy_metadata import PolicyDefinition, PolicyParser
from devcovenant.core.policy_registry import (
    PolicyRegistry,
    load_policy_descriptor,
)

REGISTRY_DIR = tracked_registry.REGISTRY_DIR
REGISTRY_REL_PATH = tracked_registry.REGISTRY_REL_PATH
RUNTIME_REGISTRY_DIR = workflow_support_module.RUNTIME_REGISTRY_DIR
GATE_STATUS_FILENAME = workflow_support_module.GATE_STATUS_FILENAME
WORKFLOW_SESSION_FILENAME = workflow_support_module.WORKFLOW_SESSION_FILENAME
LATEST_RUNTIME_FILENAME = workflow_support_module.LATEST_RUNTIME_FILENAME
SESSION_SNAPSHOT_FILENAME = workflow_support_module.SESSION_SNAPSHOT_FILENAME

DEFAULT_CORE_DIRS = [
    "devcovenant",
    "devcovenant/builtin",
    "devcovenant/builtin/policies",
    "devcovenant/builtin/profiles",
    "devcovenant/builtin/profiles/github",
    "devcovenant/builtin/profiles/github/assets",
    "devcovenant/builtin/profiles/global",
    "devcovenant/builtin/profiles/global/assets",
    "devcovenant/core",
    "devcovenant/licenses",
    "devcovenant/logs",
    REGISTRY_DIR,
]
DEFAULT_SCAN_EXCLUDED_CORE_PATHS = [
    "devcovenant/core",
    "devcovenant/builtin",
    "devcovenant/licenses",
    "devcovenant/__init__.py",
    "devcovenant/__main__.py",
    "devcovenant/asset.py",
    "devcovenant/cli.py",
    "devcovenant/check.py",
    "devcovenant/clean.py",
    "devcovenant/gate.py",
    "devcovenant/run.py",
    "devcovenant/policy.py",
    "devcovenant/install.py",
    "devcovenant/deploy.py",
    "devcovenant/upgrade.py",
    "devcovenant/refresh.py",
    "devcovenant/uninstall.py",
    "devcovenant/undeploy.py",
    "devcovenant/runtime-requirements.lock",
    "devcovenant/registry",
]
DEFAULT_CORE_FILES = [
    "devcovenant/__init__.py",
    "devcovenant/__main__.py",
    "devcovenant/asset.py",
    "devcovenant/cli.py",
    "devcovenant/check.py",
    "devcovenant/gate.py",
    "devcovenant/run.py",
    "devcovenant/policy.py",
    "devcovenant/install.py",
    "devcovenant/deploy.py",
    "devcovenant/upgrade.py",
    "devcovenant/refresh.py",
    "devcovenant/uninstall.py",
    "devcovenant/undeploy.py",
    "devcovenant/config.yaml",
    "devcovenant/README.md",
    "devcovenant/VERSION",
    "devcovenant/runtime-requirements.lock",
    "devcovenant/licenses/LICENSE",
    "devcovenant/licenses/README.md",
    "devcovenant/licenses/THIRD_PARTY_LICENSES.md",
    "devcovenant/logs/README.md",
    f"{REGISTRY_DIR}/README.md",
    REGISTRY_REL_PATH,
    "devcovenant/builtin/profiles/github/assets/ci.yml",
    "devcovenant/builtin/profiles/global/assets/gitignore.yaml",
    "devcovenant/builtin/profiles/README.md",
    "devcovenant/builtin/policies/README.md",
    "devcovenant/core/README.md",
    "devcovenant/core/__init__.py",
    "devcovenant/core/agents_blocks.py",
    "devcovenant/core/asset_materialization.py",
    "devcovenant/core/cleanup.py",
    "devcovenant/core/cli_support.py",
    "devcovenant/core/execution.py",
    "devcovenant/core/gate_runtime.py",
    "devcovenant/core/managed_docs.py",
    "devcovenant/core/policy_autofix.py",
    "devcovenant/core/policy_contract.py",
    "devcovenant/core/policy_runtime.py",
    "devcovenant/core/policy_commands.py",
    "devcovenant/core/policy_metadata.py",
    "devcovenant/core/policy_registry.py",
    "devcovenant/core/policy_runtime_actions.py",
    "devcovenant/core/refresh_runtime.py",
    "devcovenant/core/profile_registry.py",
    "devcovenant/core/project_governance.py",
    "devcovenant/core/repository_paths.py",
    "devcovenant/core/repository_validation.py",
    "devcovenant/core/run_events.py",
    "devcovenant/core/run_logs.py",
    "devcovenant/core/runtime_errors.py",
    "devcovenant/core/runtime_profile.py",
    "devcovenant/core/selectors.py",
    "devcovenant/core/tracked_registry.py",
    "devcovenant/core/translator.py",
    "devcovenant/core/workflow_support.py",
    "devcovenant/core/document_exemptions.py",
]
DEFAULT_AVAILABLE_DOCS = [
    "AGENTS.md",
    "README.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "SPEC.md",
    "PLAN.md",
    "SECURITY.md",
    "PRIVACY.md",
    "SUPPORT.md",
    "LICENSE",
    "devcovenant/README.md",
]
DEFAULT_ENABLED_DOCS = [
    "AGENTS.md",
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "SPEC.md",
    "PLAN.md",
    "devcovenant/README.md",
]
DEFAULT_CUSTOM_DIRS = [
    "devcovenant/custom",
    "devcovenant/custom/policies",
    "devcovenant/custom/profiles",
]
DEFAULT_CUSTOM_FILES = [
    "devcovenant/custom/profiles/README.md",
    "devcovenant/custom/policies/README.md",
]
DEFAULT_GENERATED_FILES = [
    f"{RUNTIME_REGISTRY_DIR}/{GATE_STATUS_FILENAME}",
    f"{RUNTIME_REGISTRY_DIR}/{LATEST_RUNTIME_FILENAME}",
    f"{RUNTIME_REGISTRY_DIR}/{WORKFLOW_SESSION_FILENAME}",
]
DEFAULT_GENERATED_DIRS: List[str] = [RUNTIME_REGISTRY_DIR]


def default_scan_excluded_core_paths() -> list[str]:
    """Return the canonical core paths hidden from normal repo scans."""
    return list(DEFAULT_SCAN_EXCLUDED_CORE_PATHS)


def manifest_path(repo_root: Path) -> Path:
    """Return the tracked registry document path used for inventory data."""
    return tracked_registry.policy_registry_path(repo_root)


def build_manifest(
    *,
    options: Dict[str, Any] | None = None,
    installed: Dict[str, Any] | None = None,
    doc_blocks: List[str] | None = None,
    available_docs: List[str] | None = None,
    enabled_docs: List[str] | None = None,
) -> Dict[str, Any]:
    """Build a deterministic inventory payload for the tracked registry."""
    manifest: Dict[str, Any] = {
        "schema_version": 3,
        "core": {
            "dirs": list(DEFAULT_CORE_DIRS),
            "files": list(DEFAULT_CORE_FILES),
        },
        "docs": {
            "available": list(available_docs or DEFAULT_AVAILABLE_DOCS),
            "enabled": list(enabled_docs or DEFAULT_ENABLED_DOCS),
        },
        "custom": {
            "dirs": list(DEFAULT_CUSTOM_DIRS),
            "files": list(DEFAULT_CUSTOM_FILES),
        },
        "generated": {
            "dirs": list(DEFAULT_GENERATED_DIRS),
            "files": list(DEFAULT_GENERATED_FILES),
        },
        "profiles": {
            "active": [],
            "resolved_pre_commit_hooks": [],
        },
    }
    if options is not None:
        manifest["options"] = options
    if installed is not None:
        manifest["installed"] = installed
    if doc_blocks is not None:
        manifest["doc_blocks"] = doc_blocks
    return manifest


def _resolved_docs_manifest(repo_root: Path) -> dict[str, list[str]]:
    """Return the available/enabled managed-doc inventory for one repo."""
    available_docs = list(DEFAULT_AVAILABLE_DOCS)
    enabled_docs = list(DEFAULT_ENABLED_DOCS)

    try:
        entries = managed_docs_service.managed_doc_descriptor_entries(
            repo_root
        )
    except ValueError:
        entries = []
    if entries:
        available_docs = [str(entry["doc"]) for entry in entries]

    config_path = repo_root / "devcovenant" / "config.yaml"
    if config_path.exists():
        try:
            config_payload = yaml_cache_service.load_yaml(config_path)
        except (OSError, yaml.YAMLError):
            config_payload = {}
        if isinstance(config_payload, dict):
            try:
                enabled_docs = managed_docs_service.managed_docs_from_config(
                    config_payload
                )
            except ValueError:
                pass

    return {
        "available": available_docs,
        "enabled": enabled_docs,
    }


def load_manifest(repo_root: Path) -> Dict[str, Any] | None:
    """Load the tracked inventory section if present, otherwise return None."""
    path = manifest_path(repo_root)
    payload = tracked_registry.load_registry_document(path)
    inventory = payload.get("inventory", {})
    return (
        dict(inventory) if isinstance(inventory, dict) and inventory else None
    )


def write_manifest(repo_root: Path, manifest: Dict[str, Any]) -> Path:
    """Write inventory data into the tracked registry document."""
    path = manifest_path(repo_root)
    payload = tracked_registry.load_registry_document(path)
    payload["inventory"] = dict(manifest)
    return tracked_registry.write_registry_document(path, payload)


def _normalize_manifest_sections(
    repo_root: Path,
    manifest: Dict[str, Any],
) -> tuple[Dict[str, Any], bool]:
    """Normalize inventory sections to the current default inventories."""
    normalized = dict(manifest)
    changed = False
    docs_manifest = _resolved_docs_manifest(repo_root)
    defaults_manifest = build_manifest(
        available_docs=docs_manifest["available"],
        enabled_docs=docs_manifest["enabled"],
    )
    for section_name in ("core", "docs", "custom", "generated"):
        defaults = defaults_manifest.get(section_name, {})
        current = normalized.get(section_name, {})
        if not isinstance(defaults, dict):
            continue
        if not isinstance(current, dict):
            normalized[section_name] = defaults
            changed = True
            continue
        merged = dict(current)
        for key, default_value in defaults.items():
            target_value = (
                list(default_value)
                if isinstance(default_value, list)
                else default_value
            )
            if merged.get(key) != target_value:
                merged[key] = target_value
                changed = True
        normalized[section_name] = merged
    return normalized, changed


def ensure_manifest(repo_root: Path) -> Dict[str, Any] | None:
    """Create the tracked inventory section when missing."""
    path = manifest_path(repo_root)
    if path.exists():
        payload = load_manifest(repo_root)
        if payload is None:
            docs_manifest = _resolved_docs_manifest(repo_root)
            payload = build_manifest(
                available_docs=docs_manifest["available"],
                enabled_docs=docs_manifest["enabled"],
            )
        normalized, changed = _normalize_manifest_sections(repo_root, payload)
        if changed:
            write_manifest(repo_root, normalized)
        return normalized
    if not (repo_root / tracked_registry.DEV_COVENANT_DIR).exists():
        return None
    docs_manifest = _resolved_docs_manifest(repo_root)
    manifest = build_manifest(
        available_docs=docs_manifest["available"],
        enabled_docs=docs_manifest["enabled"],
    )
    write_manifest(repo_root, manifest)
    return manifest


CHECK_ID = "integrity-validation"
_DEFAULT_STATUS_PATH = (
    Path("devcovenant") / "registry" / "runtime" / "gate_status.json"
)
_DEFAULT_POLICY_DEFINITIONS = Path("AGENTS.md")
_DEFAULT_REGISTRY_FILE = Path("devcovenant/registry/registry.yaml")


def _load_config_payload_or_empty(repo_root: Path) -> dict[str, object]:
    """Load config when present, otherwise return an empty payload."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _merged_section(
    repo_root: Path,
    context_config: dict[str, object],
    section_name: str,
) -> dict[str, object]:
    """Return one config section merged with in-context overrides."""
    merged: dict[str, object] = {}
    repo_payload = _load_config_payload_or_empty(repo_root)
    repo_section = repo_payload.get(section_name)
    if isinstance(repo_section, dict):
        merged.update(repo_section)
    context_section = context_config.get(section_name)
    if isinstance(context_section, dict):
        merged.update(context_section)
    return merged


def _relative_path_option(
    raw_mapping: dict[str, object],
    key: str,
    default: str | Path,
) -> Path:
    """Return one repo-relative path from string-or-list config values."""
    value = raw_mapping.get(key, default)
    if isinstance(value, (list, tuple)):
        for entry in value:
            token = str(entry or "").strip()
            if token:
                return Path(token)
        return Path(str(default))
    token = str(value or "").strip()
    if token:
        return Path(token)
    return Path(str(default))


def _string_list_option(
    raw_mapping: dict[str, object],
    key: str,
) -> list[str]:
    """Return one list-valued config option as cleaned strings."""
    raw_value = raw_mapping.get(key, [])
    if isinstance(raw_value, str):
        token = raw_value.strip()
        return [token] if token else []
    if not isinstance(raw_value, list):
        return []
    values: list[str] = []
    for entry in raw_value:
        token = str(entry or "").strip()
        if token:
            values.append(token)
    return values


def _normalize_policy_text(text_value: str) -> str:
    """Normalize policy text for descriptor comparisons."""
    return "\n".join(line.rstrip() for line in text_value.strip().splitlines())


def _has_meaningful_description(description: str) -> bool:
    """Return True when the policy description is non-empty and useful."""
    if not description:
        return False
    normalized = description.strip()
    if not normalized:
        return False
    if normalized.lower().startswith("<!-- devcov:"):
        return False
    if all(line.strip() in {"---", ""} for line in normalized.splitlines()):
        return False
    return True


def _requires_status_update(
    rel_path: Path,
    watched_roots: set[str],
    watched_files: set[str],
) -> bool:
    """Return True when rel_path should trigger a gate-status refresh."""
    if not rel_path.parts:
        return False
    if rel_path == _DEFAULT_STATUS_PATH:
        return False
    first_segment = rel_path.parts[0]
    if first_segment in watched_roots:
        return True
    if rel_path.name in watched_files:
        return True
    return False


def _load_policies(
    agents_path: Path,
) -> tuple[list[PolicyDefinition], list[Violation]]:
    """Return parsed AGENTS policies or a blocking violation."""
    if not agents_path.exists():
        return [], [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=agents_path,
                message="Policy definitions file is missing.",
                suggestion="Restore AGENTS.md before running checks.",
            )
        ]
    parsed = PolicyParser(agents_path).parse_agents_md()
    return parsed, []


def _check_policy_text_integrity(
    context: CheckContext,
    agents_path: Path,
    policies: list[PolicyDefinition],
) -> list[Violation]:
    """Validate descriptor parity and non-empty policy descriptions."""
    violations: list[Violation] = []
    for policy in policies:
        description = policy.description.strip()
        if not _has_meaningful_description(description):
            violations.append(
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=agents_path,
                    message=(
                        "Policy definitions must include descriptive text. "
                        f"Missing text for policy '{policy.policy_id}'."
                    ),
                    suggestion=(
                        "Add meaningful prose immediately after the "
                        f"`policy-def` block for '{policy.policy_id}'."
                    ),
                )
            )

        descriptor = load_policy_descriptor(
            context.repo_root, policy.policy_id
        )
        if not descriptor or not descriptor.text:
            continue
        if _normalize_policy_text(description) == _normalize_policy_text(
            descriptor.text
        ):
            continue
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="warning",
                file_path=agents_path,
                message=(
                    "Descriptor policy text differs from AGENTS. Policy "
                    f"'{policy.policy_id}' should match its descriptor text."
                ),
                suggestion=(
                    "Regenerate AGENTS from descriptors or update the "
                    "descriptor text to match the intended policy prose."
                ),
            )
        )
    return violations


def _check_registry_sync(
    context: CheckContext,
    registry_path: Path,
    policies: list[PolicyDefinition],
) -> list[Violation]:
    """Validate registry hash synchronization for discovered policies."""
    if not registry_path.exists():
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=registry_path,
                message="Policy registry file is missing.",
                suggestion="Run `devcovenant refresh`.",
            )
        ]

    registry = PolicyRegistry(registry_path, context.repo_root)
    sync_issues = registry.check_policy_sync(policies)
    violations: list[Violation] = []
    for issue in sync_issues:
        if issue.issue_type == "script_missing":
            message = f"Policy script missing for policy '{issue.policy_id}'."
            suggestion = "Add the policy script or remove the policy."
        else:
            message = (
                "Policy registry hash mismatch for policy "
                f"'{issue.policy_id}'."
            )
            suggestion = "Run `devcovenant refresh`."
        violations.append(
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=issue.script_path or registry_path,
                message=message,
                suggestion=suggestion,
            )
        )
    return violations


def _check_gate_status(
    context: CheckContext,
    status_relative: Path,
    watched_dirs: list[str],
    watched_files: list[str],
) -> list[Violation]:
    """Validate gate-status metadata when watched files are modified."""
    changed_paths: Iterable[Path] = context.changed_files or []
    watched_roots = set(watched_dirs)
    watched_file_names = {Path(entry).name for entry in watched_files}

    status_changed = False
    relevant_change = False
    for changed_path in changed_paths:
        try:
            rel_path = changed_path.relative_to(context.repo_root)
        except ValueError:
            continue
        if rel_path == status_relative:
            status_changed = True
        if _requires_status_update(
            rel_path, watched_roots, watched_file_names
        ):
            relevant_change = True

    if not relevant_change:
        return []

    status_path = context.repo_root / status_relative
    if not status_changed:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_path,
                line_number=1,
                message=(
                    "Code changes require a fresh gate status update. Run "
                    "`devcovenant run` so the workflow runs execute and "
                    "the status file is refreshed."
                ),
            )
        ]

    try:
        status_validation.validate_gate_status_payload(status_path)
    except ValueError as exc:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=status_path,
                line_number=1,
                message=f"{status_relative.as_posix()} is invalid: {exc}",
            )
        ]
    return []


def check_integrity(context: CheckContext) -> list[Violation]:
    """Run descriptor, registry, and gate-status integrity checks."""
    path_settings = _merged_section(context.repo_root, context.config, "paths")
    integrity_settings = _merged_section(
        context.repo_root,
        context.config,
        "integrity",
    )
    agents_relative = _relative_path_option(
        path_settings,
        "policy_definitions",
        _DEFAULT_POLICY_DEFINITIONS,
    )
    agents_path = context.repo_root / agents_relative
    policies, policy_load_violations = _load_policies(agents_path)
    if policy_load_violations:
        return policy_load_violations

    registry_relative = _relative_path_option(
        path_settings,
        "registry_file",
        _DEFAULT_REGISTRY_FILE,
    )
    status_relative = _relative_path_option(
        path_settings,
        "gate_status_file",
        _DEFAULT_STATUS_PATH,
    )

    violations: list[Violation] = []
    violations.extend(
        _check_policy_text_integrity(context, agents_path, policies)
    )
    violations.extend(
        _check_registry_sync(
            context,
            context.repo_root / registry_relative,
            policies,
        )
    )
    violations.extend(
        _check_gate_status(
            context,
            status_relative,
            watched_dirs=_string_list_option(integrity_settings, "watch_dirs"),
            watched_files=_string_list_option(
                integrity_settings,
                "watch_files",
            ),
        )
    )
    return violations


CHECK_ID = "structure-validation"


def _read_active_profiles(repo_root: Path) -> list[str]:
    """Read active profiles from repo config when available."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return []
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return []
    if not isinstance(payload, dict):
        return []
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return []
    active = profiles.get("active")
    if not isinstance(active, list):
        return []
    return [str(token).strip() for token in active if str(token).strip()]


def _repo_requires_bytecode_hygiene(repo_root: Path) -> bool:
    """Return True when developer-mode repos enforce bytecode hygiene."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return False
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return False
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("developer_mode", False))


def _find_nested_core_paths(repo_root: Path) -> list[str]:
    """Return repo-relative nested paths under `devcovenant/core/`."""
    core_root = repo_root / "devcovenant" / "core"
    if not core_root.exists():
        return []
    nested: list[str] = []
    for path in core_root.rglob("*"):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(repo_root)
        if len(relative.parts) <= 3:
            continue
        nested.append(relative.as_posix())
    return sorted(nested)


def _find_bytecode_artifacts(repo_root: Path) -> list[str]:
    """Return repo-relative bytecode artifacts under devcovenant/."""
    devcovenant_root = repo_root / "devcovenant"
    if not devcovenant_root.exists():
        return []
    artifacts: list[str] = []
    for path in devcovenant_root.rglob("*"):
        if path.is_dir() and path.name == "__pycache__":
            artifacts.append(str(path.relative_to(repo_root)))
            continue
        if path.is_file() and path.suffix in {".pyc", ".pyo", ".pyd"}:
            artifacts.append(str(path.relative_to(repo_root)))
    return artifacts


def check_structure(context: CheckContext) -> list[Violation]:
    """Check for required DevCovenant files, directories, and hygiene."""
    manifest = ensure_manifest(context.repo_root)
    if manifest is None:
        return [
            Violation(
                policy_id=CHECK_ID,
                severity="error",
                file_path=context.repo_root / "devcovenant",
                message=(
                    "Manifest is missing and could not be created for the "
                    "current repository."
                ),
                suggestion="Restore `devcovenant/` and rerun refresh.",
                can_auto_fix=False,
            )
        ]

    core = manifest.get("core", {})
    docs = manifest.get("docs", {})
    required_dirs = core.get("dirs", [])
    required_files = core.get("files", [])
    required_docs = docs.get("enabled", [])

    missing: list[str] = []
    for rel_path in required_dirs:
        path = context.repo_root / rel_path
        if not path.is_dir():
            missing.append(rel_path)
    for rel_path in list(required_files) + list(required_docs):
        path = context.repo_root / rel_path
        if not path.exists():
            missing.append(rel_path)

    if not missing:
        nested_core_paths = _find_nested_core_paths(context.repo_root)
        if nested_core_paths:
            sample = nested_core_paths[0]
            return [
                Violation(
                    policy_id=CHECK_ID,
                    severity="error",
                    file_path=context.repo_root / sample,
                    message=(
                        "Nested paths were found under `devcovenant/core/`: "
                        f"{sample}"
                    ),
                    suggestion=(
                        "Keep `devcovenant/core/` flat and move nested code "
                        "into top-level core modules."
                    ),
                    can_auto_fix=False,
                )
            ]
        if _repo_requires_bytecode_hygiene(context.repo_root):
            artifacts = _find_bytecode_artifacts(context.repo_root)
            if artifacts:
                sample = artifacts[0]
                return [
                    Violation(
                        policy_id=CHECK_ID,
                        severity="error",
                        file_path=context.repo_root / sample,
                        message=(
                            "Repo-local bytecode artifacts were found under "
                            f"devcovenant/: {sample}"
                        ),
                        suggestion=(
                            "Remove bytecode artifacts under devcovenant/ "
                            "and rerun the gate sequence."
                        ),
                        can_auto_fix=False,
                    )
                ]
        return []

    message = "Missing required DevCovenant paths: " + ", ".join(missing)
    return [
        Violation(
            policy_id=CHECK_ID,
            severity="error",
            file_path=context.repo_root / missing[0],
            message=message,
            suggestion="Run `devcovenant refresh` to restore managed files.",
            can_auto_fix=False,
        )
    ]
