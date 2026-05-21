"""Policy-born command and runtime-action declarations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

import devcovenant.core.policy_registry as policy_registry
import devcovenant.core.repository_paths as repository_paths
import devcovenant.core.tracked_registry as tracked_registry


@dataclass(frozen=True)
class PolicyRuntimeActionDefinition:
    """Declared runtime action exposed by one policy."""

    action_id: str
    description: str
    mutates_repo: bool


@dataclass(frozen=True)
class PolicyCommandArgumentDefinition:
    """Declarative argparse-facing argument metadata for one policy command."""

    flags: tuple[str, ...]
    dest: str
    help_text: str
    value_type: str = "str"
    required: bool = False
    default: object = None
    choices: tuple[str, ...] = ()
    nargs: str | None = None
    metavar: str | None = None
    action: str | None = None


@dataclass(frozen=True)
class PolicyCommandDefinition:
    """Declared CLI command exposed by one policy."""

    name: str
    help_text: str
    runtime_action: str
    mutates_repo: bool
    aliases: tuple[str, ...] = ()
    visible_when_enabled: bool = True
    arguments: tuple[PolicyCommandArgumentDefinition, ...] = ()


def _read_registry_payload(repo_root: Path) -> dict[str, object]:
    """Load the tracked registry mapping when available."""
    path = tracked_registry.policy_registry_path(repo_root)
    if not path.exists():
        return {}
    try:
        payload = repository_paths.load_yaml(path)
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _registry_policy_entry(
    repo_root: Path,
    policy_id: str,
) -> dict[str, object]:
    """Return one policy registry entry when available."""
    payload = _read_registry_payload(repo_root)
    policies = payload.get("policies") if isinstance(payload, dict) else None
    if not isinstance(policies, dict):
        return {}
    entry = policies.get(policy_id)
    return entry if isinstance(entry, dict) else {}


def _normalized_action_definitions(
    raw: object,
) -> list[PolicyRuntimeActionDefinition]:
    """Normalize runtime-action declarations into typed definitions."""
    if not isinstance(raw, list):
        return []
    definitions: list[PolicyRuntimeActionDefinition] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        action_id = str(entry.get("id", "")).strip()
        if not action_id:
            continue
        definitions.append(
            PolicyRuntimeActionDefinition(
                action_id=action_id,
                description=str(entry.get("description", "")).strip(),
                mutates_repo=bool(entry.get("mutates_repo", False)),
            )
        )
    return definitions


def _normalize_argument_definitions(
    raw: object,
) -> tuple[PolicyCommandArgumentDefinition, ...]:
    """Normalize command argument declarations into typed definitions."""
    if not isinstance(raw, list):
        return ()
    arguments: list[PolicyCommandArgumentDefinition] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        raw_flags = entry.get("flags")
        flags: list[str] = []
        if isinstance(raw_flags, list):
            flags = [
                str(item).strip() for item in raw_flags if str(item).strip()
            ]
        elif isinstance(raw_flags, str) and raw_flags.strip():
            flags = [raw_flags.strip()]
        dest = str(entry.get("dest", "")).strip()
        if not dest:
            if flags:
                dest = flags[-1].lstrip("-").replace("-", "_")
            else:
                continue
        choices = entry.get("choices")
        normalized_choices: tuple[str, ...] = ()
        if isinstance(choices, list):
            normalized_choices = tuple(
                str(item).strip() for item in choices if str(item).strip()
            )
        arguments.append(
            PolicyCommandArgumentDefinition(
                flags=tuple(flags),
                dest=dest,
                help_text=str(entry.get("help", "")).strip(),
                value_type=str(entry.get("type", "str")).strip() or "str",
                required=bool(entry.get("required", False)),
                default=entry.get("default"),
                choices=normalized_choices,
                nargs=(
                    str(entry.get("nargs")).strip()
                    if entry.get("nargs") is not None
                    else None
                ),
                metavar=(
                    str(entry.get("metavar")).strip()
                    if entry.get("metavar") is not None
                    else None
                ),
                action=(
                    str(entry.get("action")).strip()
                    if entry.get("action") is not None
                    else None
                ),
            )
        )
    return tuple(arguments)


def _normalized_command_definitions(
    raw: object,
) -> list[PolicyCommandDefinition]:
    """Normalize command declarations into typed definitions."""
    if not isinstance(raw, list):
        return []
    commands: list[PolicyCommandDefinition] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name", "")).strip()
        runtime_action = str(entry.get("runtime_action", "")).strip()
        if not name or not runtime_action:
            continue
        raw_aliases = entry.get("aliases")
        aliases: tuple[str, ...] = ()
        if isinstance(raw_aliases, list):
            aliases = tuple(
                str(item).strip() for item in raw_aliases if str(item).strip()
            )
        elif isinstance(raw_aliases, str) and raw_aliases.strip():
            aliases = (raw_aliases.strip(),)
        commands.append(
            PolicyCommandDefinition(
                name=name,
                help_text=str(entry.get("help", "")).strip(),
                runtime_action=runtime_action,
                mutates_repo=bool(entry.get("mutates_repo", False)),
                aliases=aliases,
                visible_when_enabled=bool(
                    entry.get("visible_when_enabled", True)
                ),
                arguments=_normalize_argument_definitions(
                    entry.get("arguments", [])
                ),
            )
        )
    return commands


def load_policy_runtime_action_definitions(
    repo_root: Path,
    policy_id: str,
) -> list[PolicyRuntimeActionDefinition]:
    """Return declared runtime actions for one policy."""
    entry = _registry_policy_entry(repo_root, policy_id)
    definitions = _normalized_action_definitions(entry.get("runtime_actions"))
    if definitions:
        return definitions
    descriptor = policy_registry.load_policy_descriptor(repo_root, policy_id)
    if descriptor is None:
        return []
    return _normalized_action_definitions(descriptor.runtime_actions)


def load_policy_command_definitions(
    repo_root: Path,
    policy_id: str,
) -> list[PolicyCommandDefinition]:
    """Return declared policy-born commands for one policy."""
    entry = _registry_policy_entry(repo_root, policy_id)
    definitions = _normalized_command_definitions(entry.get("commands"))
    if definitions:
        return definitions
    descriptor = policy_registry.load_policy_descriptor(repo_root, policy_id)
    if descriptor is None:
        return []
    return _normalized_command_definitions(descriptor.commands)


def validate_runtime_action_declared(
    repo_root: Path,
    *,
    policy_id: str,
    action: str,
) -> None:
    """Raise when a runtime action is not declared by the policy descriptor."""
    declared = {
        definition.action_id
        for definition in load_policy_runtime_action_definitions(
            repo_root,
            policy_id,
        )
    }
    if action not in declared:
        raise ValueError(
            f"Policy `{policy_id}` does not declare runtime action `{action}`."
        )


def find_policy_command(
    repo_root: Path,
    *,
    policy_id: str,
    command_name: str,
) -> PolicyCommandDefinition | None:
    """Return one policy command definition by name or alias."""
    for definition in load_policy_command_definitions(repo_root, policy_id):
        if (
            command_name == definition.name
            or command_name in definition.aliases
        ):
            return definition
    return None


def enabled_policy_commands(
    repo_root: Path,
) -> dict[str, list[PolicyCommandDefinition]]:
    """Return visible command definitions for enabled policies."""
    payload = _read_registry_payload(repo_root)
    policies = payload.get("policies") if isinstance(payload, dict) else None
    if not isinstance(policies, dict):
        return {}
    commands: dict[str, list[PolicyCommandDefinition]] = {}
    for policy_id, raw_entry in policies.items():
        entry = raw_entry if isinstance(raw_entry, dict) else {}
        enabled = bool(entry.get("enabled", False))
        policy_commands = load_policy_command_definitions(
            repo_root, str(policy_id)
        )
        visible = [
            definition
            for definition in policy_commands
            if enabled or not definition.visible_when_enabled
        ]
        if visible:
            commands[str(policy_id)] = visible
    return commands


def build_policy_command_parser(
    policy_id: str,
    command: PolicyCommandDefinition,
) -> argparse.ArgumentParser:
    """Build an argparse parser for one declared policy command."""
    parser = argparse.ArgumentParser(
        prog=f"devcovenant policy {policy_id} {command.name}",
        description=command.help_text or None,
    )
    for argument in command.arguments:
        kwargs: Dict[str, Any] = {}
        if argument.help_text:
            kwargs["help"] = argument.help_text
        if argument.required:
            kwargs["required"] = True
        if argument.default is not None:
            kwargs["default"] = argument.default
        if argument.choices:
            kwargs["choices"] = list(argument.choices)
        if argument.nargs is not None:
            kwargs["nargs"] = argument.nargs
        if argument.metavar is not None:
            kwargs["metavar"] = argument.metavar
        if argument.action:
            kwargs["action"] = argument.action
        else:
            if argument.value_type == "int":
                kwargs["type"] = int
            elif argument.value_type == "float":
                kwargs["type"] = float
            else:
                kwargs["type"] = str
        kwargs["dest"] = argument.dest
        flags = list(argument.flags)
        if not flags:
            flags = [argument.dest]
        parser.add_argument(*flags, **kwargs)
    return parser


def parse_policy_command_payload(
    policy_id: str,
    command: PolicyCommandDefinition,
    argv: list[str],
) -> dict[str, Any]:
    """Parse one policy command argv list into a runtime-action payload."""
    parser = build_policy_command_parser(policy_id, command)
    namespace = parser.parse_args(argv)
    return vars(namespace)


def canonical_policy_command_invocation(
    policy_id: str,
    command_name: str,
) -> str:
    """Return the canonical CLI invocation for one policy command."""
    return f"devcovenant policy {policy_id} {command_name}"
