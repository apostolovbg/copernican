"""AGENTS policy-block rendering and refresh helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

import devcovenant.core.policy_metadata as metadata_runtime
import devcovenant.core.repository_paths as yaml_cache_service
from devcovenant.core.policy_metadata import parse_policy_metadata_block
from devcovenant.core.policy_registry import (
    POLICY_BLOCK_RE,
    PolicyDescriptor,
    load_policy_descriptor,
)
from devcovenant.core.repository_paths import display_path
from devcovenant.core.tracked_registry import policy_registry_path

POLICIES_BEGIN = "<!-- DEVCOV-POLICIES:BEGIN -->"
POLICIES_END = "<!-- DEVCOV-POLICIES:END -->"


@dataclass(frozen=True)
class PolicyBlockRefreshResult:
    """Summary of policy-block refresh work."""

    changed_policies: Tuple[str, ...]
    skipped_policies: Tuple[str, ...]
    updated: bool


@dataclass
class _PolicyEntry:
    """Track a policy block's key attributes during refresh."""

    policy_id: str
    text: str


def _read_yaml(path: Path) -> dict[str, object]:
    """Load a YAML mapping payload from disk."""
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


def _locate_block(
    text: str,
    begin_marker: str,
    end_marker: str,
    label: str,
) -> tuple[int, int, str]:
    """Return the span and content for one managed AGENTS block."""
    try:
        start = text.index(begin_marker)
        end = text.index(end_marker, start + len(begin_marker))
    except ValueError as exc:
        raise ValueError(
            f"{label} block markers not found in AGENTS.md"
        ) from exc
    block_start = start
    block_end = end + len(end_marker)
    return block_start, block_end, text[block_start:block_end]


def _ensure_policy_block_scaffold(
    agents_path: Path,
    content: str,
) -> tuple[str, bool]:
    """Ensure AGENTS has exactly one policy marker block scaffold."""
    has_begin = POLICIES_BEGIN in content
    has_end = POLICIES_END in content
    if has_begin and has_end:
        return content, False
    stripped = (
        content.replace(POLICIES_BEGIN, "").replace(POLICIES_END, "").rstrip()
    )
    scaffold = f"{POLICIES_BEGIN}\n{POLICIES_END}\n"
    rebuilt = f"{stripped}\n\n{scaffold}"
    agents_path.write_text(rebuilt, encoding="utf-8")
    return rebuilt, True


def _metadata_from_registry(
    policy_id: str,
    metadata_map: object,
) -> Tuple[List[str], Dict[str, object]]:
    """Return ordered metadata keys and values sourced from registry data."""
    if not isinstance(metadata_map, dict):
        return ["id"], {"id": policy_id}
    order: List[str] = []
    values: Dict[str, object] = {}
    for key, raw_value in metadata_map.items():
        key_name = str(key).strip()
        if not key_name:
            continue
        order.append(key_name)
        values[key_name] = raw_value
    values["id"] = policy_id
    if "id" not in order:
        order.insert(0, "id")
    return order, values


def _assemble_sections(entries: List[_PolicyEntry]) -> str:
    """Build a policy block ordered alphabetically."""
    if not entries:
        return ""
    sorted_entries = sorted(entries, key=lambda item: item.policy_id)
    sections_text: List[str] = []
    for idx, entry in enumerate(sorted_entries):
        if idx > 0:
            sections_text.append("\n\n---\n\n")
        sections_text.append(entry.text)
    final = "".join(sections_text)
    if not final.endswith("\n"):
        final += "\n"
    return final


def _section_map(block_text: str) -> Dict[str, str]:
    """Return a map of policy id to rendered section from a policy block."""
    sections: Dict[str, str] = {}
    for match in POLICY_BLOCK_RE.finditer(block_text):
        heading = match.group(1)
        metadata_block = match.group(2).strip()
        order, values = parse_policy_metadata_block(metadata_block)
        raw_policy_id = values.get("id", "")
        policy_id = str(raw_policy_id or "").strip()
        if not policy_id:
            continue
        description = match.group(3).strip()
        rendered = metadata_runtime.render_metadata_block(order, values)
        section = f"{heading}```policy-def\n{rendered}\n```\n\n{description}\n"
        sections[policy_id] = section
    return sections


def _descriptor_text_or_error(
    descriptor: PolicyDescriptor | None,
    policy_id: str,
) -> str:
    """Return canonical descriptor text or raise when missing."""
    if descriptor is None:
        raise ValueError(
            f"Missing policy descriptor for `{policy_id}`."
            " Add a <policy>.yaml file with a non-empty `text` field."
        )
    text = str(descriptor.text or "").strip()
    if text:
        return text
    raise ValueError(
        f"Missing descriptor text for `{policy_id}`."
        " Set the `text` field in the policy descriptor YAML."
    )


def refresh_agents_policy_block(
    agents_path: Path,
    schema_path: Path | None,
    *,
    repo_root: Path | None = None,
) -> PolicyBlockRefreshResult:
    """Refresh the AGENTS policy block from registry policy entries."""
    del schema_path
    if not agents_path.exists():
        return PolicyBlockRefreshResult((), (), False)
    repo_root = repo_root or agents_path.parent
    content = agents_path.read_text(encoding="utf-8")
    scaffolded = False
    try:
        _, block_end, _ = _locate_block(
            content,
            POLICIES_BEGIN,
            POLICIES_END,
            "Policy",
        )
        block_start = content.index(POLICIES_BEGIN) + len(POLICIES_BEGIN)
        block_text = content[block_start : block_end - len(POLICIES_END)]
    except ValueError:
        content, scaffolded = _ensure_policy_block_scaffold(
            agents_path,
            content,
        )
        try:
            _, block_end, _ = _locate_block(
                content,
                POLICIES_BEGIN,
                POLICIES_END,
                "Policy",
            )
            block_start = content.index(POLICIES_BEGIN) + len(POLICIES_BEGIN)
            block_text = content[block_start : block_end - len(POLICIES_END)]
        except ValueError:
            return PolicyBlockRefreshResult((), (), scaffolded)

    registry_path = policy_registry_path(repo_root)
    if not registry_path.exists():
        return PolicyBlockRefreshResult((), (), scaffolded)
    payload = _read_yaml(registry_path)
    policies = payload.get("policies", {})
    if not isinstance(policies, dict) or not policies:
        raise ValueError(
            "Policy registry payload is invalid; expected non-empty "
            f"`policies` mapping in {registry_path}."
        )

    previous_sections = _section_map(block_text)
    generated_sections: Dict[str, str] = {}
    skipped: List[str] = []
    entries: List[_PolicyEntry] = []
    for policy_id in sorted(policies):
        payload_entry = policies.get(policy_id, {})
        if not isinstance(payload_entry, dict):
            skipped.append(policy_id)
            continue
        order, values = _metadata_from_registry(
            policy_id,
            payload_entry.get("metadata"),
        )
        rendered = metadata_runtime.render_metadata_block(order, values)
        heading_name = (
            str(payload_entry.get("description", "")).strip()
            or policy_id.replace("-", " ").title()
        )
        heading = f"## Policy: {heading_name}\n\n"
        description = str(payload_entry.get("policy_text", "")).strip()
        if not description:
            descriptor = load_policy_descriptor(repo_root, policy_id)
            try:
                description = _descriptor_text_or_error(descriptor, policy_id)
            except ValueError:
                skipped.append(policy_id)
                continue
        final_text = (
            f"{heading}```policy-def\n{rendered}\n```\n\n{description}\n"
        )
        generated_sections[policy_id] = final_text
        entries.append(_PolicyEntry(policy_id=policy_id, text=final_text))

    if not entries:
        return PolicyBlockRefreshResult((), tuple(skipped), scaffolded)

    new_block = _assemble_sections(entries)
    updated = new_block.strip() != block_text.strip()
    changed_file = scaffolded or updated
    if updated:
        prefix = content[:block_start]
        suffix = content[block_end - len(POLICIES_END) :]
        rebuilt = (
            f"{prefix}\n{new_block.rstrip()}\n{suffix}"
            if not prefix.endswith("\n")
            else f"{prefix}{new_block.rstrip()}\n{suffix}"
        )
        agents_path.write_text(rebuilt, encoding="utf-8")
    changed = sorted(
        {*previous_sections.keys(), *generated_sections.keys()}
        - {
            policy_id
            for policy_id, section in previous_sections.items()
            if generated_sections.get(policy_id, "").strip() == section.strip()
        }
    )
    return PolicyBlockRefreshResult(
        tuple(changed),
        tuple(skipped),
        changed_file,
    )
