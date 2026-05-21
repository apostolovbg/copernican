"""Project governance state, headings, and identity rendering."""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

import devcovenant.core.repository_paths as repository_paths

_DEFAULT_STAGES = (
    "prototype",
    "alpha",
    "beta",
    "stable",
    "deprecated",
    "archived",
)
_DEFAULT_MAINTENANCE_STANCES = (
    "active",
    "maintenance",
    "frozen",
    "sunset",
)
_ALLOWED_COMPATIBILITY_POLICIES = {
    "backward-compatible",
    "breaking-allowed",
    "forward-only",
    "unspecified",
}
_ALLOWED_VERSIONING_MODES = {"versioned", "unversioned"}
_DEFAULT_UNVERSIONED_LABEL = "Unversioned"
_DEFAULT_UNRELEASED_HEADING = "## Unreleased"
_DEFAULT_CHANGELOG_FILE = "CHANGELOG.md"
_DEFAULT_PROJECT_NAME = "Project Name"
_DEFAULT_PROJECT_DESCRIPTION = (
    "Describe the project: what it does, who it helps, and what problem "
    "it solves."
)
_DEFAULT_COPYRIGHT_NOTICE = "YEAR Legal Owner Name"
_LOG_MARKER = "## Log changes here"
_MANAGED_BEGIN = "<!-- DEVCOV:BEGIN -->"
_MANAGED_END = "<!-- DEVCOV:END -->"
_COMPATIBILITY_POLICY_GUIDANCE = {
    "backward-compatible": (
        "Preserve the current public contract. Add compatibility bridges "
        "only when they are intentional, documented, and tested."
    ),
    "breaking-allowed": (
        "Compatibility is optional. Do not imply support you do not intend "
        "to keep."
    ),
    "forward-only": (
        "Do not leave legacy fallbacks behind. Remove deprecated readers, "
        "aliases, and bridge paths instead of preserving them."
    ),
    "unspecified": (
        "No compatibility promise is implied. Make contract changes explicit "
        "before code or docs start depending on them."
    ),
}


def _project_name_path_value(raw_name: str) -> str:
    """Return one filesystem-safe default package-path token."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(raw_name or "").strip())
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("._")
    return normalized or "project"


@dataclass(frozen=True)
class ProjectGovernanceState:
    """Resolved project-governance state for one repository runtime."""

    enabled: bool = True
    project_name: str = _DEFAULT_PROJECT_NAME
    project_description: str = _DEFAULT_PROJECT_DESCRIPTION
    copyright_notice: str = _DEFAULT_COPYRIGHT_NOTICE
    stage: str = ""
    maintenance_stance: str = ""
    compatibility_policy: str = "unspecified"
    versioning_mode: str = "versioned"
    codename: str = ""
    build_identity: str = ""
    unversioned_label: str = _DEFAULT_UNVERSIONED_LABEL
    unreleased_heading: str = _DEFAULT_UNRELEASED_HEADING
    changelog_file: str = _DEFAULT_CHANGELOG_FILE
    allowed_stages: tuple[str, ...] = _DEFAULT_STAGES
    allowed_maintenance_stances: tuple[str, ...] = _DEFAULT_MAINTENANCE_STANCES

    @property
    def is_unversioned(self) -> bool:
        """Return True when the repository is intentionally unversioned."""
        return self.versioning_mode == "unversioned"

    def displayed_project_version(self, declared_version: str) -> str:
        """Return the rendered Project Version header value."""
        if self.is_unversioned:
            return self.unversioned_label
        token = str(declared_version or "").strip()
        if token:
            return token
        raise ValueError(
            "Versioned repository is missing a declared project version."
        )

    def governance_header_lines(self) -> list[str]:
        """Return managed-doc governance header lines for opted-in docs."""
        lines = [
            f"**Project Stage:** {self.stage}",
            f"**Maintenance Stance:** {self.maintenance_stance}",
            f"**Compatibility Policy:** {self.compatibility_policy}",
            f"**Versioning Mode:** {self.versioning_mode}",
        ]
        if self.codename:
            lines.append(f"**Project Codename:** {self.codename}")
        if self.build_identity:
            lines.append(f"**Build Identity:** {self.build_identity}")
        return lines

    def section_lines(self, declared_version: str) -> list[str]:
        """Return the AGENTS project-governance section lines."""
        lines = [
            "## Project Governance",
            (
                "This block reflects the repository's active "
                "project-governance state."
            ),
            (
                "- Project Version: "
                f"{self.displayed_project_version(declared_version)}"
            ),
            f"- Project Stage: {self.stage}",
            f"- Maintenance Stance: {self.maintenance_stance}",
            f"- Compatibility Policy: {self.compatibility_policy}",
            f"- Versioning Mode: {self.versioning_mode}",
        ]
        lines.extend(self.compatibility_policy_guidance_lines())
        if self.codename:
            lines.append(f"- Project Codename: {self.codename}")
        if self.build_identity:
            lines.append(f"- Build Identity: {self.build_identity}")
        return lines

    def compatibility_policy_guidance(self) -> str:
        """Return the AGENTS guidance text for the active policy."""
        return _COMPATIBILITY_POLICY_GUIDANCE[self.compatibility_policy]

    def compatibility_policy_guidance_lines(self) -> list[str]:
        """Render wrapped AGENTS lines for compatibility guidance."""
        wrapped = textwrap.wrap(
            self.compatibility_policy_guidance(),
            width=72,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        return ["- Compatibility Guidance:", *wrapped]

    def registry_payload(self, declared_version: str) -> dict[str, object]:
        """Return a deterministic registry mapping for project governance."""
        payload: dict[str, object] = {
            "project_name": self.project_name,
            "project_description": self.project_description,
            "copyright_notice": self.copyright_notice,
            "project_version": self.displayed_project_version(
                declared_version
            ),
            "stage": self.stage,
            "maintenance_stance": self.maintenance_stance,
            "compatibility_policy": self.compatibility_policy,
            "versioning_mode": self.versioning_mode,
            "unversioned_label": self.unversioned_label,
            "unreleased_heading": self.unreleased_heading,
            "changelog_file": self.changelog_file,
            "release_headings": release_headings_for_state(self),
            "allowed_stages": list(self.allowed_stages),
            "allowed_maintenance_stances": list(
                self.allowed_maintenance_stances
            ),
        }
        if self.codename:
            payload["codename"] = self.codename
        if self.build_identity:
            payload["build_identity"] = self.build_identity
        return payload


def resolve_runtime_state(
    repo_root: Path,
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> ProjectGovernanceState:
    """Return validated project-governance state for one repo runtime."""
    repo_root = Path(repo_root).resolve()
    payload = _load_runtime_config(repo_root, config_payload)
    raw_block = payload.get("project-governance")
    if not isinstance(raw_block, dict):
        raise ValueError(
            "Configure `project-governance` as a mapping in "
            "`devcovenant/config.yaml`."
        )

    project_name = (
        _string_option(raw_block, "project_name") or _DEFAULT_PROJECT_NAME
    )
    project_description = (
        _string_option(raw_block, "project_description")
        or _DEFAULT_PROJECT_DESCRIPTION
    )
    copyright_notice = (
        _string_option(raw_block, "copyright_notice")
        or _DEFAULT_COPYRIGHT_NOTICE
    )
    stage = _required_string(raw_block, "stage")
    maintenance_stance = _required_string(raw_block, "maintenance_stance")
    compatibility_policy = _required_string(
        raw_block,
        "compatibility_policy",
    ).lower()
    versioning_mode = _required_string(raw_block, "versioning_mode").lower()
    if compatibility_policy not in _ALLOWED_COMPATIBILITY_POLICIES:
        raise ValueError(
            "`project-governance.compatibility_policy` must be one of: "
            + ", ".join(sorted(_ALLOWED_COMPATIBILITY_POLICIES))
            + "."
        )
    if versioning_mode not in _ALLOWED_VERSIONING_MODES:
        raise ValueError(
            "`project-governance.versioning_mode` must be `versioned` "
            "or `unversioned`."
        )

    allowed_stages = tuple(
        _normalized_list(
            raw_block.get("allowed_stages"),
            default=_DEFAULT_STAGES,
        )
    )
    if stage not in allowed_stages:
        raise ValueError(
            "`project-governance.stage` must be one of: "
            + ", ".join(allowed_stages)
            + "."
        )

    allowed_maintenance_stances = tuple(
        _normalized_list(
            raw_block.get("allowed_maintenance_stances"),
            default=_DEFAULT_MAINTENANCE_STANCES,
        )
    )
    if maintenance_stance not in allowed_maintenance_stances:
        raise ValueError(
            "`project-governance.maintenance_stance` must be one of: "
            + ", ".join(allowed_maintenance_stances)
            + "."
        )

    changelog_file = (
        _string_option(raw_block, "changelog_file") or _DEFAULT_CHANGELOG_FILE
    )
    return ProjectGovernanceState(
        project_name=project_name,
        project_description=project_description,
        copyright_notice=copyright_notice,
        stage=stage,
        maintenance_stance=maintenance_stance,
        compatibility_policy=compatibility_policy,
        versioning_mode=versioning_mode,
        codename=_string_option(raw_block, "codename"),
        build_identity=_string_option(raw_block, "build_identity"),
        unversioned_label=(
            _string_option(raw_block, "unversioned_label")
            or _DEFAULT_UNVERSIONED_LABEL
        ),
        unreleased_heading=(
            _string_option(raw_block, "unreleased_heading")
            or _DEFAULT_UNRELEASED_HEADING
        ),
        changelog_file=changelog_file,
        allowed_stages=allowed_stages,
        allowed_maintenance_stances=allowed_maintenance_stances,
    )


def render_identity_placeholders(
    text: str,
    state: ProjectGovernanceState,
    *,
    project_version: str = "",
) -> str:
    """Render project-identity placeholders from governance state."""
    rendered = str(text or "")
    replacements = {
        "{{ PROJECT_NAME }}": state.project_name,
        "{{ PROJECT_NAME_PATH }}": _project_name_path_value(
            state.project_name
        ),
        "{{ PROJECT_VERSION }}": str(project_version or "").strip(),
        "{{ PROJECT_DESCRIPTION }}": state.project_description,
        "{{ COPYRIGHT_NOTICE }}": state.copyright_notice,
        "{{ PROJECT_DESCRIPTION_PARAGRAPH }}": textwrap.fill(
            state.project_description,
            width=72,
            break_long_words=False,
            break_on_hyphens=False,
        ),
        "{{ PROJECT_DESCRIPTION_TOML }}": render_toml_string(
            state.project_description
        ),
    }
    for placeholder, value in replacements.items():
        rendered = rendered.replace(placeholder, value)
    return rendered


def render_toml_string(value: str) -> str:
    """Render *value* as a TOML string while keeping source lines short."""
    if len(value) <= 60:
        return yaml.safe_dump(value, default_style='"').strip()
    wrapped = textwrap.wrap(
        value,
        width=56,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return '""'
    parts = []
    for index, line in enumerate(wrapped):
        if index < len(wrapped) - 1:
            parts.append(f"{line} \\")
        else:
            parts.append(line)
    body = "\n".join(parts)
    return f'"""\n{body}"""'


def resolve_release_headings(
    repo_root: Path,
    *,
    config_payload: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return active changelog release headings for one repo runtime."""
    state = resolve_runtime_state(
        repo_root,
        config_payload=config_payload,
    )
    validate_changelog_contract(
        repo_root,
        config_payload=config_payload,
        state=state,
    )
    return release_headings_for_state(state)


def release_headings_for_state(
    state: ProjectGovernanceState,
) -> list[str]:
    """Return active changelog headings from one resolved state."""
    if state.is_unversioned:
        return [state.unreleased_heading]
    return ["## Version"]


def validate_changelog_contract(
    repo_root: Path,
    *,
    config_payload: Mapping[str, Any] | None = None,
    state: ProjectGovernanceState | None = None,
) -> None:
    """Validate the changelog contract implied by project-governance."""
    repo_root = Path(repo_root).resolve()
    runtime_state = state or resolve_runtime_state(
        repo_root,
        config_payload=config_payload,
    )
    if not runtime_state.is_unversioned:
        return
    changelog_path = repo_root / runtime_state.changelog_file
    if not changelog_path.exists():
        raise ValueError(
            "Configured project-governance changelog file is missing."
        )
    top_heading = _top_visible_release_heading(changelog_path)
    if top_heading != runtime_state.unreleased_heading:
        raise ValueError(
            "Unversioned project-governance requires the top changelog "
            f"heading to be `{runtime_state.unreleased_heading}`."
        )


def _load_runtime_config(
    repo_root: Path,
    config_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return runtime config payload, defaulting to repo config.yaml."""
    if config_payload:
        return dict(config_payload)
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = repository_paths.load_yaml(config_path)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load runtime config: {exc}") from exc
    if isinstance(payload, dict):
        return payload
    raise ValueError("Runtime config must be a YAML mapping.")


def _string_option(raw_block: Mapping[str, Any], key: str) -> str:
    """Return one normalized string option from the config block."""
    return str(raw_block.get(key, "") or "").strip()


def _required_string(raw_block: Mapping[str, Any], key: str) -> str:
    """Return one required string config option or raise a clear error."""
    token = _string_option(raw_block, key)
    if token:
        return token
    raise ValueError(
        f"Configure `project-governance.{key}` explicitly in "
        "`devcovenant/config.yaml`."
    )


def _normalized_list(
    raw: object,
    *,
    default: Iterable[str],
) -> list[str]:
    """Return a normalized non-empty string list."""
    if isinstance(raw, str):
        items = [entry.strip() for entry in raw.split(",") if entry.strip()]
    elif isinstance(raw, list):
        items = [str(entry).strip() for entry in raw if str(entry).strip()]
    else:
        items = []
    return items or [
        str(entry).strip() for entry in default if str(entry).strip()
    ]


def _visible_changelog_lines(changelog_text: str) -> list[str]:
    """Return changelog lines outside managed blocks and fenced examples."""
    start = changelog_text.find(_LOG_MARKER)
    content = changelog_text[start:] if start >= 0 else changelog_text
    visible: list[str] = []
    in_managed = False
    in_fence = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == _LOG_MARKER:
            continue
        if stripped == _MANAGED_BEGIN:
            in_managed = True
            continue
        if stripped == _MANAGED_END:
            in_managed = False
            continue
        if in_managed:
            continue
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped:
            visible.append(stripped)
    return visible


def _top_visible_release_heading(changelog_path: Path) -> str:
    """Return the top visible release heading from a changelog."""
    content = changelog_path.read_text(encoding="utf-8")
    for line in _visible_changelog_lines(content):
        if line.startswith("## "):
            return line
    return ""
