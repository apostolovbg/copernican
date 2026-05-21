"""Managed-doc runtime service for descriptor loading, rendering, and sync."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml
from packaging.version import InvalidVersion, Version

import devcovenant.core.agents_blocks as agents_blocks_lib
import devcovenant.core.profile_registry as profile_registry_service
import devcovenant.core.project_governance as project_governance_service
import devcovenant.core.repository_paths as yaml_cache_service

ProjectGovernanceState = project_governance_service.ProjectGovernanceState

BLOCK_BEGIN = "<!-- DEVCOV:BEGIN -->"
BLOCK_END = "<!-- DEVCOV:END -->"
WORKFLOW_BEGIN = "<!-- DEVCOV-WORKFLOW:BEGIN -->"
WORKFLOW_END = "<!-- DEVCOV-WORKFLOW:END -->"
POLICIES_BEGIN = agents_blocks_lib.POLICIES_BEGIN
POLICIES_END = agents_blocks_lib.POLICIES_END
USER_PRESERVE_BEGIN = "<!-- DEVCOV-USER-PRESERVE:BEGIN -->"
USER_PRESERVE_END = "<!-- DEVCOV-USER-PRESERVE:END -->"
DOC_ID_LABEL = "**Doc ID:**"
DOC_TYPE_LABEL = "**Doc Type:**"
PROJECT_VERSION_LABEL = "**Project Version:**"
PROJECT_STAGE_LABEL = "**Project Stage:**"
MAINTENANCE_STANCE_LABEL = "**Maintenance Stance:**"
COMPATIBILITY_POLICY_LABEL = "**Compatibility Policy:**"
VERSIONING_MODE_LABEL = "**Versioning Mode:**"
PROJECT_CODENAME_LABEL = "**Project Codename:**"
BUILD_IDENTITY_LABEL = "**Build Identity:**"
LAST_UPDATED_LABEL = "**Last Updated:**"
DEVCOV_VERSION_LABEL = "**DevCovenant Version:**"

_MANAGED_DOC_DESCRIPTOR_KEYS = frozenset(
    {
        "title",
        "target_path",
        "doc_id",
        "doc_type",
        "project_version",
        "last_updated",
        "devcovenant_version",
        "project_governance_headers",
        "import_seed",
        "authoritative_source",
        "managed_block",
        "body",
        "workflow_block",
    }
)
_MANAGED_DOC_MULTILINE_KEYS = ("managed_block", "body", "workflow_block")
_MANAGED_DOC_REQUIRED_KEYS = (
    "title",
    "target_path",
    "doc_id",
    "doc_type",
    "project_version",
    "last_updated",
    "devcovenant_version",
    "managed_block",
    "body",
)
_MANAGED_DOC_OPTIONAL_BOOLEAN_KEYS = (
    "project_governance_headers",
    "import_seed",
    "authoritative_source",
)
_MANAGED_DOC_REQUIRED_BOOLEAN_KEYS = (
    "project_version",
    "last_updated",
    "devcovenant_version",
)


def utc_today() -> str:
    """Return the current UTC date."""
    return datetime.now(timezone.utc).date().isoformat()


def normalize_doc_name(name: str) -> str:
    """Normalize configured doc names to canonical markdown paths."""
    raw = str(name or "").strip()
    if not raw:
        return ""
    mapping = {
        "AGENTS": "AGENTS.md",
        "README": "README.md",
        "CONTRIBUTING": "CONTRIBUTING.md",
        "SPEC": "SPEC.md",
        "PLAN": "PLAN.md",
        "CHANGELOG": "CHANGELOG.md",
    }
    upper = raw.upper()
    if upper in mapping:
        return mapping[upper]
    if upper.endswith(".MD") and upper[:-3] in mapping:
        return mapping[upper[:-3]]
    return raw


def descriptor_target_path(descriptor: dict[str, object]) -> str:
    """Return the normalized target path declared by one descriptor."""
    raw_value = descriptor.get("target_path")
    if not isinstance(raw_value, str):
        raise ValueError(
            "Managed doc descriptor field `target_path` must be a string."
        )
    normalized = normalize_doc_name(raw_value)
    if not normalized:
        raise ValueError(
            "Managed doc descriptor field `target_path` must be non-empty."
        )
    return normalized


def install_import_managed_docs(config: dict[str, object]) -> set[str]:
    """Return install-recorded managed-doc import seeds from config."""
    install_block = config.get("install")
    if not isinstance(install_block, dict):
        return set()
    raw_docs = install_block.get("import_managed_docs")
    if not isinstance(raw_docs, list):
        return set()
    selected: set[str] = set()
    for item in raw_docs:
        normalized = normalize_doc_name(str(item))
        if normalized:
            selected.add(normalized)
    return selected


def managed_docs_from_config(config: dict[str, object]) -> list[str]:
    """Resolve autogen managed docs from config doc_assets."""
    doc_assets = config.get("doc_assets")
    if not isinstance(doc_assets, dict):
        raise ValueError(
            "`doc_assets` must be a mapping in devcovenant/config.yaml."
        )

    raw_autogen = doc_assets.get("autogen")
    raw_user = doc_assets.get("user")
    if not isinstance(raw_autogen, list):
        raise ValueError("`doc_assets.autogen` must be a list.")
    if not isinstance(raw_user, list):
        raise ValueError("`doc_assets.user` must be a list.")

    autogen = [normalize_doc_name(item) for item in raw_autogen]
    autogen = [doc for doc in autogen if doc]
    if not autogen:
        raise ValueError(
            "`doc_assets.autogen` must contain at least one document."
        )

    user_docs = {normalize_doc_name(item) for item in raw_user if item}
    selected = [doc for doc in autogen if doc and doc not in user_docs]
    if not selected:
        raise ValueError(
            "`doc_assets.autogen` resolved to no documents after "
            "excluding `doc_assets.user` entries."
        )
    if "AGENTS.md" not in selected:
        raise ValueError(
            "`doc_assets.autogen` must include AGENTS.md as a managed doc."
        )

    ordered: list[str] = []
    for doc in selected:
        if doc not in ordered:
            ordered.append(doc)
    return ordered


def managed_doc_assets_root(repo_root: Path) -> Path:
    """Return the active global managed-doc assets root."""
    builtin_assets_dir = (
        repo_root
        / "devcovenant"
        / "builtin"
        / "profiles"
        / "global"
        / "assets"
    )
    core_assets_dir = (
        repo_root / "devcovenant" / "core" / "profiles" / "global" / "assets"
    )
    return (
        builtin_assets_dir if builtin_assets_dir.exists() else core_assets_dir
    )


def _read_repo_config_payload(repo_root: Path) -> dict[str, object]:
    """Load the repository config payload when it exists."""
    config_path = repo_root / "devcovenant" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        payload = yaml_cache_service.load_yaml(config_path)
    except (OSError, yaml.YAMLError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _effective_repo_config_payload(
    repo_root: Path,
    config_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return caller config when present, otherwise load the repo config."""
    if isinstance(config_payload, dict) and config_payload:
        return config_payload
    return _read_repo_config_payload(repo_root)


def managed_doc_assets_roots(
    repo_root: Path,
    *,
    config_payload: dict[str, object] | None = None,
) -> list[Path]:
    """Return descriptor roots for builtin global and active profiles."""
    roots: list[Path] = [managed_doc_assets_root(repo_root)]
    payload = _effective_repo_config_payload(repo_root, config_payload)
    active_profiles = profile_registry_service.parse_active_profiles(
        payload,
        include_global=False,
    )
    profile_base_dirs = (
        repo_root / "devcovenant" / "builtin" / "profiles",
        repo_root / "devcovenant" / "custom" / "profiles",
    )
    for profile_name in active_profiles:
        for base_dir in profile_base_dirs:
            assets_root = base_dir / profile_name / "assets"
            if assets_root.exists() and assets_root not in roots:
                roots.append(assets_root)
    return roots


def source_managed_doc_assets_roots(source_dir: Path) -> list[Path]:
    """Return descriptor roots bundled under one source package tree."""
    roots: list[Path] = []
    global_assets = source_dir / "builtin" / "profiles" / "global" / "assets"
    if global_assets.exists():
        roots.append(global_assets)
    profile_base_dirs = (
        source_dir / "builtin" / "profiles",
        source_dir / "custom" / "profiles",
    )
    for base_dir in profile_base_dirs:
        if not base_dir.exists():
            continue
        for profile_dir in sorted(base_dir.iterdir()):
            if (
                not profile_dir.is_dir()
                or profile_dir.name.startswith(".")
                or profile_dir.name == "global"
            ):
                continue
            assets_root = profile_dir / "assets"
            if assets_root.exists() and assets_root not in roots:
                roots.append(assets_root)
    return roots


def descriptor_path_from_assets_root(assets_root: Path, doc_name: str) -> Path:
    """Resolve the YAML descriptor path for one doc from one assets root."""
    doc_path = Path(doc_name)
    if doc_path.parent != Path("."):
        return assets_root / doc_path.with_suffix(".yaml")
    return assets_root / f"{doc_path.stem}.yaml"


def descriptor_path(
    repo_root: Path,
    doc_name: str,
    *,
    config_payload: dict[str, object] | None = None,
) -> Path:
    """Resolve the managed-doc descriptor path for one repository doc."""
    normalized_doc = normalize_doc_name(doc_name)
    for entry in managed_doc_descriptor_entries(
        repo_root,
        config_payload=config_payload,
    ):
        if str(entry["doc"]) == normalized_doc:
            return Path(str(entry["descriptor_path"]))
    raise ValueError(f"Missing managed doc descriptor for `{normalized_doc}`.")


def _yaml_scalar_style_token(raw_yaml: str, key: str) -> str:
    """Return the inline scalar style token from one YAML key line."""
    pattern = re.compile(
        rf"(?m)^[ \t]*{re.escape(key)}[ \t]*:[ \t]*(?P<token>[^\n]*)$"
    )
    match = pattern.search(raw_yaml)
    if match is None:
        return ""
    return str(match.group("token") or "").strip()


def _require_literal_block_scalar(
    descriptor_path_value: Path,
    *,
    doc_name: str,
    field_name: str,
    field_value: str,
    raw_yaml: str,
) -> None:
    """Require literal block scalar style for multiline descriptor fields."""
    if "\n" not in field_value:
        return
    style_token = _yaml_scalar_style_token(raw_yaml, field_name)
    if style_token.startswith("|"):
        return
    raise ValueError(
        "Managed doc descriptor "
        f"`{descriptor_path_value}` field `{field_name}` in `{doc_name}` "
        "contains multiline text and must use YAML literal block style "
        f"(`{field_name}: |-`)."
    )


def validate_managed_doc_descriptor(
    descriptor: dict[str, object],
    *,
    descriptor_path_value: Path,
    doc_name: str,
    raw_yaml: str,
) -> None:
    """Validate managed-doc descriptor schema and multiline style rules."""
    descriptor_keys = [str(key) for key in descriptor.keys()]
    unknown_keys = sorted(
        str(key)
        for key in descriptor_keys
        if str(key) not in _MANAGED_DOC_DESCRIPTOR_KEYS
    )
    if unknown_keys:
        raise ValueError(
            "Managed doc descriptor "
            f"`{descriptor_path_value}` has unsupported keys: "
            f"{', '.join(unknown_keys)}."
        )

    required_prefix = [
        "title",
        "target_path",
        "doc_id",
        "doc_type",
        "project_version",
        "last_updated",
        "devcovenant_version",
    ]
    for field_name in _MANAGED_DOC_OPTIONAL_BOOLEAN_KEYS:
        if field_name in descriptor:
            required_prefix.append(field_name)
    required_prefix.extend(["managed_block", "body"])
    if descriptor_keys[: len(required_prefix)] != required_prefix:
        raise ValueError(
            "Managed doc descriptor "
            f"`{descriptor_path_value}` must declare keys in this order: "
            f"{', '.join(required_prefix)}."
        )

    for field_name in _MANAGED_DOC_REQUIRED_KEYS:
        if field_name not in descriptor:
            raise ValueError(
                "Managed doc descriptor "
                f"`{descriptor_path_value}` is missing required key "
                f"`{field_name}`."
            )

    for field_name in (
        "title",
        "target_path",
        "doc_id",
        "doc_type",
        "managed_block",
        "body",
        "workflow_block",
    ):
        raw_value = descriptor.get(field_name)
        if raw_value is None:
            continue
        if not isinstance(raw_value, str):
            raise ValueError(
                "Managed doc descriptor "
                f"`{descriptor_path_value}` field `{field_name}` must be "
                "a string."
            )
        if field_name in _MANAGED_DOC_MULTILINE_KEYS:
            _require_literal_block_scalar(
                descriptor_path_value,
                doc_name=doc_name,
                field_name=field_name,
                field_value=raw_value,
                raw_yaml=raw_yaml,
            )

    for field_name in ("title", "target_path", "doc_id", "doc_type"):
        if not str(descriptor.get(field_name, "")).strip():
            raise ValueError(
                "Managed doc descriptor "
                f"`{descriptor_path_value}` field `{field_name}` must be "
                "non-empty."
            )

    for field_name in _MANAGED_DOC_REQUIRED_BOOLEAN_KEYS:
        raw_value = descriptor.get(field_name)
        if not isinstance(raw_value, bool):
            raise ValueError(
                "Managed doc descriptor "
                f"`{descriptor_path_value}` field `{field_name}` must be "
                "boolean."
            )
    for field_name in _MANAGED_DOC_OPTIONAL_BOOLEAN_KEYS:
        if field_name not in descriptor:
            continue
        raw_value = descriptor.get(field_name)
        if not isinstance(raw_value, bool):
            raise ValueError(
                "Managed doc descriptor "
                f"`{descriptor_path_value}` field `{field_name}` must be "
                "boolean."
            )
    doc_type = str(descriptor.get("doc_type", "")).strip()
    if (
        descriptor.get("devcovenant_version") is not True
        and doc_type != "license"
    ):
        raise ValueError(
            "Managed doc descriptor "
            f"`{descriptor_path_value}` field `devcovenant_version` must "
            "be true."
        )


def load_managed_doc_descriptor(
    descriptor_path_value: Path,
    *,
    doc_name: str,
) -> dict[str, object]:
    """Load and validate one managed-doc descriptor payload."""
    if not descriptor_path_value.exists():
        raise ValueError(
            "Missing managed doc descriptor for "
            f"`{doc_name}`: {descriptor_path_value}"
        )
    try:
        raw_yaml = yaml_cache_service.read_text(descriptor_path_value)
    except OSError as exc:
        raise ValueError(
            "Unable to read managed doc descriptor "
            f"{descriptor_path_value}: {exc}"
        ) from exc
    try:
        payload = yaml_cache_service.load_yaml(descriptor_path_value)
    except yaml.YAMLError as exc:
        raise ValueError(
            "Invalid YAML in managed doc descriptor "
            f"{descriptor_path_value}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            "Managed doc descriptor "
            f"`{descriptor_path_value}` must contain a YAML mapping."
        )
    validate_managed_doc_descriptor(
        payload,
        descriptor_path_value=descriptor_path_value,
        doc_name=doc_name,
        raw_yaml=raw_yaml,
    )
    expected_target = normalize_doc_name(doc_name)
    actual_target = descriptor_target_path(payload)
    if actual_target != expected_target:
        raise ValueError(
            "Managed doc descriptor "
            f"`{descriptor_path_value}` targets `{actual_target}` but was "
            f"loaded for `{expected_target}`."
        )
    return payload


def descriptor_bool(descriptor: dict[str, object], field_name: str) -> bool:
    """Return a required boolean field from a managed descriptor."""
    raw_value = descriptor.get(field_name)
    if not isinstance(raw_value, bool):
        raise ValueError(
            f"Managed doc descriptor field `{field_name}` must be boolean."
        )
    return raw_value


def descriptor_optional_bool(
    descriptor: dict[str, object],
    field_name: str,
) -> bool:
    """Return an optional boolean field from a managed descriptor."""
    raw_value = descriptor.get(field_name)
    if raw_value is None:
        return False
    if not isinstance(raw_value, bool):
        raise ValueError(
            f"Managed doc descriptor field `{field_name}` must be boolean."
        )
    return raw_value


def descriptor_import_seed_enabled(descriptor: dict[str, object]) -> bool:
    """Return whether one managed doc participates in seed import."""
    return descriptor_optional_bool(descriptor, "import_seed")


def descriptor_is_authoritative_source(
    descriptor: dict[str, object],
) -> bool:
    """Return whether one managed doc is authoritative for asset sync."""
    return descriptor_optional_bool(descriptor, "authoritative_source")


def _managed_doc_descriptor_entries_from_root(
    assets_root: Path,
) -> list[dict[str, object]]:
    """Return validated managed-doc descriptors discovered under one root."""
    entries: list[dict[str, object]] = []
    for descriptor_file in sorted(assets_root.rglob("*.yaml")):
        try:
            raw_yaml = yaml_cache_service.read_text(descriptor_file)
        except OSError:
            continue
        try:
            payload = yaml_cache_service.load_yaml(descriptor_file)
        except yaml.YAMLError:
            continue
        if not isinstance(payload, dict):
            continue
        if not all(key in payload for key in _MANAGED_DOC_REQUIRED_KEYS):
            continue
        doc_name = descriptor_target_path(payload)
        validate_managed_doc_descriptor(
            payload,
            descriptor_path_value=descriptor_file,
            doc_name=doc_name,
            raw_yaml=raw_yaml,
        )
        entries.append(
            {
                "doc": doc_name,
                "descriptor_path": descriptor_file,
                "descriptor": payload,
            }
        )
    return entries


def managed_doc_descriptor_entries_from_roots(
    assets_roots: list[Path],
) -> list[dict[str, object]]:
    """Return winning managed-doc descriptors by precedence-ordered root."""
    discovered: dict[str, dict[str, object]] = {}
    for assets_root in assets_roots:
        for entry in _managed_doc_descriptor_entries_from_root(assets_root):
            doc_name = str(entry["doc"])
            discovered[doc_name] = entry
    return [discovered[key] for key in sorted(discovered)]


def managed_doc_descriptor_entries(
    repo_root: Path,
    *,
    config_payload: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return validated managed-doc descriptors for one repository."""
    return managed_doc_descriptor_entries_from_roots(
        managed_doc_assets_roots(
            repo_root,
            config_payload=config_payload,
        )
    )


def authoritative_managed_doc_entries(
    repo_root: Path,
    *,
    config_payload: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Return authoritative doc/descriptor entries for asset-sync checks."""
    payload = _effective_repo_config_payload(repo_root, config_payload)
    enabled_docs = set(managed_docs_from_config(payload)) if payload else set()
    entries = managed_doc_descriptor_entries(
        repo_root,
        config_payload=payload,
    )
    return [
        entry
        for entry in entries
        if descriptor_is_authoritative_source(entry["descriptor"])
        and (not enabled_docs or str(entry["doc"]) in enabled_docs)
    ]


def normalize_body_text(text: str) -> str:
    """Return normalized managed-doc body text for fingerprinting."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip(
        "\n"
    )


def body_fingerprint(text: str) -> str:
    """Return SHA256 fingerprint for normalized body text."""
    return hashlib.sha256(
        normalize_body_text(text).encode("utf-8")
    ).hexdigest()


def descriptor_body_fingerprint(descriptor: dict[str, object]) -> str:
    """Return the current descriptor body fingerprint."""
    return body_fingerprint(str(descriptor.get("body", "")))


def managed_docs_registry_payload(
    repo_root: Path,
    *,
    config_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the tracked registry view for managed-doc descriptors."""
    payload = _effective_repo_config_payload(repo_root, config_payload)
    enabled_docs = managed_docs_from_config(payload) if payload else []
    enabled_set = set(enabled_docs)
    descriptor_roots = [
        str(path.relative_to(repo_root)).replace("\\", "/")
        for path in managed_doc_assets_roots(
            repo_root,
            config_payload=payload,
        )
    ]
    descriptors: dict[str, dict[str, object]] = {}
    for entry in managed_doc_descriptor_entries(
        repo_root,
        config_payload=payload,
    ):
        doc_name = str(entry["doc"])
        descriptor = entry["descriptor"]
        descriptor_path_value = Path(str(entry["descriptor_path"]))
        descriptors[doc_name] = {
            "enabled": doc_name in enabled_set,
            "descriptor_path": str(
                descriptor_path_value.relative_to(repo_root)
            ).replace("\\", "/"),
            "project_governance_headers": descriptor_optional_bool(
                descriptor,
                "project_governance_headers",
            ),
            "import_seed": descriptor_import_seed_enabled(descriptor),
            "authoritative_source": descriptor_is_authoritative_source(
                descriptor
            ),
            "body_fingerprint": descriptor_body_fingerprint(descriptor),
        }
    return {
        "descriptor_roots": descriptor_roots,
        "enabled_docs": enabled_docs,
        "descriptors": descriptors,
    }


def render_project_governance_header_lines(
    state: ProjectGovernanceState,
) -> list[str]:
    """Return generated project-governance header lines."""
    return [
        f"{label} {value}"
        for label, value in (
            (PROJECT_STAGE_LABEL, state.stage),
            (MAINTENANCE_STANCE_LABEL, state.maintenance_stance),
            (COMPATIBILITY_POLICY_LABEL, state.compatibility_policy),
            (VERSIONING_MODE_LABEL, state.versioning_mode),
            (PROJECT_CODENAME_LABEL, state.codename),
            (BUILD_IDENTITY_LABEL, state.build_identity),
        )
        if str(value).strip()
    ]


def project_governance_labels(
    state: ProjectGovernanceState,
) -> list[str]:
    """Return the governance header labels expected for one state."""
    labels = [
        PROJECT_STAGE_LABEL,
        MAINTENANCE_STANCE_LABEL,
        COMPATIBILITY_POLICY_LABEL,
        VERSIONING_MODE_LABEL,
    ]
    if state.codename:
        labels.append(PROJECT_CODENAME_LABEL)
    if state.build_identity:
        labels.append(BUILD_IDENTITY_LABEL)
    return labels


def render_project_governance_section_block(
    *,
    project_version: str,
    project_governance_state: ProjectGovernanceState,
) -> str:
    """Render the AGENTS project-governance section block."""
    return render_block(
        BLOCK_BEGIN,
        BLOCK_END,
        "\n".join(project_governance_state.section_lines(project_version)),
    )


def render_generated_header(
    doc_name: str,
    descriptor: dict[str, object],
    *,
    project_version: str,
    devcovenant_version: str,
    project_governance_state: ProjectGovernanceState,
) -> list[str]:
    """Render deterministic top-of-doc header lines from descriptor keys."""
    doc_type = str(descriptor.get("doc_type", "")).strip()
    title = project_governance_service.render_identity_placeholders(
        str(descriptor.get("title", "")),
        project_governance_state,
        project_version=project_version,
    ).strip()
    if not title:
        raise ValueError("Managed doc descriptor field `title` is required.")
    doc_id = str(descriptor.get("doc_id", "")).strip()
    lines: list[str] = [f"# {title}"]
    if doc_type == "license":
        return lines
    if doc_id:
        lines.append(f"{DOC_ID_LABEL} {doc_id}")
    if doc_type:
        lines.append(f"{DOC_TYPE_LABEL} {doc_type}")
    if descriptor_bool(descriptor, "project_version"):
        lines.append(f"{PROJECT_VERSION_LABEL} {project_version}")
    if descriptor_optional_bool(descriptor, "project_governance_headers"):
        lines.extend(
            render_project_governance_header_lines(project_governance_state)
        )
    if descriptor_bool(descriptor, "last_updated"):
        lines.append(f"{LAST_UPDATED_LABEL} {utc_today()}")
    if descriptor_bool(descriptor, "devcovenant_version"):
        lines.append(f"{DEVCOV_VERSION_LABEL} {devcovenant_version}")
    return lines


def marker_line_regex(marker: str) -> re.Pattern[str]:
    """Return a line-anchored regex for marker lookup."""
    return re.compile(rf"(?m)^[ \t]*{re.escape(marker)}[ \t]*$")


def block_spans(
    text: str,
    begin_marker: str,
    end_marker: str,
) -> list[tuple[int, int, str]]:
    """Return positional spans for marker-delimited blocks in text."""
    spans: list[tuple[int, int, str]] = []
    begin_re = marker_line_regex(begin_marker)
    end_re = marker_line_regex(end_marker)
    search_start = 0
    while True:
        begin_match = begin_re.search(text, search_start)
        if begin_match is None:
            return spans
        end_match = end_re.search(text, begin_match.end())
        if end_match is None:
            return spans
        block_start = begin_match.start()
        end_marker_start = text.find(
            end_marker,
            end_match.start(),
            end_match.end(),
        )
        if end_marker_start < 0:
            return spans
        block_end = end_marker_start + len(end_marker)
        spans.append((block_start, block_end, text[block_start:block_end]))
        search_start = end_match.end()


def render_block(begin_marker: str, end_marker: str, body: str) -> str:
    """Render a managed marker block from marker pair and body."""
    return "\n".join([begin_marker, body.rstrip("\n"), end_marker])


def validate_preserve_markers(text: str, *, doc_name: str) -> None:
    """Validate DEVCOV preserve marker structure for one document text."""
    begin_re = marker_line_regex(USER_PRESERVE_BEGIN)
    end_re = marker_line_regex(USER_PRESERVE_END)
    events: list[tuple[int, str]] = []
    for match in begin_re.finditer(text):
        events.append((match.start(), "begin"))
    for match in end_re.finditer(text):
        events.append((match.start(), "end"))
    events.sort(key=lambda item: item[0])

    depth = 0
    for _, token in events:
        if token == "begin":
            if depth != 0:
                raise ValueError(
                    f"{doc_name} contains nested DEVCOV-USER-PRESERVE "
                    "blocks."
                )
            depth = 1
            continue
        if depth == 0:
            raise ValueError(
                f"{doc_name} contains DEVCOV-USER-PRESERVE end marker "
                "without begin marker."
            )
        depth = 0
    if depth != 0:
        raise ValueError(
            f"{doc_name} contains unterminated DEVCOV-USER-PRESERVE block."
        )


def preserve_block_spans(text: str) -> list[tuple[int, int, str]]:
    """Return positional spans for DEVCOV-USER-PRESERVE blocks."""
    return block_spans(text, USER_PRESERVE_BEGIN, USER_PRESERVE_END)


def preserve_blocks(text: str) -> list[str]:
    """Return preserve blocks in encounter order."""
    return [block for _, _, block in preserve_block_spans(text)]


def split_leading_preserve_blocks(text: str) -> tuple[list[str], str]:
    """Split contiguous top-of-document preserve blocks from text."""
    spans = preserve_block_spans(text)
    if not spans:
        return [], text
    leading: list[str] = []
    cursor = 0
    for start, end, block in spans:
        if text[cursor:start].strip():
            break
        leading.append(block)
        cursor = end
    if not leading:
        return [], text
    remainder = text[cursor:].lstrip("\n")
    return leading, remainder


def merge_preserve_blocks_into_replacement(
    current_block: str,
    replacement_block: str,
) -> str:
    """Merge preserve blocks from current block into replacement block."""
    current_preserves = preserve_blocks(current_block)
    if not current_preserves:
        return replacement_block

    missing = [
        block
        for block in current_preserves
        if block.strip() and block not in replacement_block
    ]
    if not missing:
        return replacement_block

    end_index = replacement_block.rfind(BLOCK_END)
    if end_index < 0:
        return replacement_block
    prefix = replacement_block[:end_index].rstrip("\n")
    suffix = replacement_block[end_index:].lstrip("\n")
    sections = [prefix]
    sections.extend(block.strip("\n") for block in missing)
    merged_prefix = "\n\n".join(
        section for section in sections if section.strip()
    )
    return f"{merged_prefix}\n{suffix}"


def merge_header_with_preserves(
    current_header: str,
    template_header: str,
) -> str:
    """Render generated header while preserving user preserve blocks."""
    leading_blocks, _ = split_leading_preserve_blocks(current_header)
    all_blocks = preserve_blocks(current_header)

    used = 0
    remaining_blocks: list[str] = []
    for block in all_blocks:
        if used < len(leading_blocks) and block == leading_blocks[used]:
            used += 1
            continue
        remaining_blocks.append(block)

    sections: list[str] = []
    sections.extend(block.strip("\n") for block in leading_blocks)
    sections.append(template_header.strip("\n"))
    sections.extend(block.strip("\n") for block in remaining_blocks)
    return "\n\n".join(section for section in sections if section).strip("\n")


def normalize_managed_block_body(body: str) -> str:
    """Strip begin/end markers from descriptor-managed block body text."""
    cleaned: list[str] = []
    for raw_line in body.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped in {BLOCK_BEGIN, BLOCK_END}:
            continue
        cleaned.append(line)
    return "\n".join(cleaned).strip("\n")


def compose_managed_block_body(descriptor: dict[str, object]) -> str:
    """Compose managed block body from descriptor-managed block text."""
    return normalize_managed_block_body(
        str(descriptor.get("managed_block", ""))
    )


def release_heading_for_render(
    project_governance_state: ProjectGovernanceState,
    project_version: str,
) -> str:
    """Return the rendered changelog release heading for one repo state."""
    if project_governance_state.is_unversioned:
        return project_governance_state.unreleased_heading
    return f"## Version {project_version}"


def render_descriptor_body(
    descriptor: dict[str, object],
    *,
    project_version: str,
    project_governance_state: ProjectGovernanceState,
) -> list[str]:
    """Render descriptor body lines with doc-specific substitutions."""
    body_value = descriptor.get("body")
    if not isinstance(body_value, str):
        return []
    rendered = project_governance_service.render_identity_placeholders(
        body_value,
        project_governance_state,
        project_version=project_version,
    ).replace(
        "{{ RELEASE_HEADING }}",
        release_heading_for_render(
            project_governance_state,
            project_version,
        ),
    )
    return [line.rstrip() for line in rendered.splitlines()]


def render_doc_from_descriptor(
    doc_name: str,
    descriptor: dict[str, object],
    *,
    project_version: str,
    devcovenant_version: str,
    project_governance_state: ProjectGovernanceState,
) -> str:
    """Render managed doc text from a validated YAML descriptor."""
    doc_type = str(descriptor.get("doc_type", "")).strip()
    header_lines = render_generated_header(
        doc_name,
        descriptor,
        project_version=project_version,
        devcovenant_version=devcovenant_version,
        project_governance_state=project_governance_state,
    )

    block_body = compose_managed_block_body(descriptor)
    managed_block = ""
    if doc_type != "license" and "managed_block" in descriptor:
        managed_block = render_block(BLOCK_BEGIN, BLOCK_END, block_body)

    body_lines = render_descriptor_body(
        descriptor,
        project_version=project_version,
        project_governance_state=project_governance_state,
    )

    workflow_body = str(descriptor.get("workflow_block", "")).rstrip("\n")
    workflow_block = ""
    if workflow_body:
        workflow_block = render_block(
            WORKFLOW_BEGIN,
            WORKFLOW_END,
            workflow_body,
        )

    project_governance_section = ""
    if doc_name == "AGENTS.md":
        project_governance_section = render_project_governance_section_block(
            project_version=project_version,
            project_governance_state=project_governance_state,
        )

    parts = []
    if header_lines:
        parts.append("\n".join(header_lines))
    if managed_block:
        parts.append(managed_block)
    if body_lines:
        parts.append("\n".join(body_lines))
    if workflow_block:
        parts.append(workflow_block)
    if project_governance_section:
        parts.append(project_governance_section)
    if doc_name == "AGENTS.md":
        parts.append(f"{POLICIES_BEGIN}\n{POLICIES_END}")
    if not parts:
        raise ValueError(
            f"Descriptor rendered no content for managed doc '{doc_name}'."
        )
    return "\n\n".join(parts).rstrip() + "\n"


def render_doc(
    repo_root: Path,
    doc_name: str,
    *,
    project_version: str,
    devcovenant_version: str,
    project_governance_state: ProjectGovernanceState,
    config_payload: dict[str, object] | None = None,
) -> str:
    """Render managed doc text from its YAML descriptor."""
    descriptor = load_managed_doc_descriptor(
        descriptor_path(
            repo_root,
            doc_name,
            config_payload=config_payload,
        ),
        doc_name=doc_name,
    )
    return render_doc_from_descriptor(
        doc_name,
        descriptor,
        project_version=project_version,
        devcovenant_version=devcovenant_version,
        project_governance_state=project_governance_state,
    )


def doc_is_placeholder(text: str) -> bool:
    """Return True for empty or effectively one-line docs."""
    lines = [line for line in text.splitlines() if line.strip()]
    return len(lines) <= 1


def doc_body_text(text: str) -> str:
    """Return current doc body text without generated headers/block."""
    body = strip_existing_generated_headers(text)
    spans = managed_block_spans(body)
    if spans:
        start, end, _ = spans[0]
        body = (body[:start] + body[end:]).strip("\n")
    return normalize_body_text(body)


def extract_managed_block(text: str) -> str | None:
    """Extract the first managed block from text."""
    spans = managed_block_spans(text)
    if not spans:
        return None
    return spans[0][2]


def rendered_header_and_block(rendered: str) -> tuple[str, str]:
    """Return rendered header text and the first managed block content."""
    managed_block = extract_managed_block(rendered)
    if managed_block is None:
        return generated_header_text(rendered), ""
    block_start = rendered.find(managed_block)
    if block_start < 0:
        return rendered.strip("\n"), managed_block
    header_text = rendered[:block_start].strip("\n")
    return header_text, managed_block


def generated_header_text(rendered: str) -> str:
    """Extract generated doc header lines from rendered markdown text."""
    lines = rendered.replace("\r\n", "\n").splitlines()
    if not lines:
        return ""

    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return ""

    header_lines: list[str] = []
    if lines[index].lstrip().startswith("#"):
        header_lines.append(lines[index].rstrip())
        index += 1

    header_prefixes = (
        "**doc id:**",
        "**doc type:**",
        "**project version:**",
        "**project stage:**",
        "**maintenance stance:**",
        "**compatibility policy:**",
        "**versioning mode:**",
        "**project codename:**",
        "**build identity:**",
        "**last updated:**",
        "**devcovenant version:**",
    )
    while index < len(lines):
        token = lines[index].strip()
        if not token:
            index += 1
            continue
        lowered = token.lower()
        if lowered.startswith(header_prefixes):
            header_lines.append(lines[index].rstrip())
            index += 1
            continue
        break

    return "\n".join(header_lines).strip("\n")


def generated_header_map(text: str) -> dict[str, str]:
    """Return normalized generated-header label/value pairs from one doc."""
    header_text = generated_header_text(text)
    if not header_text:
        return {}
    result: dict[str, str] = {}
    for line in header_text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("**") or ":**" not in stripped:
            continue
        label_part, value_part = stripped.split(":**", 1)
        label = label_part.strip("* ").strip().lower()
        result[label] = value_part.strip()
    return result


def normalize_generated_last_updated_for_compare(text: str) -> str:
    """Stabilize generated Last Updated values for no-op sync comparison."""
    header_text = generated_header_text(text)
    if not header_text:
        return text
    normalized_lines: list[str] = []
    found_last_updated = False
    for line in header_text.splitlines():
        if line.strip().startswith(LAST_UPDATED_LABEL):
            normalized_lines.append(f"{LAST_UPDATED_LABEL} <preserved>")
            found_last_updated = True
            continue
        normalized_lines.append(line.rstrip())
    if not found_last_updated:
        return text
    normalized_header = "\n".join(normalized_lines).strip("\n")
    return text.replace(header_text, normalized_header, 1)


def expected_managed_block(descriptor: dict[str, object]) -> str:
    """Build the managed block payload expected in rendered docs."""
    body = str(descriptor.get("managed_block", ""))
    lines: list[str] = []
    for raw_line in body.splitlines():
        stripped = raw_line.strip()
        if stripped in {BLOCK_BEGIN, BLOCK_END}:
            continue
        lines.append(raw_line.rstrip())
    return "\n".join(lines).strip("\n")


def descriptor_contains_generated_headers(
    descriptor: dict[str, object],
) -> bool:
    """Return True when managed_block duplicates generated headers."""
    body = str(descriptor.get("managed_block", ""))
    for line in body.splitlines():
        stripped = line.strip()
        for label in (
            DOC_ID_LABEL,
            DOC_TYPE_LABEL,
            PROJECT_VERSION_LABEL,
            PROJECT_STAGE_LABEL,
            MAINTENANCE_STANCE_LABEL,
            COMPATIBILITY_POLICY_LABEL,
            VERSIONING_MODE_LABEL,
            PROJECT_CODENAME_LABEL,
            BUILD_IDENTITY_LABEL,
            LAST_UPDATED_LABEL,
            DEVCOV_VERSION_LABEL,
        ):
            if stripped.startswith(label):
                return True
    return False


def is_importable_managed_doc(current: str, rendered: str) -> bool:
    """Return True when current doc is a valid import seed for refresh."""
    current_headers = generated_header_map(current)
    rendered_headers = generated_header_map(rendered)
    if not current_headers or not rendered_headers:
        return False

    for key in ("doc id", "doc type"):
        current_value = current_headers.get(key, "")
        rendered_value = rendered_headers.get(key, "")
        if not current_value or current_value != rendered_value:
            return False

    current_version = current_headers.get("devcovenant version", "")
    rendered_version = rendered_headers.get("devcovenant version", "")
    if not current_version or not rendered_version:
        return False
    try:
        return parse_devcovenant_version_for_compare(
            current_version
        ) >= parse_devcovenant_version_for_compare(rendered_version)
    except ValueError:
        return False


def is_devcovenant_shaped_target_doc(current: str, rendered: str) -> bool:
    """Return True when current doc already matches target doc identity."""
    current_headers = generated_header_map(current)
    rendered_headers = generated_header_map(rendered)
    if not current_headers or not rendered_headers:
        return False
    for key in ("doc id", "doc type"):
        current_value = current_headers.get(key, "")
        rendered_value = rendered_headers.get(key, "")
        if not current_value or current_value != rendered_value:
            return False
    return True


def merge_header_only_import_doc(
    current: str,
    rendered: str,
    *,
    doc_type: str = "",
) -> tuple[str, bool]:
    """Inject managed content while preserving imported body content."""
    header_text, managed_block = rendered_header_and_block(rendered)
    if not header_text:
        return rendered, rendered != current
    preserved = strip_existing_generated_headers(current).strip("\n")
    if doc_type == "license":
        preserved = strip_existing_license_legacy_prefix(preserved)
    parts = [header_text.strip("\n")]
    if managed_block:
        parts.append(managed_block.strip("\n"))
    if preserved:
        parts.append(preserved)
    updated = "\n\n".join(part for part in parts if part).rstrip() + "\n"
    return updated, updated != current


def first_block_text(
    text: str,
    begin_marker: str,
    end_marker: str,
) -> str | None:
    """Return first marker-delimited block text."""
    spans = block_spans(text, begin_marker, end_marker)
    if not spans:
        return None
    return spans[0][2]


def first_marker_start(text: str, marker: str, search_start: int) -> int:
    """Return marker start from offset, or -1 when missing."""
    match = marker_line_regex(marker).search(text, search_start)
    if match is None:
        return -1
    return match.start()


def managed_block_spans(text: str) -> list[tuple[int, int, str]]:
    """Return positional spans for every managed block in text."""
    return block_spans(text, BLOCK_BEGIN, BLOCK_END)


def replace_managed_block(current: str, template: str) -> tuple[str, bool]:
    """Replace managed blocks in current text with template block content."""
    current_blocks = managed_block_spans(current)
    template_blocks = managed_block_spans(template)
    if not current_blocks:
        return current, False

    template_header, _ = rendered_header_and_block(template)
    if not template_blocks:
        updated = current
        for start, end, current_block in reversed(current_blocks):
            preserved = "\n\n".join(
                block.strip("\n") for block in preserve_blocks(current_block)
            ).strip("\n")
            prefix = updated[:start].rstrip("\n")
            suffix = updated[end:].lstrip("\n")
            chunks: list[str] = []
            if prefix:
                chunks.append(prefix)
            if preserved:
                chunks.append(preserved)
            if suffix:
                chunks.append(suffix)
            updated = "\n\n".join(chunks)

        first_block_start = current_blocks[0][0]
        current_header = current[:first_block_start]
        merged_header = merge_header_with_preserves(
            current_header,
            template_header,
        )
        body = updated[first_block_start:].lstrip("\n")
        rebuilt_chunks: list[str] = [merged_header.rstrip("\n")]
        if body:
            rebuilt_chunks.append(body)
        rebuilt = "\n\n".join(chunk for chunk in rebuilt_chunks if chunk)
        rebuilt = rebuilt.rstrip()
        rebuilt = rebuilt + "\n" if rebuilt else ""
        return rebuilt, rebuilt != current

    replacement_count = min(len(current_blocks), len(template_blocks))
    updated = current
    changed = False
    for index in range(replacement_count - 1, -1, -1):
        start, end, current_block = current_blocks[index]
        replacement = merge_preserve_blocks_into_replacement(
            current_block,
            template_blocks[index][2],
        )
        if updated[start:end] == replacement:
            continue
        updated = updated[:start] + replacement + updated[end:]
        changed = True

    if template_header and current_blocks:
        first_block_start = current_blocks[0][0]
        current_header = updated[:first_block_start]
        merged_header = merge_header_with_preserves(
            current_header,
            template_header,
        )
        if merged_header != current_header.strip("\n"):
            updated = (
                merged_header.rstrip("\n")
                + "\n\n"
                + updated[first_block_start:].lstrip("\n")
            )
            changed = True
    return updated, changed


def strip_existing_generated_headers(current: str) -> str:
    """Strip leading generated header metadata from existing doc text."""
    lines = current.replace("\r\n", "\n").splitlines()
    if not lines:
        return ""

    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1

    if index < len(lines) and lines[index].lstrip().startswith("#"):
        index += 1

    while index < len(lines):
        token = lines[index].strip().lower()
        if not token:
            index += 1
            continue
        if token.startswith("**last updated:**"):
            index += 1
            continue
        if token.startswith("**project version:**"):
            index += 1
            continue
        if token.startswith("**project stage:**"):
            index += 1
            continue
        if token.startswith("**maintenance stance:**"):
            index += 1
            continue
        if token.startswith("**compatibility policy:**"):
            index += 1
            continue
        if token.startswith("**versioning mode:**"):
            index += 1
            continue
        if token.startswith("**project codename:**"):
            index += 1
            continue
        if token.startswith("**build identity:**"):
            index += 1
            continue
        if token.startswith("**devcovenant version:**"):
            index += 1
            continue
        if token.startswith("**doc id:**"):
            index += 1
            continue
        if token.startswith("**doc type:**"):
            index += 1
            continue
        break

    trimmed = "\n".join(lines[index:]).strip("\n")
    if trimmed:
        return trimmed
    return current.strip("\n")


def strip_existing_license_legacy_prefix(text: str) -> str:
    """Drop old non-markdown license headers that used metadata lines."""
    lines = str(text or "").replace("\r\n", "\n").splitlines()
    index = 0
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines):
        return ""
    if lines[index].lstrip().startswith("#"):
        return "\n".join(lines[index:]).strip("\n")

    probe = index + 1
    while probe < len(lines) and not lines[probe].strip():
        probe += 1
    if probe >= len(lines):
        return "\n".join(lines[index:]).strip("\n")

    legacy_prefixes = (
        "project version:",
        "project stage:",
        "maintenance stance:",
        "compatibility policy:",
        "versioning mode:",
        "last updated:",
        "devcovenant version:",
        "doc id:",
        "doc type:",
    )
    if not lines[probe].strip().lower().startswith(legacy_prefixes):
        return "\n".join(lines[index:]).strip("\n")

    index = probe
    while index < len(lines):
        token = lines[index].strip().lower()
        if not token:
            index += 1
            continue
        if token.startswith(legacy_prefixes):
            index += 1
            continue
        break
    return "\n".join(lines[index:]).strip("\n")


def inject_managed_header_and_block(
    current: str,
    rendered: str,
    *,
    doc_type: str = "",
) -> tuple[str, bool]:
    """Inject rendered header/managed block into unmanaged existing docs."""
    if is_devcovenant_shaped_target_doc(
        current,
        rendered,
    ) and not is_importable_managed_doc(current, rendered):
        return rendered, rendered != current

    header_text, managed_block = rendered_header_and_block(rendered)
    if not managed_block:
        return merge_header_only_import_doc(
            current,
            rendered,
            doc_type=doc_type,
        )

    preserved = strip_existing_generated_headers(current)
    leading_preserve_blocks, preserved_remainder = (
        split_leading_preserve_blocks(preserved)
    )
    sections: list[str] = [
        *(block.strip("\n") for block in leading_preserve_blocks),
        header_text,
        managed_block,
    ]
    if preserved_remainder:
        sections.append(preserved_remainder)
    updated = "\n\n".join(part for part in sections if part).rstrip() + "\n"
    return updated, updated != current


def next_control_block_start(text: str, search_start: int) -> int:
    """Return start of the next managed control block after one offset."""
    starts: list[int] = []

    managed_spans = managed_block_spans(text)
    if len(managed_spans) > 1:
        starts.append(managed_spans[1][0])

    workflow_spans = block_spans(text, WORKFLOW_BEGIN, WORKFLOW_END)
    if workflow_spans:
        starts.append(workflow_spans[0][0])

    policy_start = first_marker_start(text, POLICIES_BEGIN, search_start)
    if policy_start >= 0:
        starts.append(policy_start)

    if not starts:
        return len(text)
    return min(starts)


def merge_first_block_preserves(
    *,
    source_text: str,
    target_text: str,
    begin_marker: str,
    end_marker: str,
) -> str:
    """Merge preserve blocks from source first block into target block."""
    source_block = first_block_text(source_text, begin_marker, end_marker)
    if source_block is None:
        return target_text
    target_block = first_block_text(target_text, begin_marker, end_marker)
    if target_block is None:
        return target_text
    merged_block = merge_preserve_blocks_into_replacement(
        source_block,
        target_block,
    )
    if merged_block == target_block:
        return target_text
    block_start = target_text.find(target_block)
    if block_start < 0:
        return target_text
    block_end = block_start + len(target_block)
    return target_text[:block_start] + merged_block + target_text[block_end:]


def sync_agents_content(
    current: str,
    rendered: str,
) -> tuple[str, bool]:
    """Sync AGENTS blocks while preserving the editable section."""
    managed_spans = managed_block_spans(current)
    if not managed_spans:
        return rendered, current != rendered

    editable_start = managed_spans[0][1]
    editable_end = next_control_block_start(current, editable_start)
    editable_section = current[editable_start:editable_end]

    rendered_spans = managed_block_spans(rendered)
    if not rendered_spans:
        return rendered, current != rendered

    rendered_editable_start = rendered_spans[0][1]
    rendered_editable_end = next_control_block_start(
        rendered,
        rendered_editable_start,
    )
    updated = (
        rendered[:rendered_editable_start]
        + editable_section
        + rendered[rendered_editable_end:]
    )

    updated = merge_first_block_preserves(
        source_text=current,
        target_text=updated,
        begin_marker=BLOCK_BEGIN,
        end_marker=BLOCK_END,
    )
    updated = merge_first_block_preserves(
        source_text=current,
        target_text=updated,
        begin_marker=WORKFLOW_BEGIN,
        end_marker=WORKFLOW_END,
    )

    current_policy_block = first_block_text(
        current,
        POLICIES_BEGIN,
        POLICIES_END,
    )
    template_policy_block = first_block_text(
        updated,
        POLICIES_BEGIN,
        POLICIES_END,
    )
    if current_policy_block and template_policy_block:
        updated = updated.replace(
            template_policy_block, current_policy_block, 1
        )

    updated_spans = managed_block_spans(updated)
    if managed_spans and updated_spans:
        current_header = current[: managed_spans[0][0]]
        template_header = updated[: updated_spans[0][0]]
        merged_header = merge_header_with_preserves(
            current_header,
            template_header,
        )
        if merged_header != template_header.strip("\n"):
            updated = (
                merged_header.rstrip("\n")
                + "\n\n"
                + updated[updated_spans[0][0] :].lstrip("\n")
            )

    return updated, updated != current


def normalize_devcovenant_version_for_compare(raw: str) -> str:
    """Normalize DevCovenant version text into canonical PEP 440."""
    token = str(raw or "").strip()
    if not token:
        raise ValueError("DevCovenant version cannot be empty.")
    try:
        return str(Version(token))
    except InvalidVersion as exc:
        raise ValueError(
            f"Invalid DevCovenant version string `{raw}`."
        ) from exc


def parse_devcovenant_version_for_compare(raw: str) -> Version:
    """Parse one DevCovenant version for ordering checks."""
    return Version(normalize_devcovenant_version_for_compare(raw))


def detect_importable_managed_docs(
    repo_root: Path,
    source_dir: Path,
) -> list[str]:
    """Return existing repo docs eligible for first managed-doc adoption."""
    runtime_version = (
        (source_dir / "VERSION").read_text(encoding="utf-8").strip()
    )
    imported: list[str] = []
    for entry in managed_doc_descriptor_entries_from_roots(
        source_managed_doc_assets_roots(source_dir)
    ):
        descriptor = entry["descriptor"]
        doc_name = str(entry["doc"])
        if not descriptor_import_seed_enabled(descriptor):
            continue
        path = repo_root / doc_name
        if not path.exists() or not path.is_file():
            continue
        headers = generated_header_map(path.read_text(encoding="utf-8"))
        doc_id = str(descriptor.get("doc_id", "")).strip()
        doc_type = str(descriptor.get("doc_type", "")).strip()
        if headers.get("doc id", "") != doc_id:
            continue
        if headers.get("doc type", "") != doc_type:
            continue
        current_version = headers.get("devcovenant version", "")
        if not current_version:
            continue
        try:
            if parse_devcovenant_version_for_compare(
                current_version
            ) < parse_devcovenant_version_for_compare(runtime_version):
                continue
        except ValueError:
            continue
        imported.append(doc_name)
    return imported


def strip_preserve_blocks(text: str) -> str:
    """Remove user-preserve blocks from text before comparison."""
    cleaned: list[str] = []
    inside = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == USER_PRESERVE_BEGIN:
            inside = True
            continue
        if stripped == USER_PRESERVE_END:
            inside = False
            continue
        if not inside:
            cleaned.append(line)
    return "\n".join(cleaned)


def extract_doc_info(doc_path: Path) -> dict[str, object]:
    """Return generated header fields and managed block text."""
    text = doc_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    header_map: dict[str, str] = {}
    title = ""
    doc_id = ""
    doc_type = ""

    for line in lines:
        stripped = line.strip()
        if stripped == BLOCK_BEGIN:
            break
        if not stripped:
            continue
        if not title and stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
        if stripped.startswith(DOC_ID_LABEL):
            doc_id = stripped.split(DOC_ID_LABEL, 1)[1].strip()
            continue
        if stripped.startswith(DOC_TYPE_LABEL):
            doc_type = stripped.split(DOC_TYPE_LABEL, 1)[1].strip()
            continue
        for label in (
            PROJECT_VERSION_LABEL,
            PROJECT_STAGE_LABEL,
            MAINTENANCE_STANCE_LABEL,
            COMPATIBILITY_POLICY_LABEL,
            VERSIONING_MODE_LABEL,
            PROJECT_CODENAME_LABEL,
            BUILD_IDENTITY_LABEL,
            LAST_UPDATED_LABEL,
            DEVCOV_VERSION_LABEL,
        ):
            if stripped.startswith(label):
                header_map[label] = stripped.split(label, 1)[1].strip()

    managed_blocks: list[str] = []
    current_block_lines: list[str] = []
    inside = False
    has_managed_block = False
    for line in lines:
        if BLOCK_BEGIN in line:
            inside = True
            has_managed_block = True
            current_block_lines = []
            continue
        if BLOCK_END in line:
            if inside:
                managed_blocks.append(
                    strip_preserve_blocks(
                        "\n".join(current_block_lines)
                    ).strip("\n")
                )
            inside = False
            current_block_lines = []
            continue
        if inside:
            current_block_lines.append(line.rstrip())

    managed_block = managed_blocks[0] if managed_blocks else ""
    return {
        "title": title,
        "doc_id": doc_id,
        "doc_type": doc_type,
        "header_map": header_map,
        "managed_block": managed_block,
        "managed_blocks": managed_blocks,
        "has_managed_block": has_managed_block,
    }


def sync_doc(
    repo_root: Path,
    doc_name: str,
    *,
    project_version: str,
    devcovenant_version: str,
    project_governance_state: ProjectGovernanceState,
    import_managed_docs: set[str],
    config_payload: dict[str, object] | None = None,
) -> bool:
    """Synchronize one managed doc from descriptor content."""
    descriptor = load_managed_doc_descriptor(
        descriptor_path(
            repo_root,
            doc_name,
            config_payload=config_payload,
        ),
        doc_name=doc_name,
    )
    doc_type = str(descriptor.get("doc_type", "")).strip()
    rendered = render_doc_from_descriptor(
        doc_name,
        descriptor,
        project_version=project_version,
        devcovenant_version=devcovenant_version,
        project_governance_state=project_governance_state,
    )
    validate_preserve_markers(rendered, doc_name=doc_name)

    target = repo_root / doc_name
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        return True

    current = target.read_text(encoding="utf-8")
    validate_preserve_markers(current, doc_name=doc_name)
    if doc_is_placeholder(current):
        target.write_text(rendered, encoding="utf-8")
        return True

    importable_seed = (
        doc_name in import_managed_docs
        and is_importable_managed_doc(current, rendered)
    )
    if doc_name == "AGENTS.md":
        updated, changed = sync_agents_content(current, rendered)
    elif managed_block_spans(current):
        updated, changed = replace_managed_block(current, rendered)
    elif importable_seed:
        updated, changed = merge_header_only_import_doc(
            current,
            rendered,
            doc_type=doc_type,
        )
    else:
        updated, changed = inject_managed_header_and_block(
            current,
            rendered,
            doc_type=doc_type,
        )
    if not changed:
        return False
    if normalize_generated_last_updated_for_compare(
        updated
    ) == normalize_generated_last_updated_for_compare(current):
        return False

    target.write_text(updated, encoding="utf-8")
    return True
