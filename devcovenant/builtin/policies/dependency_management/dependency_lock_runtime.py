"""Policy-owned runtime for dependency lock and license refresh workflows."""

from __future__ import annotations

import concurrent.futures
import email
import hashlib
import importlib.metadata as importlib_metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess  # nosec B404
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement

from devcovenant.builtin.policies.dependency_management import (
    dependency_management,
)
from devcovenant.core.policy_metadata import (
    build_metadata_context,
    metadata_value_list,
    resolve_policy_metadata_bundle,
)
from devcovenant.core.policy_registry import (
    PolicyRegistry,
    load_policy_descriptor,
    resolve_script_location,
)
from devcovenant.core.tracked_registry import policy_registry_path

POLICY_ID = "dependency-management"
_TARGET_RESOLUTION_MAX_WORKERS = 4
_VULNERABILITY_AUDIT_CACHE_TTL = timedelta(hours=1)
_VULNERABILITY_AUDIT_MAX_WORKERS = 8
_VULNERABILITY_AUDIT_TIMEOUT_SECONDS = 10
_PYTHON_LOCK_OPTION_TOKENS = {
    "--cert",
    "--client-cert",
    "--constraint",
    "--extra-index-url",
    "--find-links",
    "--index-url",
    "--no-binary",
    "--no-index",
    "--only-binary",
    "--prefer-binary",
    "--pre",
    "--requirement",
    "--trusted-host",
    "-c",
    "-f",
    "-i",
    "-r",
}


def _normalize_repo_relative_path_token(raw_value: object) -> str:
    """Return one normalized repo-relative path token."""

    return str(raw_value or "").replace("\\", "/").strip()


@dataclass(frozen=True)
class LockFilePieces:
    """Describe the body of a generated requirements.lock snapshot."""

    body: List[str]


@dataclass(frozen=True)
class LockHandlerResult:
    """Outcome from one lockfile refresh strategy."""

    lock_file: str
    changed: bool
    attempted: bool
    message: str


@dataclass(frozen=True)
class SurfaceResolutionInputs:
    """Direct dependency lines plus inherited managed lock providers."""

    dependency_lines: List[str]
    inherited_lock_files: List[str]


@dataclass(frozen=True)
class DependencySurfaceVulnerability:
    """One vulnerability finding for a locked dependency surface."""

    package_name: str
    version: str
    vulnerability_id: str
    aliases: Tuple[str, ...]
    fix_versions: Tuple[str, ...]
    summary: str
    target_ids: Tuple[str, ...]


def _compute_file_hash(path: Path) -> str:
    """Return a stable hash digest for a file."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_hash_or_missing(path: Path) -> str:
    """Return digest string, with placeholder when file is absent."""

    if not path.exists():
        return "__missing__"
    return _compute_file_hash(path)


def _directory_hash_or_missing(path: Path) -> str:
    """Return one stable recursive digest string for a directory tree."""

    if not path.exists():
        return "__missing__"
    if not path.is_dir():
        return "__not_dir__"
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_compute_file_hash(child).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _dependency_refresh_engine_hash(repo_root: Path) -> str:
    """Return one stable digest for the active dependency-refresh engine."""

    digest = hashlib.sha256()
    active_policy_script = resolve_script_location(repo_root, POLICY_ID)
    repo_root = repo_root.resolve()
    paths: list[tuple[str, Path]] = [("runtime", Path(__file__).resolve())]
    if getattr(dependency_management, "__file__", None):
        paths.append(
            (
                "policy_module",
                Path(str(dependency_management.__file__)).resolve(),
            )
        )
    if active_policy_script is not None:
        paths.append(
            (
                "active_policy_script",
                active_policy_script.path.resolve(),
            )
        )
        paths.append(
            (
                "active_policy_descriptor",
                active_policy_script.path.with_suffix(".yaml").resolve(),
            )
        )
    for label, path in paths:
        component_token = _stable_engine_component_token(
            repo_root,
            path,
            label=label,
        )
        digest.update(component_token.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_file_hash_or_missing(path).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _stable_engine_component_token(
    repo_root: Path,
    path: Path,
    *,
    label: str,
) -> str:
    """Return one operator-stable engine component identity token."""

    resolved_path = path.resolve()
    repo_root = repo_root.resolve()
    path_parts = resolved_path.parts
    if "devcovenant" in path_parts:
        package_root_index = max(
            index
            for index, path_part in enumerate(path_parts)
            if path_part == "devcovenant"
        )
        package_relative = Path(*path_parts[package_root_index:]).as_posix()
        return f"module:{package_relative}"
    try:
        return "repo:" + resolved_path.relative_to(repo_root).as_posix()
    except ValueError:
        return f"component:{label}"


def _serialize_hash_targets(
    hash_targets: Sequence[dependency_management.DependencySurfaceTarget],
) -> list[dict[str, object]]:
    """Return one stable JSON-serializable view of surface hash targets."""

    serialized: list[dict[str, object]] = []
    for target in hash_targets:
        serialized.append(
            {
                "id": target.target_id,
                "marker": target.marker,
                "pip": dict(target.pip),
            }
        )
    return serialized


def _surface_input_fingerprint(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
) -> str:
    """Return one stable fingerprint for a surface's refresh inputs."""

    dependency_inputs = sorted(
        {
            _normalize_repo_relative_path_token(entry)
            for entry in [
                *surface.direct_dependency_files,
                *surface.dependency_files,
            ]
            if _normalize_repo_relative_path_token(entry)
        }
    )
    serialized = json.dumps(
        {
            "surface_id": surface.surface_id,
            "lock_file": _normalize_repo_relative_path_token(
                surface.lock_file
            ),
            "direct_dependency_files": list(surface.direct_dependency_files),
            "dependency_files": list(surface.dependency_files),
            "third_party_file": _normalize_repo_relative_path_token(
                surface.third_party_file
            ),
            "licenses_dir": _normalize_repo_relative_path_token(
                surface.licenses_dir
            ),
            "report_heading": surface.report_heading,
            "manage_licenses_readme": surface.manage_licenses_readme,
            "generate_hashes": surface.generate_hashes,
            "hash_targets": _serialize_hash_targets(surface.hash_targets),
            "dependency_inputs": {
                path_text: _file_hash_or_missing(repo_root / path_text)
                for path_text in dependency_inputs
            },
            "engine_hash": _dependency_refresh_engine_hash(repo_root),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _surface_output_fingerprints(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
) -> dict[str, str]:
    """Return one stable fingerprint map for a surface's managed outputs."""

    return {
        "lock_file": _file_hash_or_missing(
            repo_root / _normalize_repo_relative_path_token(surface.lock_file)
        ),
        "third_party_file": _file_hash_or_missing(
            repo_root
            / _normalize_repo_relative_path_token(surface.third_party_file)
        ),
        "licenses_dir": _directory_hash_or_missing(
            repo_root
            / _normalize_repo_relative_path_token(surface.licenses_dir)
        ),
    }


def _surface_runtime_state_is_current(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
    runtime_state: Mapping[str, object] | None,
) -> bool:
    """Return True when one stored runtime-state still matches disk."""

    if not isinstance(runtime_state, Mapping):
        return False
    stored_input_fingerprint = str(
        runtime_state.get("input_fingerprint", "")
    ).strip()
    if not stored_input_fingerprint:
        return False
    output_fingerprints = runtime_state.get("output_fingerprints", {})
    if not isinstance(output_fingerprints, Mapping):
        return False
    current_outputs = _surface_output_fingerprints(
        repo_root,
        surface=surface,
    )
    if any(
        current_outputs[key] in {"__missing__", "__not_dir__"}
        for key in ("lock_file", "third_party_file", "licenses_dir")
    ):
        return False
    return (
        stored_input_fingerprint
        == _surface_input_fingerprint(repo_root, surface=surface)
        and dict(output_fingerprints) == current_outputs
    )


def _build_surface_runtime_state(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
) -> dict[str, object]:
    """Build one runtime-state snapshot for a managed dependency surface."""

    return {
        "input_fingerprint": _surface_input_fingerprint(
            repo_root,
            surface=surface,
        ),
        "output_fingerprints": _surface_output_fingerprints(
            repo_root,
            surface=surface,
        ),
    }


def _order_surfaces_for_refresh(
    surfaces: Sequence[dependency_management.DependencySurface],
) -> list[dependency_management.DependencySurface]:
    """Refresh lock-provider surfaces before surfaces that consume them."""

    ordered = list(surfaces)
    if len(ordered) < 2:
        return ordered
    surface_by_id = {surface.surface_id: surface for surface in ordered}
    order_index = {
        surface.surface_id: index for index, surface in enumerate(ordered)
    }
    provider_by_lock: dict[str, str] = {}
    for surface in ordered:
        lock_token = _normalize_repo_relative_path_token(surface.lock_file)
        if lock_token and lock_token not in provider_by_lock:
            provider_by_lock[lock_token] = surface.surface_id
    dependencies: dict[str, set[str]] = {
        surface.surface_id: set() for surface in ordered
    }
    dependents: dict[str, set[str]] = {
        surface.surface_id: set() for surface in ordered
    }
    for surface in ordered:
        for dependency_file in surface.dependency_files:
            dependency_token = _normalize_repo_relative_path_token(
                dependency_file
            )
            provider_id = provider_by_lock.get(dependency_token)
            if not provider_id or provider_id == surface.surface_id:
                continue
            dependencies[surface.surface_id].add(provider_id)
            dependents[provider_id].add(surface.surface_id)
    ready = sorted(
        [
            surface_id
            for surface_id, providers in dependencies.items()
            if not providers
        ],
        key=order_index.__getitem__,
    )
    queued = set(ready)
    emitted: set[str] = set()
    result: list[dependency_management.DependencySurface] = []
    while ready:
        current_id = ready.pop(0)
        queued.discard(current_id)
        emitted.add(current_id)
        result.append(surface_by_id[current_id])
        for dependent_id in sorted(
            dependents[current_id],
            key=order_index.__getitem__,
        ):
            dependencies[dependent_id].discard(current_id)
            if (
                not dependencies[dependent_id]
                and dependent_id not in emitted
                and dependent_id not in queued
            ):
                ready.append(dependent_id)
                queued.add(dependent_id)
        ready.sort(key=order_index.__getitem__)
    if len(result) != len(ordered):
        return ordered
    return result


def _ensure_tool(command_name: str) -> bool:
    """Return True when a command is available on PATH."""

    return shutil.which(command_name) is not None


def _run_command(
    repo_root: Path,
    args: Sequence[str],
    *,
    extra_env: Dict[str, str] | None = None,
) -> None:
    """Run one lockfile command and raise on failure."""

    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    try:
        # Reviewed tokenized local command execution; shell use stays
        # forbidden.
        subprocess.run(  # nosec B603
            list(args),
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        rendered = " ".join(str(token) for token in args)
        output_parts: List[str] = []
        stdout = str(exc.stdout or "").strip()
        stderr = str(exc.stderr or "").strip()
        if stdout:
            output_parts.append("stdout:\n" + stdout)
        if stderr:
            output_parts.append("stderr:\n" + stderr)
        suffix = ""
        if output_parts:
            suffix = "\n\n" + "\n\n".join(output_parts[-2:])
        raise RuntimeError(
            "dependency-management command failed "
            f"({int(exc.returncode)}): {rendered}{suffix}"
        ) from exc


def _run_and_detect_change(
    repo_root: Path, lock_path: Path, command: Sequence[str]
) -> bool:
    """Run command and return True when target lockfile changed."""

    before = _file_hash_or_missing(lock_path)
    _run_command(repo_root, command)
    after = _file_hash_or_missing(lock_path)
    return before != after


def _normalise_header(lines: Sequence[str]) -> List[str]:
    """Stabilise pip-compile banners across Python versions."""

    return _normalise_header_for_mode(lines, generate_hashes=False)


def _normalise_header_for_mode(
    lines: Sequence[str],
    *,
    generate_hashes: bool,
    output_name: str = "requirements.lock",
    input_name: str = "requirements.in",
) -> List[str]:
    """Stabilise pip-compile banners across Python versions and modes."""

    python_banner = "# This file is autogenerated by pip-compile"
    command_tokens = ["pip-compile", "--allow-unsafe"]
    if generate_hashes:
        command_tokens.append("--generate-hashes")
    command_tokens.extend(
        [
            "--strip-extras",
            f"--output-file={output_name}",
            input_name,
        ]
    )
    target = "#    " + " ".join(command_tokens)
    result = list(lines)
    for index, line in enumerate(result):
        if line.startswith("#    pip-compile "):
            result[index] = target
            break
    for index, line in enumerate(result):
        if line.startswith(
            "# This file is autogenerated by pip-compile with Python"
        ):
            result[index] = python_banner
            break
    return result


def _split_last_updated(lines: Iterable[str]) -> LockFilePieces:
    """Drop optional Last Updated banner from lockfile body."""

    collected = list(lines)
    if collected and collected[0].startswith("# Last Updated:"):
        return LockFilePieces(collected[1:])
    return LockFilePieces(collected)


def _is_python_lock_option_line(raw_line: str) -> bool:
    """Return True for environment-specific pip option directives."""

    stripped = str(raw_line).strip()
    if not stripped or raw_line[:1].isspace():
        return False
    token = stripped.split(maxsplit=1)[0].split("=", 1)[0]
    return token in _PYTHON_LOCK_OPTION_TOKENS


def _strip_python_lock_option_lines(lines: Sequence[str]) -> List[str]:
    """Remove non-semantic pip option lines from lockfile content."""

    return [
        str(raw_line)
        for raw_line in lines
        if not _is_python_lock_option_line(str(raw_line))
    ]


def _normalise_input_reference_comments(
    lines: Sequence[str],
    *,
    input_name: str,
) -> List[str]:
    """Rewrite `# via -r ...` comments to the configured input label."""

    normalized: List[str] = []
    for raw_line in lines:
        raw_text = str(raw_line)
        match = re.match(
            r"^(?P<indent>\s*#\s*)(?:(?P<via>via)\s+)?-r\s+.+$",
            raw_text,
        )
        if match is not None:
            prefix = match.group("indent")
            via = "via " if match.group("via") else ""
            normalized.append(f"{prefix}{via}-r {input_name}")
            continue
        normalized.append(raw_text)
    return normalized


def _normalize_python_lock_semantics(lines: Sequence[str]) -> List[str]:
    """Compare Python lockfiles by resolved pins, not banner formatting."""

    return _normalize_python_lock_semantics_for_mode(
        lines,
        generate_hashes=False,
    )


def _normalize_python_lock_semantics_for_mode(
    lines: Sequence[str],
    *,
    generate_hashes: bool,
) -> List[str]:
    """Compare Python lockfiles by semantic content for one lock mode."""

    normalized: List[str] = []
    for raw_line in _strip_python_lock_option_lines(lines):
        stripped = str(raw_line).strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--hash=sha256:"):
            if generate_hashes:
                normalized.append(stripped.rstrip("\\").strip())
            continue
        if raw_line[:1].isspace():
            continue
        normalized.append(stripped.rstrip("\\").strip())
    return normalized


def _compile_requirements_lock(
    repo_root: Path,
    requirements_in: Path,
    *,
    generate_hashes: bool = False,
    existing_lock_lines: Sequence[str] | None = None,
    output_name: str = "requirements.lock",
    input_name: str = "requirements.in",
) -> LockFilePieces:
    """Generate normalised requirements.lock content without touching disk."""

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_lock = Path(tmpdir) / output_name
        _run_pip_compile(
            repo_root,
            requirements_in,
            tmp_lock,
            generate_hashes=generate_hashes,
        )
        normalised = _normalise_header_for_mode(
            tmp_lock.read_text().splitlines(),
            generate_hashes=generate_hashes,
            output_name=output_name,
            input_name=input_name,
        )
        cleaned = _normalise_input_reference_comments(
            _strip_python_lock_option_lines(normalised),
            input_name=input_name,
        )
    if generate_hashes:
        return _split_last_updated(cleaned)
    with_direct_conditionals = _preserve_direct_conditional_requirements(
        cleaned,
        requirements_in,
        source_display_name=input_name,
    )
    return _split_last_updated(with_direct_conditionals)


def _preserve_direct_conditional_requirements(
    compiled_lines: Sequence[str],
    requirements_in: Path,
    *,
    source_display_name: str = "requirements.in",
) -> List[str]:
    """Keep direct exact conditional requirements visible in the lock."""

    preserved_requirements = _collect_direct_conditional_requirements(
        requirements_in
    )
    if not preserved_requirements:
        return list(compiled_lines)

    existing_entries = {
        str(raw_line).strip().rstrip("\\").strip()
        for raw_line in compiled_lines
        if str(raw_line).strip()
        and not str(raw_line).lstrip().startswith("#")
        and not str(raw_line)[:1].isspace()
    }
    result = list(compiled_lines)
    for requirement in preserved_requirements:
        pin_line = _format_direct_conditional_requirement(requirement)
        if pin_line in existing_entries:
            continue
        if result and result[-1] != "":
            result.append("")
        result.append(pin_line)
        result.append(f"    # via -r {source_display_name}")
    return result


def _collect_direct_conditional_requirements(
    requirements_in: Path,
) -> List[Requirement]:
    """Return direct exact conditional requirements from requirements.in."""

    collected: List[Requirement] = []
    for raw_line in requirements_in.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _is_python_lock_option_line(raw_line):
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement:
            continue
        if requirement.marker is None:
            continue
        specifiers = list(requirement.specifier)
        if len(specifiers) != 1 or specifiers[0].operator != "==":
            continue
        collected.append(requirement)
    return collected


def _format_direct_conditional_requirement(requirement: Requirement) -> str:
    """Return canonical text for one direct exact conditional requirement."""

    specifiers = list(requirement.specifier)
    exact = specifiers[0]
    extras = (
        f"[{','.join(sorted(requirement.extras))}]"
        if requirement.extras
        else ""
    )
    return (
        f"{requirement.name}{extras}=={exact.version} ; "
        f"{requirement.marker}"
    )


def _build_hashed_requirement_block(
    requirement_line: str,
    hashes: Sequence[str],
    *,
    source_display_name: str = "requirements.in",
) -> List[str]:
    """Return one pip-compile-style hashed requirement block."""

    unique_hashes = sorted(
        {str(entry).strip() for entry in hashes if str(entry).strip()}
    )
    if not unique_hashes:
        raise RuntimeError(
            f"Missing hashes for locked requirement `{requirement_line}`."
        )
    block = [f"{requirement_line} \\"]
    last_index = len(unique_hashes) - 1
    for index, digest in enumerate(unique_hashes):
        suffix = " \\" if index < last_index else ""
        block.append(f"    --hash=sha256:{digest}{suffix}")
    block.append(f"    # via -r {source_display_name}")
    return block


def _build_plain_requirement_block(
    requirement_line: str,
    *,
    source_display_name: str = "requirements.in",
) -> List[str]:
    """Return one deterministic non-hash requirement block."""

    return [
        requirement_line,
        f"    # via -r {source_display_name}",
    ]


def _run_pip_compile(
    repo_root: Path,
    requirements_in: Path,
    output_path: Path,
    *,
    generate_hashes: bool = False,
) -> None:
    """Run pip-compile with the repository's canonical options."""

    if importlib.util.find_spec("piptools") is None:
        raise RuntimeError(
            "pip-tools is required for requirements.lock updates."
        )
    with tempfile.TemporaryDirectory() as cache_dir:
        _run_command(
            repo_root,
            (
                sys.executable,
                "-m",
                "piptools",
                "compile",
                "--quiet",
                "--allow-unsafe",
                *(("--generate-hashes",) if generate_hashes else ()),
                "--strip-extras",
                "--output-file",
                str(output_path),
                str(requirements_in),
            ),
            extra_env={"PIP_TOOLS_CACHE_DIR": cache_dir},
        )


def _refresh_python_surface_lock(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
    all_surfaces: Sequence[dependency_management.DependencySurface] | None = (
        None
    ),
) -> LockHandlerResult:
    """Refresh one declared Python dependency surface."""

    normalized_lock = _normalize_repo_relative_path_token(surface.lock_file)
    lock_path = repo_root / normalized_lock
    if not surface.direct_dependency_files:
        return LockHandlerResult(
            normalized_lock,
            changed=False,
            attempted=False,
            message="Skipped: no direct dependency files are declared.",
        )
    input_display_name = (
        _normalize_repo_relative_path_token(surface.direct_dependency_files[0])
        if len(surface.direct_dependency_files) == 1
        else "configured dependency inputs"
    )
    previous = (
        _split_last_updated(lock_path.read_text().splitlines())
        if lock_path.exists()
        else LockFilePieces([])
    )
    inherited_provider_locks = [
        _normalize_repo_relative_path_token(other.lock_file)
        for other in (all_surfaces or [])
        if other.surface_id != surface.surface_id
        and _normalize_repo_relative_path_token(other.lock_file)
    ]
    if surface.hash_targets:
        compiled = _compile_target_surface_lock(
            repo_root,
            surface_id=surface.surface_id,
            dependency_files=surface.direct_dependency_files,
            provider_lock_files=inherited_provider_locks,
            hash_targets=surface.hash_targets,
            source_display_name=input_display_name,
            generate_hashes=surface.generate_hashes,
        )
    elif surface.generate_hashes:
        raise RuntimeError(
            "Hash-locked dependency surface "
            f"`{surface.surface_id}` requires configured hash_targets."
        )
    elif len(surface.direct_dependency_files) == 1 and (
        Path(surface.direct_dependency_files[0]).name == "requirements.in"
    ):
        compiled = _compile_requirements_lock(
            repo_root,
            repo_root / surface.direct_dependency_files[0],
            generate_hashes=False,
            existing_lock_lines=previous.body,
            output_name=Path(normalized_lock).name,
            input_name=input_display_name,
        )
    else:
        compiled = _compile_lock_from_dependency_files(
            repo_root,
            dependency_files=surface.direct_dependency_files,
            output_name=Path(normalized_lock).name,
            generate_hashes=False,
            existing_lock_lines=previous.body,
            input_display_name=input_display_name,
        )
    compiled_cleaned = LockFilePieces(
        _strip_python_lock_option_lines(compiled.body)
    )
    previous_cleaned = LockFilePieces(
        _strip_python_lock_option_lines(previous.body)
    )
    if _normalize_python_lock_semantics_for_mode(
        previous_cleaned.body,
        generate_hashes=surface.generate_hashes,
    ) == _normalize_python_lock_semantics_for_mode(
        compiled_cleaned.body,
        generate_hashes=surface.generate_hashes,
    ):
        if previous_cleaned.body != previous.body:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                "\n".join(compiled_cleaned.body) + "\n",
                encoding="utf-8",
            )
            return LockHandlerResult(
                normalized_lock,
                changed=True,
                attempted=True,
                message=(
                    f"Normalized {normalized_lock} by removing "
                    "environment-specific pip option lines."
                ),
            )
        if previous.body != compiled_cleaned.body:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path.write_text(
                "\n".join(compiled_cleaned.body) + "\n",
                encoding="utf-8",
            )
            return LockHandlerResult(
                normalized_lock,
                changed=True,
                attempted=True,
                message=(
                    f"Normalized {normalized_lock} without changing "
                    "resolved pins."
                ),
            )
        return LockHandlerResult(
            normalized_lock,
            changed=False,
            attempted=True,
            message="No content change after pip-compile.",
        )

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("\n".join(compiled.body) + "\n", encoding="utf-8")
    return LockHandlerResult(
        normalized_lock,
        changed=True,
        attempted=True,
        message=f"Updated {normalized_lock}.",
    )


def _refresh_python_requirements_lock(
    repo_root: Path,
    *,
    generate_hashes: bool = False,
) -> LockHandlerResult:
    """Refresh requirements.lock from requirements.in."""
    return _refresh_python_surface_lock(
        repo_root,
        surface=dependency_management.DependencySurface(
            surface_id="requirements-lock",
            enabled=True,
            active=True,
            lock_file="requirements.lock",
            direct_dependency_files=["requirements.in"],
            dependency_files=["requirements.in"],
            dependency_globs=[],
            dependency_dirs=[],
            third_party_file="licenses/THIRD_PARTY_LICENSES.md",
            licenses_dir="licenses",
            report_heading=dependency_management.DEFAULT_REPORT_HEADING,
            manage_licenses_readme=True,
            generate_hashes=generate_hashes,
            required_paths=[],
            hash_targets=[],
            audit_service="",
            audit_ignore_ids=[],
        ),
    )


def _compile_lock_from_dependency_files(
    repo_root: Path,
    *,
    dependency_files: Sequence[str],
    output_name: str,
    generate_hashes: bool,
    existing_lock_lines: Sequence[str] | None = None,
    input_display_name: str,
) -> LockFilePieces:
    """Compile one lock from dependency strings declared in manifest files."""

    dependency_lines: List[str] = []
    for raw_path in dependency_files:
        path_token = _normalize_repo_relative_path_token(raw_path)
        if not path_token:
            continue
        manifest_path = repo_root / path_token
        dependency_lines.extend(
            dependency_management._direct_dependency_strings_from_file(
                manifest_path
            )
        )
    if not dependency_lines:
        return LockFilePieces([])

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "runtime-requirements.in"
        tmp_in.write_text(
            "\n".join(dependency_lines).rstrip() + "\n",
            encoding="utf-8",
        )
        return _compile_requirements_lock(
            repo_root,
            tmp_in,
            generate_hashes=generate_hashes,
            existing_lock_lines=existing_lock_lines,
            output_name=output_name,
            input_name=input_display_name,
        )


def _surface_dependency_strings(
    repo_root: Path,
    *,
    dependency_files: Sequence[str],
) -> List[str]:
    """Collect dependency strings from one surface's direct inputs."""

    return _surface_resolution_inputs(
        repo_root,
        dependency_files=dependency_files,
    ).dependency_lines


def _surface_resolution_inputs(
    repo_root: Path,
    *,
    dependency_files: Sequence[str],
    provider_lock_files: Sequence[str] = (),
) -> SurfaceResolutionInputs:
    """Collect surface inputs while preserving inherited lock providers."""

    dependency_lines: List[str] = []
    inherited_lock_files: List[str] = []
    seen_paths: set[Path] = set()
    provider_lock_tokens = {
        _normalize_repo_relative_path_token(entry)
        for entry in provider_lock_files
        if _normalize_repo_relative_path_token(entry)
    }
    for raw_path in dependency_files:
        path_token = _normalize_repo_relative_path_token(raw_path)
        if not path_token:
            continue
        manifest_path = repo_root / path_token
        _collect_surface_resolution_inputs_from_manifest(
            repo_root,
            manifest_path,
            seen_paths=seen_paths,
            provider_lock_tokens=provider_lock_tokens,
            dependency_lines=dependency_lines,
            inherited_lock_files=inherited_lock_files,
        )
    return SurfaceResolutionInputs(
        dependency_lines=_normalize_python_lock_semantics_for_mode(
            dependency_lines,
            generate_hashes=False,
        ),
        inherited_lock_files=list(dict.fromkeys(inherited_lock_files)),
    )


def _collect_dependency_strings_from_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    seen_paths: set[Path],
) -> List[str]:
    """Collect one manifest's direct dependency strings with `-r` expansion."""

    resolved_path = manifest_path.resolve()
    if resolved_path in seen_paths:
        return []
    seen_paths.add(resolved_path)
    if not manifest_path.exists():
        raise RuntimeError(
            "dependency-management input is missing: "
            f"{manifest_path.relative_to(repo_root)}"
        )
    collected: List[str] = []
    entries = dependency_management._direct_dependency_strings_from_file(
        manifest_path
    )
    for entry in entries:
        include_target = _extract_requirements_include_target(str(entry))
        if include_target is None:
            collected.append(str(entry).strip())
            continue
        include_path = (manifest_path.parent / include_target).resolve()
        collected.extend(
            _collect_dependency_strings_from_manifest(
                repo_root,
                include_path,
                seen_paths=seen_paths,
            )
        )
    return collected


def _collect_surface_resolution_inputs_from_manifest(
    repo_root: Path,
    manifest_path: Path,
    *,
    seen_paths: set[Path],
    provider_lock_tokens: set[str],
    dependency_lines: List[str],
    inherited_lock_files: List[str],
) -> None:
    """Collect direct dependency lines while preserving owned lock inputs."""

    manifest_token = _repo_relative_path_token(repo_root, manifest_path)
    if manifest_token and manifest_token in provider_lock_tokens:
        inherited_lock_files.append(manifest_token)
        return
    resolved_path = manifest_path.resolve()
    if resolved_path in seen_paths:
        return
    seen_paths.add(resolved_path)
    if not manifest_path.exists():
        raise RuntimeError(
            "dependency-management input is missing: "
            f"{manifest_path.relative_to(repo_root)}"
        )
    entries = dependency_management._direct_dependency_strings_from_file(
        manifest_path
    )
    for entry in entries:
        include_target = _extract_requirements_include_target(str(entry))
        if include_target is None:
            dependency_lines.append(str(entry).strip())
            continue
        include_path = (manifest_path.parent / include_target).resolve()
        include_token = _repo_relative_path_token(repo_root, include_path)
        if include_token and include_token in provider_lock_tokens:
            inherited_lock_files.append(include_token)
            continue
        _collect_surface_resolution_inputs_from_manifest(
            repo_root,
            include_path,
            seen_paths=seen_paths,
            provider_lock_tokens=provider_lock_tokens,
            dependency_lines=dependency_lines,
            inherited_lock_files=inherited_lock_files,
        )


def _repo_relative_path_token(repo_root: Path, path: Path) -> str:
    """Return one normalized repo-relative token when a path stays in-repo."""

    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return ""


def _extract_requirements_include_target(raw_line: str) -> str | None:
    """Return the referenced path for supported requirements includes."""

    stripped = str(raw_line).strip()
    if not stripped:
        return None
    for prefix in ("-r", "--requirement"):
        if stripped == prefix:
            return None
        if stripped.startswith(prefix + "="):
            return stripped.split("=", 1)[1].strip()
        if stripped.startswith(prefix + " "):
            return stripped.split(None, 1)[1].strip()
        if prefix == "-r" and stripped.startswith("-r") and len(stripped) > 2:
            return stripped[2:].strip()
    return None


def _canonical_target_python_version(
    target: dependency_management.DependencySurfaceTarget,
) -> str:
    """Return one stable dotted Python version string for a target."""

    raw_value = (
        target.pip.get("python-version")
        or target.pip.get("python_version")
        or ""
    )
    text = str(raw_value).strip()
    abi = str(target.pip.get("abi", "")).strip().lower()
    abi_match = re.fullmatch(r"cp(?P<major>\d)(?P<minor>\d{2})", abi)
    if abi_match is None:
        return text
    abi_version = f"{abi_match.group('major')}.{abi_match.group('minor')}"
    if not text:
        return abi_version
    if text != abi_version and text.startswith(f"{abi_match.group('major')}."):
        return abi_version
    return text


def _canonicalize_requirement_string(requirement: Requirement) -> str:
    """Return one marker-free requirement string for resolver input."""

    extras = (
        f"[{','.join(sorted(requirement.extras))}]"
        if requirement.extras
        else ""
    )
    if requirement.url:
        return f"{requirement.name}{extras} @ {requirement.url}"
    specifier = str(requirement.specifier)
    return f"{requirement.name}{extras}{specifier}"


def _target_marker_environment(
    target: dependency_management.DependencySurfaceTarget,
) -> Dict[str, str]:
    """Build one PEP 508 marker environment for a configured target."""

    environment = default_environment()
    normalized_pip = {
        str(key).strip().replace("_", "-"): str(value).strip()
        for key, value in target.pip.items()
        if str(key).strip() and str(value).strip()
    }
    python_version = _canonical_target_python_version(
        target
    ) or environment.get(
        "python_version",
        "3.11",
    )
    environment["python_version"] = python_version
    environment["python_full_version"] = (
        python_version
        if python_version.count(".") >= 2
        else f"{python_version}.0"
    )
    implementation = normalized_pip.get("implementation", "").lower()
    if implementation == "cp":
        environment["implementation_name"] = "cpython"
        environment["platform_python_implementation"] = "CPython"
    elif implementation:
        environment["implementation_name"] = implementation
        environment["platform_python_implementation"] = implementation.upper()
    platform_token = normalized_pip.get("platform", "")
    if platform_token.startswith("manylinux") or "linux" in platform_token:
        environment["sys_platform"] = "linux"
        environment["os_name"] = "posix"
        environment["platform_system"] = "Linux"
        environment["platform_machine"] = platform_token.rsplit("_", 1)[-1]
    elif platform_token.startswith("win"):
        environment["sys_platform"] = "win32"
        environment["os_name"] = "nt"
        environment["platform_system"] = "Windows"
        machine = platform_token.split("_", 1)[-1]
        environment["platform_machine"] = (
            "AMD64" if machine.lower() == "amd64" else machine
        )
    elif platform_token.startswith("macosx"):
        environment["sys_platform"] = "darwin"
        environment["os_name"] = "posix"
        environment["platform_system"] = "Darwin"
        environment["platform_machine"] = platform_token.rsplit("_", 1)[-1]
    environment["extra"] = ""
    return {str(key): str(value) for key, value in environment.items()}


def _requirement_is_active_for_target(
    requirement: Requirement,
    *,
    target_environment: Mapping[str, str],
    selected_extras: Iterable[str] = (),
) -> bool:
    """Return True when one requirement applies to the configured target."""

    if requirement.marker is None:
        return True
    environment = dict(target_environment)
    environment["extra"] = ""
    if requirement.marker.evaluate(environment):
        return True
    for extra in selected_extras:
        environment = dict(target_environment)
        environment["extra"] = str(extra)
        if requirement.marker.evaluate(environment):
            return True
    return False


def _iter_active_target_requirements(
    requirement_lines: Sequence[str],
    *,
    target_environment: Mapping[str, str],
    selected_extras: Iterable[str] = (),
) -> List[Requirement]:
    """Return active requirements after evaluating target markers."""

    active: List[Requirement] = []
    for raw_line in requirement_lines:
        stripped = str(raw_line).strip()
        if not stripped or stripped.startswith("#"):
            continue
        if _is_python_lock_option_line(stripped):
            continue
        try:
            requirement = Requirement(stripped)
        except InvalidRequirement:
            continue
        if _requirement_is_active_for_target(
            requirement,
            target_environment=target_environment,
            selected_extras=selected_extras,
        ):
            active.append(requirement)
    return active


def _record_selected_extras(
    selected_extras: Dict[str, set[str]],
    requirement: Requirement,
) -> bool:
    """Merge selected extras for one requirement and report growth."""

    normalized_name = dependency_management._normalize_distribution_name(
        requirement.name
    )
    if not requirement.extras:
        selected_extras.setdefault(normalized_name, set())
        return False
    current = selected_extras.setdefault(normalized_name, set())
    before = set(current)
    current.update(str(extra) for extra in requirement.extras)
    return current != before


def _resolved_requirement_satisfies(
    requirement: Requirement,
    resolved_entries: Mapping[str, Mapping[str, object]],
) -> bool:
    """Return True when one resolved entry satisfies a requirement."""

    normalized_name = dependency_management._normalize_distribution_name(
        requirement.name
    )
    entry = resolved_entries.get(normalized_name)
    if entry is None:
        return False
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    version = str(metadata.get("version", "")).strip()
    if not version:
        return False
    if not requirement.specifier:
        return True
    return requirement.specifier.contains(version, prereleases=True)


def _merge_target_report_entries(
    resolved_entries: Dict[str, Dict[str, object]],
    installs: Sequence[Mapping[str, object]],
) -> List[str]:
    """Merge one pip report payload into the resolved target closure."""

    added_names: List[str] = []
    for item in installs:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        name = str(metadata.get("name", "")).strip()
        version = str(metadata.get("version", "")).strip()
        if not name or not version:
            continue
        normalized_name = dependency_management._normalize_distribution_name(
            name
        )
        existing = resolved_entries.get(normalized_name)
        if existing is None:
            resolved_entries[normalized_name] = dict(item)
            added_names.append(normalized_name)
            continue
        existing_metadata = existing.get("metadata", {})
        existing_version = (
            str(existing_metadata.get("version", "")).strip()
            if isinstance(existing_metadata, Mapping)
            else ""
        )
        if existing_version and existing_version != version:
            raise RuntimeError(
                "Target dependency closure resolved conflicting versions for "
                f"`{name}`: `{existing_version}` vs `{version}`."
            )
    return added_names


def _target_report_command(
    target: dependency_management.DependencySurfaceTarget,
    *,
    report_path: Path,
    requirements_path: Path,
) -> List[str]:
    """Return one pip dry-run report command for a target."""

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--report",
        str(report_path),
        "--only-binary=:all:",
    ]
    for key, value in target.pip.items():
        normalized_key = str(key).strip().replace("_", "-")
        normalized_value = (
            _canonical_target_python_version(target)
            if normalized_key == "python-version"
            else str(value).strip()
        )
        if not normalized_key or not normalized_value:
            continue
        command.extend([f"--{normalized_key}", normalized_value])
    command.extend(["-r", str(requirements_path)])
    return command


def _run_pip_hash_target_report(
    repo_root: Path,
    *,
    dependency_lines: Sequence[str],
    target: dependency_management.DependencySurfaceTarget,
) -> List[Dict[str, object]]:
    """Resolve one full dependency closure for a configured hash target."""

    if not dependency_lines:
        return []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_in = Path(tmpdir) / "hash-target.in"
        tmp_report = Path(tmpdir) / "hash-target-report.json"
        tmp_in.write_text(
            "\n".join(dependency_lines).rstrip() + "\n",
            encoding="utf-8",
        )
        command = _target_report_command(
            target,
            report_path=tmp_report,
            requirements_path=tmp_in,
        )
        env = dict(os.environ)
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        # Reviewed tokenized local command execution.
        # Shell use stays forbidden.
        subprocess.run(  # nosec B603
            command,
            cwd=repo_root,
            check=True,
            text=True,
            capture_output=True,
            env=env,
        )
        payload = json.loads(tmp_report.read_text(encoding="utf-8"))
    installs = payload.get("install", [])
    if not isinstance(installs, list):
        raise RuntimeError(
            "pip hash-target report did not contain an install list."
        )
    return installs


def _read_distribution_requirements_from_wheel(
    wheel_path: Path,
) -> List[str]:
    """Return `Requires-Dist` lines from one wheel's METADATA payload."""

    with zipfile.ZipFile(wheel_path) as archive:
        metadata_members = [
            member
            for member in archive.namelist()
            if member.endswith(".dist-info/METADATA")
        ]
        if not metadata_members:
            raise RuntimeError(
                "Downloaded wheel did not contain a dist-info METADATA file: "
                f"{wheel_path.name}"
            )
        payload = archive.read(metadata_members[0]).decode(
            "utf-8",
            errors="replace",
        )
    message = email.message_from_string(payload)
    values = message.get_all("Requires-Dist")
    if values is None:
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def _load_target_distribution_requirements(
    repo_root: Path,
    *,
    target: dependency_management.DependencySurfaceTarget,
    distribution_name: str,
    version: str,
    metadata_cache: Dict[Tuple[str, str, str], List[str]],
) -> List[str]:
    """Download one target wheel and return its active requirement lines."""

    normalized_name = dependency_management._normalize_distribution_name(
        distribution_name
    )
    cache_key = (target.target_id, normalized_name, str(version).strip())
    cached = metadata_cache.get(cache_key)
    if cached is not None:
        return list(cached)
    try:
        installed = importlib_metadata.distribution(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        installed = None
    if (
        installed is not None
        and str(installed.version).strip() == str(version).strip()
    ):
        requirements = [
            str(value).strip()
            for value in (installed.requires or [])
            if str(value).strip()
        ]
        metadata_cache[cache_key] = list(requirements)
        return list(requirements)
    with tempfile.TemporaryDirectory() as tmpdir:
        download_dir = Path(tmpdir)
        command = [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--no-deps",
            "--only-binary=:all:",
            "--dest",
            str(download_dir),
        ]
        for key, value in target.pip.items():
            normalized_key = str(key).strip().replace("_", "-")
            normalized_value = (
                _canonical_target_python_version(target)
                if normalized_key == "python-version"
                else str(value).strip()
            )
            if not normalized_key or not normalized_value:
                continue
            command.extend([f"--{normalized_key}", normalized_value])
        command.append(f"{distribution_name}=={version}")
        _run_command(repo_root, command)
        files = [path for path in download_dir.iterdir() if path.is_file()]
        if len(files) != 1:
            raise RuntimeError(
                "Expected exactly one downloaded wheel for target metadata "
                f"inspection of `{distribution_name}=={version}`."
            )
        wheel_path = files[0]
        if wheel_path.suffix != ".whl":
            raise RuntimeError(
                "Target-aware dependency closure requires wheel metadata for "
                f"`{distribution_name}=={version}`, but pip downloaded "
                f"`{wheel_path.name}`."
            )
        requirements = _read_distribution_requirements_from_wheel(wheel_path)
    metadata_cache[cache_key] = list(requirements)
    return list(requirements)


def _resolve_complete_target_report(
    repo_root: Path,
    *,
    dependency_lines: Sequence[str],
    target: dependency_management.DependencySurfaceTarget,
) -> List[Dict[str, object]]:
    """Resolve one target closure completely, independent of the host."""

    if not dependency_lines:
        return []
    target_environment = _target_marker_environment(target)
    resolved_entries: Dict[str, Dict[str, object]] = {}
    selected_extras: Dict[str, set[str]] = {}
    scanned_extras: Dict[str, set[str]] = {}
    metadata_cache: Dict[Tuple[str, str, str], List[str]] = {}
    top_level_requirements = _iter_active_target_requirements(
        dependency_lines,
        target_environment=target_environment,
    )
    for requirement in top_level_requirements:
        _record_selected_extras(selected_extras, requirement)
    scan_queue = set(
        _merge_target_report_entries(
            resolved_entries,
            _run_pip_hash_target_report(
                repo_root,
                dependency_lines=dependency_lines,
                target=target,
            ),
        )
    )
    while True:
        missing_requirements: List[Requirement] = []
        for requirement in top_level_requirements:
            if not _resolved_requirement_satisfies(
                requirement,
                resolved_entries,
            ):
                missing_requirements.append(requirement)
        packages_to_scan = sorted(scan_queue)
        scan_queue.clear()
        for normalized_name in packages_to_scan:
            extras = set(selected_extras.get(normalized_name, set()))
            if (
                normalized_name in scanned_extras
                and extras == scanned_extras[normalized_name]
            ):
                continue
            entry = resolved_entries[normalized_name]
            metadata = entry.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            distribution_name = str(metadata.get("name", "")).strip()
            version = str(metadata.get("version", "")).strip()
            if not distribution_name or not version:
                continue
            requirement_lines = _load_target_distribution_requirements(
                repo_root,
                target=target,
                distribution_name=distribution_name,
                version=version,
                metadata_cache=metadata_cache,
            )
            active_requirements = _iter_active_target_requirements(
                requirement_lines,
                target_environment=target_environment,
                selected_extras=extras,
            )
            scanned_extras[normalized_name] = set(extras)
            for requirement in active_requirements:
                dependency_name = (
                    dependency_management._normalize_distribution_name(
                        requirement.name
                    )
                )
                extras_grew = _record_selected_extras(
                    selected_extras,
                    requirement,
                )
                if dependency_name in resolved_entries and extras_grew:
                    scan_queue.add(dependency_name)
                if not _resolved_requirement_satisfies(
                    requirement,
                    resolved_entries,
                ):
                    missing_requirements.append(requirement)
        pending_lines = sorted(
            {
                _canonicalize_requirement_string(requirement)
                for requirement in missing_requirements
            }
        )
        if not pending_lines:
            break
        scan_queue.update(
            _merge_target_report_entries(
                resolved_entries,
                _run_pip_hash_target_report(
                    repo_root,
                    dependency_lines=pending_lines,
                    target=target,
                ),
            )
        )
    reachable_names = _collect_reachable_target_entries(
        repo_root,
        resolved_entries=resolved_entries,
        top_level_requirements=top_level_requirements,
        target=target,
        target_environment=target_environment,
        selected_extras=selected_extras,
        metadata_cache=metadata_cache,
    )
    return [resolved_entries[name] for name in sorted(reachable_names)]


def _collect_reachable_target_entries(
    repo_root: Path,
    *,
    resolved_entries: Mapping[str, Mapping[str, object]],
    top_level_requirements: Sequence[Requirement],
    target: dependency_management.DependencySurfaceTarget,
    target_environment: Mapping[str, str],
    selected_extras: Mapping[str, set[str]],
    metadata_cache: Dict[Tuple[str, str, str], List[str]],
) -> set[str]:
    """Return only packages reachable through active target requirements."""

    reachable: set[str] = set()
    scan_queue: List[str] = []
    for requirement in top_level_requirements:
        if not _resolved_requirement_satisfies(requirement, resolved_entries):
            raise RuntimeError(
                "Target dependency closure did not resolve the active "
                f"requirement `{requirement}`."
            )
        normalized_name = dependency_management._normalize_distribution_name(
            requirement.name
        )
        if normalized_name not in reachable:
            reachable.add(normalized_name)
            scan_queue.append(normalized_name)
    scanned_states: set[Tuple[str, Tuple[str, ...]]] = set()
    while scan_queue:
        normalized_name = scan_queue.pop(0)
        entry = resolved_entries.get(normalized_name)
        if entry is None:
            continue
        extras = tuple(sorted(selected_extras.get(normalized_name, set())))
        state = (normalized_name, extras)
        if state in scanned_states:
            continue
        scanned_states.add(state)
        metadata = entry.get("metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        distribution_name = str(metadata.get("name", "")).strip()
        version = str(metadata.get("version", "")).strip()
        if not distribution_name or not version:
            continue
        requirement_lines = _load_target_distribution_requirements(
            repo_root,
            target=target,
            distribution_name=distribution_name,
            version=version,
            metadata_cache=metadata_cache,
        )
        active_requirements = _iter_active_target_requirements(
            requirement_lines,
            target_environment=target_environment,
            selected_extras=extras,
        )
        for requirement in active_requirements:
            dependency_name = (
                dependency_management._normalize_distribution_name(
                    requirement.name
                )
            )
            if not _resolved_requirement_satisfies(
                requirement,
                resolved_entries,
            ):
                raise RuntimeError(
                    "Target dependency closure is missing reachable "
                    f"requirement `{requirement}`."
                )
            if dependency_name not in reachable:
                reachable.add(dependency_name)
                scan_queue.append(dependency_name)
    return reachable


def _merge_target_reports(
    *,
    targets: Sequence[dependency_management.DependencySurfaceTarget],
    report_entries: Mapping[str, Sequence[Mapping[str, object]]],
    source_display_name: str,
    generate_hashes: bool,
) -> List[str]:
    """Build one deterministic requirements body from target reports."""

    grouped = _group_target_reports(
        targets=targets,
        report_entries=report_entries,
    )
    return _render_grouped_target_reports(
        targets=targets,
        grouped=grouped,
        source_display_name=source_display_name,
        generate_hashes=generate_hashes,
    )


def _group_target_reports(
    *,
    targets: Sequence[dependency_management.DependencySurfaceTarget],
    report_entries: Mapping[str, Sequence[Mapping[str, object]]],
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Return grouped package/version entries merged across targets."""

    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    for target in targets:
        installs = report_entries.get(target.target_id, [])
        for item in installs:
            if not isinstance(item, Mapping):
                continue
            metadata = item.get("metadata", {})
            if not isinstance(metadata, Mapping):
                continue
            name = str(metadata.get("name", "")).strip()
            version = str(metadata.get("version", "")).strip()
            if not name or not version:
                continue
            archive_hashes = (
                item.get("download_info", {})
                .get("archive_info", {})
                .get("hashes", {})
            )
            hashes: set[str] = set()
            if isinstance(archive_hashes, Mapping):
                sha256 = str(archive_hashes.get("sha256", "")).strip()
                if sha256:
                    hashes.add(sha256)
            if not hashes:
                raise RuntimeError(
                    "pip hash-target report did not provide sha256 hashes "
                    f"for `{name}=={version}` in target "
                    f"`{target.target_id}`."
                )
            _merge_grouped_target_entry(
                grouped,
                display_name=name,
                version=version,
                hashes=hashes,
                target_ids={target.target_id},
            )
    return grouped


def _group_inherited_lock_files(
    repo_root: Path,
    *,
    lock_files: Sequence[str],
    targets: Sequence[dependency_management.DependencySurfaceTarget],
    generate_hashes: bool,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Return grouped entries parsed from inherited managed lock files."""

    grouped: Dict[str, Dict[str, Dict[str, object]]] = {}
    if not lock_files:
        return grouped
    target_environments = {
        target.target_id: _target_marker_environment(target)
        for target in targets
    }
    for raw_lock_file in lock_files:
        lock_token = _normalize_repo_relative_path_token(raw_lock_file)
        if not lock_token:
            continue
        lock_path = repo_root / lock_token
        if not lock_path.exists():
            raise RuntimeError(
                "dependency-management inherited lock input is missing: "
                f"{lock_token}"
            )
        for requirement_line, hashes in _parse_locked_requirements(lock_path):
            try:
                requirement = Requirement(requirement_line)
            except InvalidRequirement as error:
                raise RuntimeError(
                    "dependency-management inherited lock file contains an "
                    f"invalid requirement line: `{requirement_line}`."
                ) from error
            match = dependency_management._PYTHON_LOCK_PIN_RE.match(
                requirement_line
            )
            if match is None:
                raise RuntimeError(
                    "dependency-management inherited lock file must contain "
                    f"exact pins, but found `{requirement_line}`."
                )
            target_ids = {
                target_id
                for target_id, target_environment in (
                    target_environments.items()
                )
                if _requirement_is_active_for_target(
                    requirement,
                    target_environment=target_environment,
                )
            }
            if not target_ids:
                continue
            if generate_hashes and not hashes:
                raise RuntimeError(
                    "dependency-management inherited hash-locked surface "
                    f"`{lock_token}` is missing hashes for "
                    f"`{requirement_line}`."
                )
            _merge_grouped_target_entry(
                grouped,
                display_name=str(match.group("name")),
                version=str(match.group("version")),
                hashes=hashes,
                target_ids=target_ids,
            )
    return grouped


def _parse_locked_requirements(
    lock_path: Path,
) -> List[Tuple[str, List[str]]]:
    """Return pinned requirement lines and hashes from one flat lock file."""

    parsed: List[Tuple[str, List[str]]] = []
    lines = lock_path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or raw_line[:1].isspace():
            index += 1
            continue
        requirement_line = stripped.rstrip("\\").strip()
        hashes: List[str] = []
        index += 1
        while index < len(lines):
            continuation = lines[index]
            continuation_stripped = continuation.strip()
            if not continuation_stripped:
                index += 1
                break
            if not continuation[:1].isspace():
                break
            if continuation_stripped.startswith("--hash=sha256:"):
                digest = (
                    continuation_stripped.removeprefix("--hash=sha256:")
                    .rstrip("\\")
                    .strip()
                )
                if digest:
                    hashes.append(digest)
            index += 1
        parsed.append((requirement_line, hashes))
    return parsed


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _vulnerability_audit_cache_dir() -> Path:
    """Return the machine-local cache root for vulnerability lookups."""

    return (
        Path(tempfile.gettempdir())
        / "devcovenant"
        / "dependency_vulnerability_cache"
    )


def _vulnerability_audit_cache_path(
    *,
    normalized_name: str,
    version: str,
) -> Path:
    """Return one stable machine-local cache file path."""

    digest = hashlib.sha256(
        f"{normalized_name}=={version}".encode("utf-8")
    ).hexdigest()
    return _vulnerability_audit_cache_dir() / f"{digest}.json"


def _parse_iso_utc(value: object) -> datetime | None:
    """Parse one cached ISO-8601 timestamp into a UTC datetime."""

    token = str(value or "").strip()
    if not token:
        return None
    try:
        parsed = datetime.fromisoformat(token.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_vulnerability_feed(
    raw_entries: object,
) -> List[Dict[str, object]]:
    """Normalize one PyPI vulnerability payload into stable mappings."""

    if not isinstance(raw_entries, list):
        return []
    normalized: List[Dict[str, object]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        vulnerability_id = str(raw_entry.get("id", "")).strip()
        if not vulnerability_id:
            continue
        withdrawn = str(raw_entry.get("withdrawn", "")).strip()
        if withdrawn:
            continue
        aliases = tuple(
            sorted(
                {
                    str(alias).strip().upper()
                    for alias in raw_entry.get("aliases", [])
                    if str(alias).strip()
                }
            )
        )
        raw_fixed_in = raw_entry.get("fixed_in", [])
        if isinstance(raw_fixed_in, list):
            fixed_in = tuple(
                sorted(
                    {
                        str(version).strip()
                        for version in raw_fixed_in
                        if str(version).strip()
                    }
                )
            )
        else:
            fixed_in = tuple(
                [str(raw_fixed_in).strip()]
                if str(raw_fixed_in).strip()
                else []
            )
        summary = str(
            raw_entry.get("summary") or raw_entry.get("details") or ""
        ).strip()
        normalized.append(
            {
                "id": vulnerability_id.upper(),
                "aliases": list(aliases),
                "fixed_in": list(fixed_in),
                "summary": summary,
            }
        )
    return normalized


def _load_cached_vulnerability_feed(
    *,
    normalized_name: str,
    version: str,
    now: datetime,
) -> List[Dict[str, object]] | None:
    """Return a fresh cached vulnerability payload when available."""

    cache_path = _vulnerability_audit_cache_path(
        normalized_name=normalized_name,
        version=version,
    )
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    checked_at = _parse_iso_utc(payload.get("checked_at_utc"))
    if checked_at is None:
        return None
    if now - checked_at > _VULNERABILITY_AUDIT_CACHE_TTL:
        return None
    return _normalize_vulnerability_feed(payload.get("vulnerabilities", []))


def _write_cached_vulnerability_feed(
    *,
    normalized_name: str,
    version: str,
    vulnerabilities: Sequence[Mapping[str, object]],
    now: datetime,
) -> None:
    """Persist one vulnerability payload in the machine-local cache."""

    cache_path = _vulnerability_audit_cache_path(
        normalized_name=normalized_name,
        version=version,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_at_utc": now.isoformat(),
        "name": normalized_name,
        "version": version,
        "vulnerabilities": list(vulnerabilities),
    }
    cache_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _query_pypi_vulnerabilities(
    normalized_name: str,
    version: str,
) -> List[Dict[str, object]]:
    """Query the PyPI JSON API for one exact package version."""

    url = (
        "https://pypi.org/pypi/"
        f"{urllib_parse.quote(normalized_name)}/"
        f"{urllib_parse.quote(version)}/json"
    )
    request = urllib_request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "devcovenant/dependency-management",
        },
    )
    try:
        with urllib_request.urlopen(  # nosec B310
            request,
            timeout=_VULNERABILITY_AUDIT_TIMEOUT_SECONDS,
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib_error.URLError as error:
        raise RuntimeError(
            "dependency-management vulnerability audit could not query "
            f"PyPI for `{normalized_name}=={version}`: {error}."
        ) from error
    raw_entries = payload.get("vulnerabilities")
    if raw_entries is None and isinstance(payload.get("info"), Mapping):
        raw_entries = payload.get("info", {}).get("vulnerabilities", [])
    return _normalize_vulnerability_feed(raw_entries)


def _surface_vulnerability_feed(
    *,
    normalized_name: str,
    version: str,
    runtime_cache: Dict[str, object] | None = None,
) -> List[Dict[str, object]]:
    """Return vulnerability data for one exact package/version pair."""

    cache_key = f"{normalized_name}=={version}"
    cached_payloads: Dict[str, object] = {}
    if isinstance(runtime_cache, dict):
        payload_bucket = runtime_cache.get("pypi_vulnerabilities")
        if not isinstance(payload_bucket, dict):
            payload_bucket = {}
            runtime_cache["pypi_vulnerabilities"] = payload_bucket
        cached_payloads = payload_bucket
        cached = cached_payloads.get(cache_key)
        if isinstance(cached, list):
            return cached
    now = _utc_now()
    cached = _load_cached_vulnerability_feed(
        normalized_name=normalized_name,
        version=version,
        now=now,
    )
    if cached is not None:
        if isinstance(runtime_cache, dict):
            cached_payloads[cache_key] = cached
        return cached
    vulnerabilities = _query_pypi_vulnerabilities(
        normalized_name,
        version,
    )
    _write_cached_vulnerability_feed(
        normalized_name=normalized_name,
        version=version,
        vulnerabilities=vulnerabilities,
        now=now,
    )
    if isinstance(runtime_cache, dict):
        cached_payloads[cache_key] = vulnerabilities
    return vulnerabilities


def _collect_surface_audit_targets(
    requirement: Requirement,
    *,
    surface: dependency_management.DependencySurface,
    target_environments: Mapping[str, Mapping[str, str]],
) -> Tuple[str, ...]:
    """Return declared target ids affected by one lock requirement."""

    if not surface.hash_targets:
        return ()
    return tuple(
        sorted(
            target_id
            for target_id, target_environment in (target_environments.items())
            if _requirement_is_active_for_target(
                requirement,
                target_environment=target_environment,
            )
        )
    )


def _normalized_audit_ignore_ids(
    ignore_ids: Sequence[str],
) -> set[str]:
    """Return the normalized ignore-id set for one audited surface."""

    return {
        str(entry).strip().upper()
        for entry in ignore_ids
        if str(entry).strip()
    }


def audit_surface_vulnerabilities(
    repo_root: Path,
    *,
    surface: dependency_management.DependencySurface,
    runtime_cache: Dict[str, object] | None = None,
) -> List[DependencySurfaceVulnerability]:
    """Audit one managed dependency surface for published vulnerabilities."""

    if not surface.audit_service:
        return []
    lock_token = _normalize_repo_relative_path_token(surface.lock_file)
    lock_path = repo_root / lock_token
    if not lock_path.exists():
        raise RuntimeError(
            "dependency-management vulnerability audit requires the "
            f"declared lock file for `{surface.surface_id}`, but "
            f"`{lock_token}` is missing."
        )
    target_environments = {
        target.target_id: _target_marker_environment(target)
        for target in surface.hash_targets
    }
    grouped_requirements: Dict[Tuple[str, str], Dict[str, object]] = {}
    for requirement_line, _ in _parse_locked_requirements(lock_path):
        try:
            requirement = Requirement(requirement_line)
        except InvalidRequirement as error:
            raise RuntimeError(
                "dependency-management vulnerability audit found an invalid "
                f"lock requirement in `{lock_token}`: `{requirement_line}`."
            ) from error
        match = dependency_management._PYTHON_LOCK_PIN_RE.match(
            requirement_line
        )
        if match is None:
            continue
        package_name = str(match.group("name"))
        version = str(match.group("version"))
        normalized_name = dependency_management._normalize_distribution_name(
            package_name
        )
        target_ids = _collect_surface_audit_targets(
            requirement,
            surface=surface,
            target_environments=target_environments,
        )
        if surface.hash_targets and not target_ids:
            continue
        key = (normalized_name, version)
        entry = grouped_requirements.setdefault(
            key,
            {
                "package_name": package_name,
                "version": version,
                "target_ids": set(),
            },
        )
        entry["target_ids"].update(target_ids)
    if not grouped_requirements:
        return []
    ignore_ids = _normalized_audit_ignore_ids(surface.audit_ignore_ids)
    findings: List[DependencySurfaceVulnerability] = []
    package_entries = sorted(grouped_requirements.items())
    max_workers = min(
        len(package_entries),
        max(1, _VULNERABILITY_AUDIT_MAX_WORKERS),
    )
    vulnerability_feeds: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    if max_workers <= 1:
        for (normalized_name, version), _entry in package_entries:
            vulnerability_feeds[(normalized_name, version)] = (
                _surface_vulnerability_feed(
                    normalized_name=normalized_name,
                    version=version,
                    runtime_cache=runtime_cache,
                )
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            submitted = {
                executor.submit(
                    _surface_vulnerability_feed,
                    normalized_name=normalized_name,
                    version=version,
                    runtime_cache=None,
                ): (normalized_name, version)
                for (normalized_name, version), _entry in package_entries
            }
            for future, key in submitted.items():
                vulnerability_feeds[key] = future.result()
    for (normalized_name, version), entry in package_entries:
        package_name = str(entry.get("package_name", "")).strip()
        target_ids = tuple(sorted(set(entry.get("target_ids", set()))))
        for vulnerability in vulnerability_feeds[(normalized_name, version)]:
            candidate_ids = {str(vulnerability.get("id", "")).strip().upper()}
            candidate_ids.update(
                {
                    str(alias).strip().upper()
                    for alias in vulnerability.get("aliases", [])
                    if str(alias).strip()
                }
            )
            if candidate_ids & ignore_ids:
                continue
            findings.append(
                DependencySurfaceVulnerability(
                    package_name=package_name,
                    version=version,
                    vulnerability_id=str(vulnerability.get("id", "")).strip(),
                    aliases=tuple(
                        str(alias).strip()
                        for alias in vulnerability.get("aliases", [])
                        if str(alias).strip()
                    ),
                    fix_versions=tuple(
                        str(item).strip()
                        for item in vulnerability.get("fixed_in", [])
                        if str(item).strip()
                    ),
                    summary=str(vulnerability.get("summary", "")).strip(),
                    target_ids=target_ids,
                )
            )
    return findings


def _merge_grouped_target_entry(
    grouped: Dict[str, Dict[str, Dict[str, object]]],
    *,
    display_name: str,
    version: str,
    hashes: Iterable[str],
    target_ids: set[str],
) -> None:
    """Merge one grouped package entry and reject overlapping versions."""

    normalized_name = dependency_management._normalize_distribution_name(
        display_name
    )
    version_map = grouped.setdefault(normalized_name, {})
    for existing_version, existing_entry in version_map.items():
        if existing_version == version:
            continue
        overlapping_targets = set(existing_entry["targets"]) & set(target_ids)
        if overlapping_targets:
            listed_targets = ", ".join(sorted(overlapping_targets))
            raise RuntimeError(
                "dependency-management inherited surfaces produced "
                f"conflicting versions for `{display_name}` in targets "
                f"`{listed_targets}`: `{existing_version}` vs `{version}`."
            )
    entry = version_map.setdefault(
        version,
        {
            "display_name": display_name,
            "hashes": set(),
            "targets": set(),
        },
    )
    entry["hashes"].update(
        {
            str(entry_hash).strip()
            for entry_hash in hashes
            if str(entry_hash).strip()
        }
    )
    entry["targets"].update(target_ids)


def _render_grouped_target_reports(
    *,
    targets: Sequence[dependency_management.DependencySurfaceTarget],
    grouped: Dict[str, Dict[str, Dict[str, object]]],
    source_display_name: str,
    generate_hashes: bool,
) -> List[str]:
    """Render grouped package/version entries into one lock body."""

    ordered_target_ids = [target.target_id for target in targets]
    target_markers = {target.target_id: target.marker for target in targets}
    all_target_ids = set(ordered_target_ids)
    lines: List[str] = [
        "# This file is autogenerated by DevCovenant dependency-management.",
        "",
    ]
    for normalized_name in sorted(grouped):
        versions = grouped[normalized_name]
        for version in sorted(versions):
            entry = versions[version]
            display_name = str(entry["display_name"])
            target_ids = set(entry["targets"])
            requirement_line = f"{display_name}=={version}"
            if target_ids != all_target_ids:
                marker_parts = [
                    f"({target_markers[target_id]})"
                    for target_id in ordered_target_ids
                    if target_id in target_ids
                ]
                requirement_line += " ; " + " or ".join(marker_parts)
            if generate_hashes:
                lines.extend(
                    _build_hashed_requirement_block(
                        requirement_line,
                        sorted(entry["hashes"]),
                        source_display_name=source_display_name,
                    )
                )
            else:
                lines.extend(
                    _build_plain_requirement_block(
                        requirement_line,
                        source_display_name=source_display_name,
                    )
                )
            lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def _compile_target_surface_lock(
    repo_root: Path,
    *,
    surface_id: str,
    dependency_files: Sequence[str],
    provider_lock_files: Sequence[str],
    hash_targets: Sequence[dependency_management.DependencySurfaceTarget],
    source_display_name: str,
    generate_hashes: bool,
) -> LockFilePieces:
    """Resolve one deterministic lock body across configured targets."""

    del surface_id
    resolution_inputs = _surface_resolution_inputs(
        repo_root,
        dependency_files=dependency_files,
        provider_lock_files=provider_lock_files,
    )
    dependency_lines = resolution_inputs.dependency_lines
    inherited_grouped = _group_inherited_lock_files(
        repo_root,
        lock_files=resolution_inputs.inherited_lock_files,
        targets=hash_targets,
        generate_hashes=generate_hashes,
    )
    if not dependency_lines and not inherited_grouped:
        return LockFilePieces([])
    reports: Dict[str, Sequence[Mapping[str, object]]] = {}
    max_workers = min(
        len(hash_targets),
        max(1, _TARGET_RESOLUTION_MAX_WORKERS),
    )
    if dependency_lines and max_workers <= 1:
        for target in hash_targets:
            reports[target.target_id] = _resolve_complete_target_report(
                repo_root,
                dependency_lines=dependency_lines,
                target=target,
            )
    elif dependency_lines:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            submitted = [
                (
                    target,
                    executor.submit(
                        _resolve_complete_target_report,
                        repo_root,
                        dependency_lines=dependency_lines,
                        target=target,
                    ),
                )
                for target in hash_targets
            ]
            for target, future in submitted:
                reports[target.target_id] = future.result()
    grouped = dict(inherited_grouped)
    resolved_grouped = _group_target_reports(
        targets=hash_targets,
        report_entries=reports,
    )
    for version_map in resolved_grouped.values():
        for version, entry in version_map.items():
            _merge_grouped_target_entry(
                grouped,
                display_name=str(entry["display_name"]),
                version=str(version),
                hashes=entry["hashes"],
                target_ids=set(entry["targets"]),
            )
    return LockFilePieces(
        _render_grouped_target_reports(
            targets=hash_targets,
            grouped=grouped,
            source_display_name=source_display_name,
            generate_hashes=generate_hashes,
        )
    )


def _refresh_npm_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh package-lock.json from package.json."""

    package_json = repo_root / "package.json"
    lock_path = repo_root / "package-lock.json"
    if not package_json.exists():
        return LockHandlerResult(
            "package-lock.json",
            changed=False,
            attempted=False,
            message="Skipped: package.json missing.",
        )
    if not _ensure_tool("npm"):
        return LockHandlerResult(
            "package-lock.json",
            changed=False,
            attempted=False,
            message="Skipped: npm not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("npm", "install", "--package-lock-only"),
    )
    message = "Updated package-lock.json." if changed else "No change."
    return LockHandlerResult("package-lock.json", changed, True, message)


def _refresh_yarn_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh yarn.lock from package.json."""

    package_json = repo_root / "package.json"
    lock_path = repo_root / "yarn.lock"
    if not package_json.exists():
        return LockHandlerResult(
            "yarn.lock",
            changed=False,
            attempted=False,
            message="Skipped: package.json missing.",
        )
    if not _ensure_tool("yarn"):
        return LockHandlerResult(
            "yarn.lock",
            changed=False,
            attempted=False,
            message="Skipped: yarn not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("yarn", "install", "--mode=update-lockfile"),
    )
    message = "Updated yarn.lock." if changed else "No change."
    return LockHandlerResult("yarn.lock", changed, True, message)


def _refresh_pnpm_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh pnpm-lock.yaml from package.json."""

    package_json = repo_root / "package.json"
    lock_path = repo_root / "pnpm-lock.yaml"
    if not package_json.exists():
        return LockHandlerResult(
            "pnpm-lock.yaml",
            changed=False,
            attempted=False,
            message="Skipped: package.json missing.",
        )
    if not _ensure_tool("pnpm"):
        return LockHandlerResult(
            "pnpm-lock.yaml",
            changed=False,
            attempted=False,
            message="Skipped: pnpm not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("pnpm", "install", "--lockfile-only"),
    )
    message = "Updated pnpm-lock.yaml." if changed else "No change."
    return LockHandlerResult("pnpm-lock.yaml", changed, True, message)


def _refresh_go_sum(repo_root: Path) -> LockHandlerResult:
    """Refresh go.sum from go.mod."""

    go_mod = repo_root / "go.mod"
    lock_path = repo_root / "go.sum"
    if not go_mod.exists():
        return LockHandlerResult(
            "go.sum",
            changed=False,
            attempted=False,
            message="Skipped: go.mod missing.",
        )
    if not _ensure_tool("go"):
        return LockHandlerResult(
            "go.sum",
            changed=False,
            attempted=False,
            message="Skipped: go not installed.",
        )
    changed = _run_and_detect_change(
        repo_root, lock_path, ("go", "mod", "tidy")
    )
    message = "Updated go.sum." if changed else "No change."
    return LockHandlerResult("go.sum", changed, True, message)


def _refresh_cargo_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh Cargo.lock from Cargo.toml."""

    cargo_toml = repo_root / "Cargo.toml"
    lock_path = repo_root / "Cargo.lock"
    if not cargo_toml.exists():
        return LockHandlerResult(
            "Cargo.lock",
            changed=False,
            attempted=False,
            message="Skipped: Cargo.toml missing.",
        )
    if not _ensure_tool("cargo"):
        return LockHandlerResult(
            "Cargo.lock",
            changed=False,
            attempted=False,
            message="Skipped: cargo not installed.",
        )
    changed = _run_and_detect_change(
        repo_root, lock_path, ("cargo", "generate-lockfile")
    )
    message = "Updated Cargo.lock." if changed else "No change."
    return LockHandlerResult("Cargo.lock", changed, True, message)


def _refresh_composer_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh composer.lock from composer.json."""

    composer_json = repo_root / "composer.json"
    lock_path = repo_root / "composer.lock"
    if not composer_json.exists():
        return LockHandlerResult(
            "composer.lock",
            changed=False,
            attempted=False,
            message="Skipped: composer.json missing.",
        )
    if not _ensure_tool("composer"):
        return LockHandlerResult(
            "composer.lock",
            changed=False,
            attempted=False,
            message="Skipped: composer not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("composer", "update", "--lock", "--no-install"),
    )
    message = "Updated composer.lock." if changed else "No change."
    return LockHandlerResult("composer.lock", changed, True, message)


def _refresh_gemfile_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh Gemfile.lock from Gemfile."""

    gemfile = repo_root / "Gemfile"
    lock_path = repo_root / "Gemfile.lock"
    if not gemfile.exists():
        return LockHandlerResult(
            "Gemfile.lock",
            changed=False,
            attempted=False,
            message="Skipped: Gemfile missing.",
        )
    if not _ensure_tool("bundle"):
        return LockHandlerResult(
            "Gemfile.lock",
            changed=False,
            attempted=False,
            message="Skipped: bundler not installed.",
        )
    changed = _run_and_detect_change(repo_root, lock_path, ("bundle", "lock"))
    message = "Updated Gemfile.lock." if changed else "No change."
    return LockHandlerResult("Gemfile.lock", changed, True, message)


def _refresh_pubspec_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh pubspec.lock from pubspec.yaml."""

    pubspec = repo_root / "pubspec.yaml"
    lock_path = repo_root / "pubspec.lock"
    if not pubspec.exists():
        return LockHandlerResult(
            "pubspec.lock",
            changed=False,
            attempted=False,
            message="Skipped: pubspec.yaml missing.",
        )
    if _ensure_tool("flutter"):
        changed = _run_and_detect_change(
            repo_root, lock_path, ("flutter", "pub", "get")
        )
        message = (
            "Updated pubspec.lock via flutter." if changed else "No change."
        )
        return LockHandlerResult("pubspec.lock", changed, True, message)
    if _ensure_tool("dart"):
        changed = _run_and_detect_change(
            repo_root, lock_path, ("dart", "pub", "get")
        )
        message = "Updated pubspec.lock via dart." if changed else "No change."
        return LockHandlerResult("pubspec.lock", changed, True, message)
    return LockHandlerResult(
        "pubspec.lock",
        changed=False,
        attempted=False,
        message="Skipped: neither flutter nor dart is installed.",
    )


def _refresh_podfile_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh Podfile.lock from Podfile."""

    podfile = repo_root / "Podfile"
    lock_path = repo_root / "Podfile.lock"
    if not podfile.exists():
        return LockHandlerResult(
            "Podfile.lock",
            changed=False,
            attempted=False,
            message="Skipped: Podfile missing.",
        )
    if not _ensure_tool("pod"):
        return LockHandlerResult(
            "Podfile.lock",
            changed=False,
            attempted=False,
            message="Skipped: cocoapods is not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("pod", "install", "--no-repo-update"),
    )
    message = "Updated Podfile.lock." if changed else "No change."
    return LockHandlerResult("Podfile.lock", changed, True, message)


def _refresh_dotnet_lock(repo_root: Path) -> LockHandlerResult:
    """Refresh packages.lock.json when .NET projects are present."""

    lock_path = repo_root / "packages.lock.json"
    csproj_files = list(repo_root.glob("*.csproj"))
    if not csproj_files:
        return LockHandlerResult(
            "packages.lock.json",
            changed=False,
            attempted=False,
            message="Skipped: no top-level *.csproj file found.",
        )
    if not _ensure_tool("dotnet"):
        return LockHandlerResult(
            "packages.lock.json",
            changed=False,
            attempted=False,
            message="Skipped: dotnet not installed.",
        )
    changed = _run_and_detect_change(
        repo_root,
        lock_path,
        ("dotnet", "restore", "--use-lock-file"),
    )
    message = "Updated packages.lock.json." if changed else "No change."
    return LockHandlerResult("packages.lock.json", changed, True, message)


LOCKFILE_HANDLERS: Dict[str, Callable[[Path], LockHandlerResult]] = {
    "requirements.lock": _refresh_python_requirements_lock,
    "package-lock.json": _refresh_npm_lock,
    "yarn.lock": _refresh_yarn_lock,
    "pnpm-lock.yaml": _refresh_pnpm_lock,
    "go.sum": _refresh_go_sum,
    "Cargo.lock": _refresh_cargo_lock,
    "composer.lock": _refresh_composer_lock,
    "Gemfile.lock": _refresh_gemfile_lock,
    "pubspec.lock": _refresh_pubspec_lock,
    "Podfile.lock": _refresh_podfile_lock,
    "packages.lock.json": _refresh_dotnet_lock,
}


def _descriptor_metadata_lists(
    repo_root: Path,
    policy_id: str,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load descriptor defaults into order/list map representation."""

    descriptor = load_policy_descriptor(repo_root, policy_id)
    if descriptor is None:
        raise ValueError(f"Missing policy descriptor for `{policy_id}`.")
    order: List[str] = []
    values: Dict[str, List[str]] = {}
    for key, raw_value in descriptor.metadata.items():
        key_name = str(key).strip()
        if not key_name:
            continue
        order.append(key_name)
        values[key_name] = metadata_value_list(raw_value)
    return order, values


def _resolve_dependency_metadata(repo_root: Path) -> Dict[str, object]:
    """Resolve dependency-management metadata from profiles and config."""

    order, values = _descriptor_metadata_lists(repo_root, POLICY_ID)
    descriptor = load_policy_descriptor(repo_root, POLICY_ID)
    context = build_metadata_context(repo_root)
    location = resolve_script_location(repo_root, POLICY_ID)
    custom_policy = bool(location and location.kind == "custom")
    bundle = resolve_policy_metadata_bundle(
        POLICY_ID,
        order,
        values,
        descriptor,
        context,
        custom_policy=custom_policy,
    )
    options = bundle.decode_options()
    surfaces = dependency_management.resolve_dependency_surfaces(
        repo_root=repo_root,
        raw_surfaces=options.get("surfaces", []),
        include_inactive=True,
    )
    license_source_overrides = (
        dependency_management.resolve_license_source_overrides(
            options.get("license_source_overrides", [])
        )
    )
    return {
        "surfaces": surfaces,
        "license_source_overrides": license_source_overrides,
    }


def refresh_all(
    repo_root: Path,
    *,
    payload: dict[str, object] | None = None,
) -> dict[str, object]:
    """Refresh selected lockfiles and dependency-management artifacts."""

    metadata = _resolve_dependency_metadata(repo_root)
    surfaces = [
        surface
        for surface in metadata.get("surfaces", [])
        if isinstance(surface, dependency_management.DependencySurface)
        and surface.active
    ]
    license_source_overrides = metadata.get("license_source_overrides", {})
    surfaces = _order_surfaces_for_refresh(surfaces)
    registry = PolicyRegistry(policy_registry_path(repo_root), repo_root)
    stored_runtime_state = registry.get_policy_runtime_state(POLICY_ID)
    stored_surface_states = stored_runtime_state.get("surfaces", {})
    if not isinstance(stored_surface_states, Mapping):
        stored_surface_states = {}
    updated_surface_states: dict[str, dict[str, object]] = {}
    current_surface_ids: set[str] = set()
    results: List[LockHandlerResult] = []
    changed_lockfiles: List[str] = []
    requested_dependency_files: list[str] = []
    if isinstance(payload, dict):
        raw_files = payload.get("changed_dependency_files")
        if isinstance(raw_files, list):
            for entry in raw_files:
                text = str(entry).strip()
                if text:
                    requested_dependency_files.append(text)
    for surface in surfaces:
        lock_name = surface.lock_file
        stored_surface_state = stored_surface_states.get(surface.surface_id)
        if (
            not requested_dependency_files
            and _surface_runtime_state_is_current(
                repo_root,
                surface=surface,
                runtime_state=stored_surface_state,
            )
        ):
            results.append(
                LockHandlerResult(
                    lock_name,
                    changed=False,
                    attempted=False,
                    message="Skipped: surface artifacts already current.",
                )
            )
            updated_surface_states[surface.surface_id] = dict(
                stored_surface_state
            )
            current_surface_ids.add(surface.surface_id)
            continue
        if not dependency_management.dependency_surface_lock_refresh_requested(
            surface,
            requested_dependency_files,
        ):
            results.append(
                LockHandlerResult(
                    lock_name,
                    changed=False,
                    attempted=False,
                    message="Skipped: no direct lock inputs changed.",
                )
            )
            continue
        result = _refresh_python_surface_lock(
            repo_root,
            surface=surface,
            all_surfaces=surfaces,
        )
        results.append(result)
        if result.changed:
            changed_lockfiles.append(lock_name)
    changed_dependency_files: List[str] = []
    for dependency_file in requested_dependency_files:
        changed_dependency_files.append(dependency_file)
    for lock_name in changed_lockfiles:
        changed_dependency_files.append(lock_name)
    if not changed_dependency_files:
        changed_dependency_files = list(changed_lockfiles)
    changed_dependency_files = list(
        dict.fromkeys(
            str(entry).strip()
            for entry in changed_dependency_files
            if str(entry).strip()
        )
    )
    modified_license_files: List[Path] = []
    for surface in surfaces:
        if surface.surface_id in current_surface_ids:
            continue
        if not surface.direct_dependency_files:
            continue
        surface_changed_dependency_files = [
            entry
            for entry in changed_dependency_files
            if dependency_management.dependency_surface_matches(
                surface,
                entry,
            )
        ]
        if not requested_dependency_files:
            surface_changed_dependency_files = sorted(
                dependency_management.dependency_surface_trigger_files(surface)
            )
        modified_license_files.extend(
            dependency_management.refresh_license_artifacts(
                repo_root,
                changed_dependency_files=surface_changed_dependency_files,
                third_party_file=surface.third_party_file,
                licenses_dir=surface.licenses_dir,
                report_heading=surface.report_heading,
                resolved_lock_file=surface.lock_file,
                direct_dependency_files=surface.direct_dependency_files,
                manage_licenses_readme=surface.manage_licenses_readme,
                license_source_overrides=license_source_overrides,
            )
        )
        updated_surface_states[surface.surface_id] = (
            _build_surface_runtime_state(
                repo_root,
                surface=surface,
            )
        )
    registry.update_policy_runtime_state(
        POLICY_ID,
        {
            **(
                stored_runtime_state
                if isinstance(stored_runtime_state, Mapping)
                else {}
            ),
            "surfaces": updated_surface_states,
        },
    )
    result_payload = [
        {
            "lock_file": result.lock_file,
            "changed": result.changed,
            "attempted": result.attempted,
            "message": result.message,
        }
        for result in results
    ]
    refreshed_artifacts = [
        (
            _normalize_repo_relative_path_token(
                os.path.relpath(
                    os.path.realpath(path),
                    os.path.realpath(repo_root),
                )
            )
            if isinstance(path, Path)
            else str(path)
        )
        for path in modified_license_files
    ]
    return {
        "message": (
            "Dependency-management refresh completed."
            if results
            else "No metadata-selected lockfiles are configured for this "
            "repository."
        ),
        "lock_results": result_payload,
        "refreshed_artifacts": refreshed_artifacts,
    }
