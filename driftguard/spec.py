# Last Updated: 2025-11-26
"""DriftGuard policy specification data structures and loader.

The loader validates ``driftguard.yml`` early so the engine can surface clear
errors instead of failing mid-run with confusing missing-field exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

import yaml
from yaml import YAMLError

from driftguard.logging_utils import get_logger


class SpecValidationError(ValueError):
    """Raised when the policy specification is missing required fields.

    The exception type is narrow so callers can distinguish schema issues from
    I/O failures and relay actionable guidance to contributors.
    """


@dataclass
class SurfaceSpec:
    """Map rules to the globs that define a surface.

    Surfaces keep rule targets declarative so future rules never rely on
    hard-coded paths.
    """

    name: str
    include: List[str]
    exclude: List[str]
    rules: List[str]
    description: Optional[str] = None


logger = get_logger()


@dataclass
class MetricThreshold:
    """Configuration for a drift metric defined in the policy spec."""

    name: str
    min_warning: Optional[float] = None
    max_warning: Optional[float] = None
    description: Optional[str] = None


@dataclass
class DriftConfig:
    """Group drift metrics together for clarity."""

    metrics: Dict[str, MetricThreshold] = field(default_factory=dict)


@dataclass
class DriftGuardSpec:
    """Aggregate policy configuration loaded from ``driftguard.yml``."""

    version: int
    project: str
    rulesets: Dict[str, str]
    surfaces: Dict[str, SurfaceSpec]
    drift: DriftConfig
    raw: Dict[str, Any] = field(default_factory=dict)


def _expect_mapping(payload: Any, context: str) -> MutableMapping[str, Any]:
    """Ensure ``payload`` is a mapping, raising a clear error otherwise."""

    if not isinstance(payload, MutableMapping):
        raise SpecValidationError(f"Expected a mapping for {context}.")
    return payload


def _validate_keys(
    payload: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    context: str,
) -> None:
    """Validate missing or unknown keys for a section.

    Early key validation keeps later parsing logic simple and ensures that
    typos in the YAML fail fast with actionable messages instead of producing
    silent defaults.
    """

    allowed = set(required) | set(optional)
    unknown = set(payload) - allowed
    if unknown:
        raise SpecValidationError(
            f"Unknown key(s) {sorted(unknown)} in {context}."
        )
    missing = set(required) - set(payload)
    if missing:
        raise SpecValidationError(
            f"Missing required key(s) {sorted(missing)} in {context}."
        )


def _parse_string_list(values: Any, context: str) -> List[str]:
    """Convert a YAML list into a list of strings with validation."""

    if values is None:
        return []
    if not isinstance(values, list) or not all(
        isinstance(item, str) for item in values
    ):
        raise SpecValidationError(f"Expected a list of strings in {context}.")
    return list(values)


def _parse_rulesets(raw_rulesets: Any) -> Dict[str, str]:
    """Parse enforcement levels for named rulesets."""

    rulesets = _expect_mapping(raw_rulesets, "rulesets")
    parsed: Dict[str, str] = {}
    for name, level in rulesets.items():
        if level not in {"hard", "warn"}:
            raise SpecValidationError(
                "Ruleset level must be 'hard' or 'warn' for ruleset "
                f"{name!r}."
            )
        parsed[str(name)] = str(level)
    return parsed


def _parse_surfaces(raw_surfaces: Any) -> Dict[str, SurfaceSpec]:
    """Convert raw surface definitions into data classes."""

    surfaces = _expect_mapping(raw_surfaces, "surfaces")
    parsed: Dict[str, SurfaceSpec] = {}
    for name, surface in surfaces.items():
        context = f"surface {name!r}"
        surface_mapping = _expect_mapping(surface, context)
        _validate_keys(
            surface_mapping,
            required=["include", "exclude", "rules"],
            optional=["description"],
            context=context,
        )
        parsed[name] = SurfaceSpec(
            name=name,
            include=_parse_string_list(
                surface_mapping.get("include"), context
            ),
            exclude=_parse_string_list(
                surface_mapping.get("exclude"), context
            ),
            rules=_parse_string_list(surface_mapping.get("rules"), context),
            description=surface_mapping.get("description"),
        )
    return parsed


def _parse_drift(raw_drift: Any) -> DriftConfig:
    """Parse drift metric thresholds with validation."""

    drift = _expect_mapping(raw_drift, "drift")
    _validate_keys(drift, required=["metrics"], optional=[], context="drift")
    metrics_raw = drift.get("metrics")
    if not isinstance(metrics_raw, list):
        raise SpecValidationError("Expected a list for drift.metrics.")
    parsed: Dict[str, MetricThreshold] = {}
    for metric in metrics_raw:
        metric_mapping = _expect_mapping(metric, "drift.metrics entry")
        _validate_keys(
            metric_mapping,
            required=["name"],
            optional=["min_warning", "max_warning", "description"],
            context="drift.metrics entry",
        )
        name = str(metric_mapping.get("name"))
        min_warning = metric_mapping.get("min_warning")
        max_warning = metric_mapping.get("max_warning")
        for value, label in (
            (min_warning, "min_warning"),
            (max_warning, "max_warning"),
        ):
            if value is None:
                continue
            if not isinstance(value, (int, float)):
                raise SpecValidationError(
                    f"Expected numeric {label} for metric {name!r}."
                )
        parsed[name] = MetricThreshold(
            name=name,
            min_warning=(
                float(min_warning) if min_warning is not None else None
            ),
            max_warning=(
                float(max_warning) if max_warning is not None else None
            ),
            description=metric_mapping.get("description"),
        )
    return DriftConfig(metrics=parsed)


def _load_yaml(spec_path: Path) -> Dict[str, Any]:
    """Read and parse the YAML spec with defensive error handling."""

    logger.info("Reading DriftGuard spec from %s", spec_path)
    try:
        raw = yaml.safe_load(spec_path.read_text())
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing DriftGuard spec at {spec_path}."
        ) from exc
    except YAMLError as exc:
        raise SpecValidationError(
            f"Failed to parse YAML at {spec_path}: {exc}."
        ) from exc
    if raw is None:
        raise SpecValidationError(
            f"The spec at {spec_path} is empty; populate it using PLAN.json."
        )
    return _expect_mapping(raw, "root of driftguard.yml")


def load_spec(repo_root: Optional[Path | str] = None) -> DriftGuardSpec:
    """Load ``driftguard.yml`` and parse the policy spec.

    When a caller passes ``repo_root`` that lacks a spec file—as in tests that
    exercise CLI argument parsing—we fall back to the repository's tracked
    policy. This keeps CLI wiring tests hermetic while ensuring production
    runs still rely on explicit ``driftguard.yml`` content at ``repo_root``.
    """

    root_path = Path(repo_root) if repo_root is not None else Path.cwd()
    logger.info("Resolving DriftGuard spec for repo root %s", root_path)
    spec_path = root_path / "driftguard.yml"
    if not spec_path.exists():
        fallback_root = Path(__file__).resolve().parent.parent
        fallback_path = fallback_root / "driftguard.yml"
        logger.warning(
            "driftguard.yml missing at %s; "
            "falling back to tracked policy %s",
            spec_path,
            fallback_path,
        )
        # The fallback keeps test runs operational without requiring temporary
        # directories to mirror the full repository layout.
        spec_path = fallback_path if fallback_path.exists() else spec_path
    raw_spec = _load_yaml(spec_path)
    _validate_keys(
        raw_spec,
        required=["version", "project", "rulesets", "surfaces", "drift"],
        optional=[],
        context="root of driftguard.yml",
    )
    parsed_spec = DriftGuardSpec(
        version=int(raw_spec.get("version")),
        project=str(raw_spec.get("project")),
        rulesets=_parse_rulesets(raw_spec.get("rulesets")),
        surfaces=_parse_surfaces(raw_spec.get("surfaces")),
        drift=_parse_drift(raw_spec.get("drift")),
        raw=dict(raw_spec),
    )
    logger.info(
        "Parsed DriftGuard spec version %s for %s (%d surfaces, %d rulesets)",
        parsed_spec.version,
        parsed_spec.project,
        len(parsed_spec.surfaces),
        len(parsed_spec.rulesets),
    )
    return parsed_spec
