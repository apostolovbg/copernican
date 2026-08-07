"""Shared control/test model selection and comparison validation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

DEFAULT_CONTROL_MODEL = "model_lcdm.yml"


@dataclass(frozen=True)
class ModelRole:
    """Identify one selected model independently of its comparison role."""

    name: str
    filename: str = ""
    identifier: str = ""

    def as_manifest(self) -> dict[str, str]:
        """Return the stable role representation stored in a manifest."""

        payload = {"name": self.name}
        if self.filename:
            payload["filename"] = self.filename
        if self.identifier:
            payload["identifier"] = self.identifier
        return payload


@dataclass(frozen=True)
class ComparisonRequest:
    """Represent the control and test models for one comparison run."""

    control_model: ModelRole
    test_model: ModelRole

    def as_manifest(self) -> dict[str, dict[str, str]]:
        """Return the canonical manifest comparison payload."""

        return {
            "control": self.control_model.as_manifest(),
            "test": self.test_model.as_manifest(),
        }

    @property
    def model_names(self) -> tuple[str, str]:
        """Return control and test display names in execution order."""

        return self.control_model.name, self.test_model.name


def model_role_from_value(
    value: str | Mapping[str, Any] | ModelRole,
    *,
    filename: str = "",
) -> ModelRole:
    """Normalize a CLI, GUI, or manifest model value into a role."""

    if isinstance(value, ModelRole):
        return value
    if isinstance(value, Mapping):
        name = str(value.get("name") or value.get("id") or "").strip()
        resolved_filename = str(value.get("filename") or filename).strip()
        identifier = str(value.get("identifier") or "").strip()
    else:
        name = str(value).strip()
        resolved_filename = filename.strip()
        identifier = ""
    if not name:
        raise ValueError("A selected model must have a non-empty name.")
    return ModelRole(
        name=name,
        filename=resolved_filename,
        identifier=identifier,
    )


def build_comparison_request(
    control_model: str | Mapping[str, Any] | ModelRole,
    test_model: str | Mapping[str, Any] | ModelRole,
    *,
    control_filename: str = "",
    test_filename: str = "",
) -> ComparisonRequest:
    """Build and validate a shared control/test comparison request."""

    return ComparisonRequest(
        control_model=model_role_from_value(
            control_model, filename=control_filename
        ),
        test_model=model_role_from_value(test_model, filename=test_filename),
    )


def comparison_from_manifest(manifest: Mapping[str, Any]) -> ComparisonRequest:
    """Read the required canonical comparison from a manifest."""

    selection = manifest.get("selection", {}) or {}
    comparison = selection.get("comparison", {})
    if isinstance(comparison, Mapping):
        control = comparison.get("control")
        test = comparison.get("test")
        if control and test:
            return build_comparison_request(control, test)
    raise ValueError(
        "Run manifests must declare selection.comparison with control and "
        "test model records."
    )


def _surface_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    """Find the first declared comparison surface in model metadata."""

    for key in keys:
        value = metadata.get(key)
        if value is not None:
            return value
    cmb = metadata.get("cmb")
    if isinstance(cmb, Mapping):
        for key in keys:
            value = cmb.get(key)
            if value is not None:
                return value
    return None


def validate_comparison_compatibility(
    request: ComparisonRequest,
    *,
    control_metadata: Mapping[str, Any] | None = None,
    test_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Reject pairs whose declared observable surfaces cannot be compared."""

    if control_metadata is None or test_metadata is None:
        return
    surfaces = {
        "observables": (
            "observables",
            "cmb_observables",
            "spectrum_observables",
        ),
        "units": ("units", "cmb_units", "spectrum_units"),
        "ell_grids": ("ell_grids", "ell_grid", "cmb_ell_grid"),
        "spectrum_roles": (
            "spectrum_roles",
            "cmb_spectrum_roles",
            "spectrum_role",
        ),
    }
    for label, keys in surfaces.items():
        control_value = _surface_value(control_metadata, keys)
        test_value = _surface_value(test_metadata, keys)
        if control_value is None or test_value is None:
            continue
        if _normalise_surface(control_value) != _normalise_surface(test_value):
            raise ValueError(
                f"Control and test models declare incompatible {label}."
            )


def _normalise_surface(value: Any) -> Any:
    """Make nested declaration values deterministic for compatibility tests."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _normalise_surface(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (list, tuple, set)):
        return tuple(_normalise_surface(item) for item in value)
    return value


def comparison_slug(request: ComparisonRequest) -> str:
    """Return a filesystem-safe identity for the selected pair."""

    def _slug(value: str) -> str:
        """Normalize one model identity for use in output filenames."""

        return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "model"

    control = _slug(request.control_model.name)
    test = _slug(request.test_model.name)
    return f"{control}-vs-{test}"


__all__ = [
    "DEFAULT_CONTROL_MODEL",
    "ComparisonRequest",
    "ModelRole",
    "build_comparison_request",
    "comparison_from_manifest",
    "comparison_slug",
    "model_role_from_value",
    "validate_comparison_compatibility",
]
