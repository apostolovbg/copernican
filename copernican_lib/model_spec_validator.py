# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

# Rationale: Model specifications are validated here because keeping schema
# checks and cache writes together avoids engines loading malformed equations
# during distributed runs.
"""Validate and sanitise Copernican Suite YAML model specifications.

This module performs schema validation, normalises LaTeX-heavy fields and
writes a cleaned cache file used by child processes. The behaviour evolved
beyond simple parsing, so the name now reflects its responsibility for
validation, sanitisation and cache management rather than mere text parsing.
"""

# This module validates model definition files against a JSON schema and writes
# a sanitized copy to ``models/cache/``. The sanitized file is used by child
# processes so that validation only happens once in the main process.

import math
import multiprocessing as _mp
from pathlib import Path

import yaml
from jsonschema import ValidationError, validate

from . import error_handler, latex_utils, priors


def _sanitise_name_to_var(name: str) -> str:
    """Return a valid Python identifier derived from a LaTeX name."""
    return latex_utils.sanitize_name(name)


def _ensure_delim(expr: str | None) -> str | None:
    """Wrap math expressions with ``$$`` when missing.

    Existing ``$...$`` or ``$$...$$`` delimiters are preserved so authors may
    choose their preferred style.
    """
    if expr is None:
        return None
    cleaned = str(expr).strip()
    if cleaned.startswith("$") and cleaned.endswith("$"):
        return cleaned
    cleaned = f"$${cleaned}$$"
    return cleaned


MODEL_SCHEMA = {
    "type": "object",
    "required": ["model_name", "version", "parameters", "equations"],
    "properties": {
        "model_name": {"type": "string"},
        "version": {"type": "string"},
        "parameters": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "bounds"],
                "properties": {
                    "name": {"type": "string"},
                    "python_var": {"type": "string"},
                    "bounds": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "number"},
                    },
                    "unit": {"type": "string"},
                    "latex_name": {"type": "string"},
                    "transform": {"type": "string"},
                    "prior": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {"type": "string"},
                            "mean": {"type": "number"},
                            "sigma": {"type": "number"},
                            "lower": {"type": "number"},
                            "upper": {"type": "number"},
                            "value": {"type": "number"},
                            "transform": {"type": "string"},
                        },
                    },
                },
            },
        },
        "equations": {"type": "object"},
        "rs_expression": {"type": "string"},
        "cmb": {"type": "object"},
        "gravitational_waves": {"type": "object"},
        "predicts_bao": {"type": "boolean"},
        # Optional human-readable fields used by upcoming UI modules
        "abstract": {"type": "string"},
        "description": {"type": "string"},
        "notes": {"type": "string"},
    },
}


def validate_and_cache_model(path, cache_dir):
    """Validate ``path`` and write cleaned YAML to ``cache_dir``.

    Validation is performed only in the main process. Worker processes simply
    read the sanitized file produced during program startup.

    Parameters
    ----------
    path : str or Path
        Source YAML model file.
    cache_dir : str or Path
        Directory where the sanitized model will be stored.

    Returns
    -------
    str
        Path to the sanitized cache file.
    """
    path = Path(path)
    try:
        with path.open("r") as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        error_handler.report_error(f"Failed to read model YAML '{path}': {e}")
        raise

    # Only validate in the main process to avoid random failures when
    # worker processes import this module under multiprocessing. The
    # sanitized file produced here is shared by child processes, so
    # repeated validation is unnecessary.
    if _mp.current_process().name == "MainProcess":
        try:
            validate(instance=data, schema=MODEL_SCHEMA)
        except ValidationError as e:
            msg = f"Model YAML validation error: {e.message}"
            error_handler.report_error(msg)
            raise ValueError(msg) from e

    # Auto-generate missing python_var fields from LaTeX names
    used_vars = {
        param.get("python_var")
        for param in data.get("parameters", [])
        if param.get("python_var")
    }
    for param in data.get("parameters", []):
        if "latex_name" not in param:
            raise ValueError("Missing required latex_name for parameter")
        if not param.get("python_var"):
            base = _sanitise_name_to_var(param["latex_name"])
            candidate = base
            idx = 2
            while candidate in used_vars:
                candidate = f"{base}_{idx}"
                idx += 1
            param["python_var"] = candidate
            used_vars.add(candidate)
        bounds = param.get("bounds", [])
        if (
            isinstance(bounds, list)
            and len(bounds) == 2
            and math.isclose(
                bounds[0], bounds[1], rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            fixed_value = float(bounds[0])
            prior = param.get("prior") or {
                "type": "fixed",
                "value": fixed_value,
            }
            if not isinstance(prior, dict):
                raise ValueError("Fixed parameter priors must be mappings")
            prior_map = dict(prior)
            try:
                priors.normalise_prior_mapping(prior_map)
            except priors.PriorError as exc:
                raise ValueError(str(exc)) from exc
            if prior_map.get("type") != "fixed":
                raise ValueError("Fixed parameters must declare a fixed prior")
            if not math.isclose(
                float(prior_map.get("value", float("nan"))),
                fixed_value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "Fixed prior value must match the declared bounds"
                )
            param["prior"] = prior_map
            param.pop("transform", None)
            continue
        prior = param.get("prior")
        if prior:
            if not isinstance(prior, dict):
                raise ValueError("Prior definitions must be mappings")
            prior_map = dict(prior)
            try:
                priors.normalise_prior_mapping(prior_map)
            except priors.PriorError as exc:
                raise ValueError(str(exc)) from exc
            if prior_map.get("type") == "fixed":
                if not (
                    isinstance(bounds, list)
                    and len(bounds) == 2
                    and math.isclose(
                        bounds[0],
                        bounds[1],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        "Fixed priors require identical parameter bounds"
                    )
            param["prior"] = prior_map
            transform = priors.transform_from_mapping(prior_map)
            if transform is not None:
                param["transform"] = prior_map.get("transform", "log")
            else:
                param.pop("transform", None)
        elif param.get("transform"):
            transform_name = param["transform"]
            if transform_name != "identity":
                raise ValueError(
                    "Transforms require a prior declaration to anchor them"
                )
            param.pop("transform", None)

    # Ensure mathematical fields are wrapped with '$$' for downstream tools
    data["Hz_expression"] = _ensure_delim(data.get("Hz_expression"))
    data["rs_expression"] = _ensure_delim(data.get("rs_expression"))
    eq_sections = data.get("equations", {})
    for key, arr in eq_sections.items():
        if isinstance(arr, list):
            eq_sections[key] = [_ensure_delim(e) for e in arr]

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"cache_{path.name}"
    with cache_path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return str(cache_path)
