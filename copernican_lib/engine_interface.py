"""Compatibility layer exposing plugin builders to numerical engines.

**Last Updated:** 2025-11-01

The interface now validates CAMB parameter mappings declared in YAML models so
neutrino sector options remain consistent with the helpers in
``copernican_lib.likelihoods.cmb``.  Allow-listing the supported keys prevents
mislabelled photon densities (the old ``omnuh2`` issue) and catches
incompatible combinations such as specifying both ``sum_mnu`` and individual
mass entries.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from .plugins import (
    REQUIRED_ATTRIBUTES,
    REQUIRED_FUNCTIONS,
    EnginePlugin,
    build_engine_plugin,
)
from .plugins import validate_plugin as _validate_plugin
from .posterior import PosteriorEvaluator, make_logposterior


_CMB_PARAM_ALLOWED_KEYS = {
    "H0",
    "ombh2",
    "omch2",
    "omnuh2",
    "omk",
    "tau",
    "As",
    "ns",
    "nrun",
    "nrunrun",
    "r",
    "Alens",
    "Neff",
    "standard_neutrino_neff",
    "num_massive_neutrinos",
    "mnu",
    "sum_mnu",
    "AccuracyBoost",
    "lAccuracyBoost",
    "kAccuracyBoost",
    "theta_H0_range",
    "YHe",
    "neutrino_hierarchy",
}
_MNU_PATTERN = re.compile(r"^mnu(\d+)$")


def _validate_cmb_param_map(plugin: EnginePlugin) -> None:
    """Ensure the plugin's CAMB parameter map only exposes supported keys."""

    if not getattr(plugin, "valid_for_cmb", True):
        return
    param_map = getattr(plugin, "CMB_PARAM_MAP", {}) or {}
    if not isinstance(param_map, Mapping):
        raise ValueError("CMB_PARAM_MAP must be a mapping of CAMB parameters")

    keys = set(str(k) for k in param_map.keys())
    dynamic_mass_keys = {
        key for key in keys if _MNU_PATTERN.match(key) is not None
    }
    invalid = keys - _CMB_PARAM_ALLOWED_KEYS - dynamic_mass_keys
    if invalid:
        invalid_str = ", ".join(sorted(invalid))
        raise ValueError(
            f"Unsupported CAMB parameter(s) in cmb.param_map: {invalid_str}"
        )

    conflicts = []
    if "mnu" in keys and "sum_mnu" in keys:
        conflicts.append("'mnu' and 'sum_mnu' are mutually exclusive")
    if dynamic_mass_keys and ("mnu" in keys or "sum_mnu" in keys):
        conflicts.append(
            "individual 'mnuN' entries cannot be combined with 'mnu' or 'sum_mnu'"
        )
    if conflicts:
        raise ValueError("; ".join(conflicts))


def build_plugin(
    model_data: Mapping[str, Any], func_dict: Mapping[str, Callable]
) -> EnginePlugin:
    """Return an :class:`EnginePlugin` constructed from YAML metadata."""

    plugin = build_engine_plugin(model_data, func_dict)
    _validate_plugin(plugin)
    _validate_cmb_param_map(plugin)
    return plugin


def validate_plugin(plugin: EnginePlugin) -> bool:
    """Validate that ``plugin`` exposes the required interface."""

    result = _validate_plugin(plugin)
    _validate_cmb_param_map(plugin)
    return result


__all__ = [
    "EnginePlugin",
    "PosteriorEvaluator",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_FUNCTIONS",
    "build_plugin",
    "make_logposterior",
    "validate_plugin",
]
