"""Compatibility layer exposing plugin builders to numerical engines.

**Last Updated:** 2025-10-31

This module now delegates to :mod:`copernican_lib.plugins` and
:mod:`copernican_lib.posterior`. The previous monolithic implementation mixed
plugin assembly, validation and posterior construction which complicated
maintenance and broke multiprocessing picklability. Engines continue to import
``build_plugin`` and ``make_logposterior`` from this module so external APIs do
not break, but the heavy lifting lives in dedicated, thoroughly documented
packages.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .plugins import (
    REQUIRED_ATTRIBUTES,
    REQUIRED_FUNCTIONS,
    EnginePlugin,
    build_engine_plugin,
)
from .plugins import validate_plugin as _validate_plugin
from .posterior import PosteriorEvaluator, make_logposterior


def build_plugin(
    model_data: Mapping[str, Any], func_dict: Mapping[str, Callable]
) -> EnginePlugin:
    """Return an :class:`EnginePlugin` constructed from YAML metadata."""

    plugin = build_engine_plugin(model_data, func_dict)
    _validate_plugin(plugin)
    return plugin


def validate_plugin(plugin: EnginePlugin) -> bool:
    """Validate that ``plugin`` exposes the required interface."""

    return _validate_plugin(plugin)


__all__ = [
    "EnginePlugin",
    "PosteriorEvaluator",
    "REQUIRED_ATTRIBUTES",
    "REQUIRED_FUNCTIONS",
    "build_plugin",
    "make_logposterior",
    "validate_plugin",
]
