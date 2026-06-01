# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""CAMB adapter-contract helpers for engine integration.

This module re-exports the validated CAMB background and perturbation
contract helpers so callers that previously imported from
``copernican.lib.camb_contract`` keep seeing the canonical adapter surface.
"""

from __future__ import annotations

from .engine_adapter import (
    _SUPPORTED_CMB_BACKEND,
    _SUPPORTED_CMB_CALL_KEYS,
    _SUPPORTED_CMB_CALL_METHODS,
    _SUPPORTED_CMB_CONTRACT_KEYS,
    _SUPPORTED_CMB_GRID_KEYS,
    _SUPPORTED_CMB_GRID_SPACING,
    _SUPPORTED_CMB_PARAM_KEYS,
    _SUPPORTED_CMB_PERTURBATION_GAUGES,
    _SUPPORTED_CMB_PERTURBATION_KEYS,
    _SUPPORTED_CMB_VALUE_KEYS,
    _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES,
    CMB_BACKEND_CAPABILITIES,
    CAMBContractEvaluator,
    CAMBParameterEvaluator,
    _validate_camb_contract_definition,
)

__all__ = [
    "CMB_BACKEND_CAPABILITIES",
    "CAMBContractEvaluator",
    "CAMBParameterEvaluator",
    "_SUPPORTED_CMB_BACKEND",
    "_SUPPORTED_CMB_CALL_KEYS",
    "_SUPPORTED_CMB_CALL_METHODS",
    "_SUPPORTED_CMB_CONTRACT_KEYS",
    "_SUPPORTED_CMB_GRID_KEYS",
    "_SUPPORTED_CMB_GRID_SPACING",
    "_SUPPORTED_CMB_PARAM_KEYS",
    "_SUPPORTED_CMB_PERTURBATION_GAUGES",
    "_SUPPORTED_CMB_PERTURBATION_KEYS",
    "_SUPPORTED_CMB_VALUE_KEYS",
    "_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES",
    "_validate_camb_contract_definition",
]
