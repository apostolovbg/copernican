# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Native CMB contract helpers for engine integration."""

from __future__ import annotations

from .engine_adapter import (
    _SUPPORTED_CMB_CALL_KEYS,
    _SUPPORTED_CMB_CALL_METHODS,
    _SUPPORTED_CMB_CONTRACT_KEYS,
    _SUPPORTED_CMB_GRID_KEYS,
    _SUPPORTED_CMB_GRID_SPACING,
    _SUPPORTED_CMB_PARAMETER_KEYS,
    _SUPPORTED_CMB_PERTURBATION_GAUGES,
    _SUPPORTED_CMB_PERTURBATION_KEYS,
    _SUPPORTED_CMB_VALUE_KEYS,
    _SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES,
    CMBContractEvaluator,
    CMBParameterEvaluator,
    _validate_cmb_contract_definition,
)

__all__ = [
    "CMBContractEvaluator",
    "CMBParameterEvaluator",
    "_SUPPORTED_CMB_CALL_KEYS",
    "_SUPPORTED_CMB_CALL_METHODS",
    "_SUPPORTED_CMB_CONTRACT_KEYS",
    "_SUPPORTED_CMB_GRID_KEYS",
    "_SUPPORTED_CMB_GRID_SPACING",
    "_SUPPORTED_CMB_PARAMETER_KEYS",
    "_SUPPORTED_CMB_PERTURBATION_GAUGES",
    "_SUPPORTED_CMB_PERTURBATION_KEYS",
    "_SUPPORTED_CMB_VALUE_KEYS",
    "_SUPPORTED_PERTURBATION_INDEPENDENT_VARIABLES",
    "_validate_cmb_contract_definition",
]
