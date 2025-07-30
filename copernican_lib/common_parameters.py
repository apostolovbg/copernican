"""Shared mathematical and physical constants for Copernican models."""

from __future__ import annotations

import yaml
from pathlib import Path
from typing import List, Dict, Any

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

from . import latex_utils

_PATH = Path(__file__).with_name("common_parameters.yml")
try:
    with _PATH.open("r") as _fh:
        _DATA: Dict[str, Any] = yaml.safe_load(_fh)
except OSError as exc:  # pragma: no cover - repo corruption
    raise RuntimeError(f"Cannot read common parameters: {_PATH}") from exc

PARAMETERS: List[Dict[str, Any]] = _DATA.get("parameters", [])

_TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
for param in PARAMETERS:
    val = param.get("value")
    if isinstance(val, str):
        try:
            sym = latex_utils.latex_to_sympy(val)
            param["value"] = float(parse_expr(sym, transformations=_TRANSFORMS))
        except Exception:  # pragma: no cover - should rarely fail
            pass

__all__ = ["PARAMETERS"]
