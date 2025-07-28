"""Central LaTeX translation utilities for the Copernican Suite."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict


# Load replacement dictionaries from ``latex_mappings.json`` once at import.
# Keeping the mappings in a JSON file allows contributors to extend the
# supported LaTeX commands without touching the code.
_mapping_path = Path(__file__).with_name("latex_mappings.json")
try:
    with _mapping_path.open("r") as _fh:
        _MAPPINGS: Dict[str, Dict[str, str]] = json.load(_fh)
except OSError as exc:  # pragma: no cover - only fails if repo is corrupted
    raise RuntimeError(f"Cannot read LaTeX mappings: {_mapping_path}") from exc

_SYMBOLS = _MAPPINGS.get("symbol_replacements", {})
_FUNCTIONS = _MAPPINGS.get("function_replacements", {})
_MACROS_REMOVE = _MAPPINGS.get("macros_remove", [])


def sanitize_name(latex: str) -> str:
    r"""Return a safe Python identifier derived from ``latex``."""
    text = str(latex)
    text = re.sub(r"^\$+|\$+$", "", text)
    for pat, repl in _SYMBOLS.items():
        # Escape patterns to treat them as literal LaTeX commands.
        text = re.sub(re.escape(pat), repl, text)
    # Drop any remaining LaTeX commands such as ``\mathrm``.
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("-", "_")
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", text)
    text = re.sub(r"__+", "_", text).strip("_")
    if not re.match(r"[A-Za-z_]", text):
        text = f"pyvar_{text}" if text else "pyvar"
    return text


def latex_to_sympy(expr: str) -> str:
    r"""Convert a LaTeX expression to a SymPy-friendly string."""
    expr = expr.strip()
    if expr.startswith("$$") and expr.endswith("$$"):
        expr = expr[2:-2]
    if "=" in expr:
        expr = expr.split("=", 1)[1]

    for pat in _MACROS_REMOVE:
        pattern = pat if "\\s" in pat else re.escape(pat)
        expr = re.sub(pattern, "", expr)
    for pat, repl in _SYMBOLS.items():
        expr = re.sub(re.escape(pat), repl, expr)
    for pat, repl in _FUNCTIONS.items():
        expr = re.sub(re.escape(pat), repl, expr)

    while "\\frac" in expr:
        expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)

    expr = re.sub(r"_{([^{}]+)}", r"_\1", expr)
    expr = re.sub(r"\^\{([^{}]+)\}", r"**(\1)", expr)
    expr = re.sub(r"\^([\w\.]+)", r"**\1", expr)
    expr = expr.replace("\\", "")
    expr = expr.replace("{", "(").replace("}", ")")
    expr = expr.replace("[", "(").replace("]", ")")
    expr = re.sub(r"\s{2,}", " ", expr)
    return expr.strip()


def wrap_math(text: str) -> str:
    r"""Return ``text`` wrapped in ``$`` for Matplotlib math rendering."""
    if text is None:
        return ""
    cleaned = re.sub(r"^\$+|\$+$", "", str(text).strip())
    for pat in _MACROS_REMOVE:
        pattern = pat if "\\s" in pat else re.escape(pat)
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = re.sub(r"\\!", "", cleaned)
    cleaned = re.sub(r"\\,", "", cleaned)
    cleaned = re.sub(r"\\rm\s*", "", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    return f"${cleaned}$" if cleaned else ""
