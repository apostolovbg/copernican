"""Central LaTeX translation utilities for the Copernican Suite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict

import yaml

# Load replacement dictionaries from ``latex_mappings.yml`` once at import.
# YAML is more readable than JSON and avoids backslash escaping issues.
_mapping_path = Path(__file__).with_name("latex_mappings.yml")
try:
    with _mapping_path.open("r") as _fh:
        _MAPPINGS: Dict[str, Dict[str, str]] = yaml.safe_load(_fh)
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
    expr = re.sub(r"^\$+|\$+$", "", expr)
    if "=" in expr:
        expr = expr.split("=", 1)[1]

    for pat in _MACROS_REMOVE:
        pattern = pat if "\\s" in pat else re.escape(pat)
        expr = re.sub(pattern, "", expr)
    # ``\\rm`` occasionally survives the initial cleanup when loaded from YAML.
    # Remove it explicitly so parameters like ``\Omega_{\rm eff}`` parse
    # correctly into ``Omega_eff`` instead of ``Omega_rm eff``.
    expr = re.sub(r"\\rm\s*", "", expr)
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


# Unicode translation table used by :func:`latex_to_unicode` for prettier
# console output. The mappings are stored in the YAML file so new symbols
# can be added without modifying this module.
_UNICODE_SYMBOLS = _MAPPINGS.get("unicode_symbols", {})

"""Tables of subscript and superscript characters used by
``latex_to_unicode``."""

# Only a subset of characters have dedicated Unicode glyphs.  The base tables
# below list those known substitutions.  During map generation we iterate over
# the full set of required characters—Latin and Greek letters, digits and basic
# math operators—falling back to the original character whenever no specialised
# glyph exists.  Ordering follows the user expectation: Latin uppercases, Latin
# lowercases, Greek uppercases, Greek lowercases, digits, parentheses,
# brackets/braces and common operators.

_SUBSCRIPT_BASE = {
    # Digits
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    # Basic operators and grouping
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    # Letters with dedicated subscript forms
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
    # Greek letters with subscript variants
    "β": "ᵦ",
    "γ": "ᵧ",
    "ρ": "ᵨ",
    "φ": "ᵩ",
    "χ": "ᵪ",
}

_SUPERSCRIPT_BASE = {
    # Digits
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    # Basic operators and grouping
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "*": "⁎",
    "/": "⁄",
    # Latin lowercase letters
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ᶦ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "q": "𐞥",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    # Latin uppercase letters
    "A": "ᴬ",
    "B": "ᴮ",
    "C": "ᶜ",
    "D": "ᴰ",
    "E": "ᴱ",
    "F": "ᶠ",
    "G": "ᴳ",
    "H": "ᴴ",
    "I": "ᴵ",
    "J": "ᴶ",
    "K": "ᴷ",
    "L": "ᴸ",
    "M": "ᴹ",
    "N": "ᴺ",
    "O": "ᴼ",
    "P": "ᴾ",
    "R": "ᴿ",
    "T": "ᵀ",
    "U": "ᵁ",
    "V": "ⱽ",
    "W": "ᵂ",
    # Greek letters with superscript variants
    "α": "ᵅ",
    "β": "ᵝ",
    "γ": "ᵞ",
    "δ": "ᵟ",
    "ε": "ᵋ",
    "θ": "ᶿ",
    "ι": "ᶥ",
    "φ": "ᵠ",
    "χ": "ᵡ",
    "ρ": "ᵨ",
}


def _build_script_maps() -> tuple[Dict[str, str], Dict[str, str]]:
    """Generate full sub/superscript lookup tables.

    Characters without specialised Unicode variants map to themselves.  The
    insertion order matches the alphabetical specification required by the
    documentation: Latin uppercases, Latin lowercases, Greek uppercases, Greek
    lowercases, digits, parentheses, brackets, braces and operators.
    """

    sub_map: Dict[str, str] = {}
    sup_map: Dict[str, str] = {}

    latin_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    latin_lower = "abcdefghijklmnopqrstuvwxyz"
    greek_upper = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    greek_lower = "αβγδεζηθικλμνξοπρστυφχψω"
    digits = "0123456789"

    for ch in latin_upper:
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch.lower(), ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(
            ch,
            _SUPERSCRIPT_BASE.get(ch.lower(), ch),
        )
    for ch in latin_lower:
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in greek_upper:
        lower = ch.lower()
        sub_map[ch] = _SUBSCRIPT_BASE.get(lower, _SUBSCRIPT_BASE.get(ch, ch))
        sup_map[ch] = _SUPERSCRIPT_BASE.get(
            ch,
            _SUPERSCRIPT_BASE.get(lower, ch),
        )
    for ch in greek_lower:
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in digits:
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in "()":
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in "[]":
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in "{}":
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    for ch in "+-*/=":
        sub_map[ch] = _SUBSCRIPT_BASE.get(ch, ch)
        sup_map[ch] = _SUPERSCRIPT_BASE.get(ch, ch)
    return sub_map, sup_map


_SUB_MAP, _SUP_MAP = _build_script_maps()


def latex_to_unicode(text: str) -> str:
    r"""Return ``text`` converted to basic Unicode math symbols."""
    cleaned = re.sub(r"^\$+|\$+$", "", str(text).strip())
    for pat, repl in _UNICODE_SYMBOLS.items():
        cleaned = re.sub(re.escape(pat), repl, cleaned)

    def _sub_repl(match: re.Match[str]) -> str:
        """Translate a ``_{} `` group into Unicode subscripts."""
        inner = match.group(1)
        inner = re.sub(r"\\rm\s*", "", inner)
        inner = inner.replace(" ", "")
        if inner in _UNICODE_SYMBOLS:
            inner_char = _UNICODE_SYMBOLS[inner]
            return _SUB_MAP.get(inner_char, inner_char)
        return "".join(_SUB_MAP.get(ch, ch) for ch in inner)

    def _sup_repl(match: re.Match[str]) -> str:
        """Translate a ``^{} `` group into Unicode superscripts."""
        inner = match.group(1)
        inner = inner.replace(" ", "")
        return "".join(_SUP_MAP.get(ch, ch) for ch in inner)

    cleaned = re.sub(r"_\{([^{}]+)\}", _sub_repl, cleaned)
    cleaned = re.sub(r"\^\{([^{}]+)\}", _sup_repl, cleaned)
    cleaned = re.sub(
        r"_([A-Za-z0-9])",
        lambda m: _SUB_MAP.get(m.group(1), m.group(1)),
        cleaned,
    )
    cleaned = re.sub(
        r"\^([A-Za-z0-9+-])",
        lambda m: _SUP_MAP.get(m.group(1), m.group(1)),
        cleaned,
    )
    return cleaned
