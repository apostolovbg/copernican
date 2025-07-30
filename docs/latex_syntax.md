# LaTeX Syntax Guide

This document describes the supported LaTeX-like syntax for cosmological model YAML files. Expressions are parsed by `latex_utils.py` and converted to NumPy-ready code using mappings from `latex_mappings.yml`.

## Exponentiation
Use `**` for powers:

```yaml
Hz_expression: "H(z) = H_0 * sqrt{\Omega_{m0}*(1+z)**3 + \Omega_{\Lambda0}}"
```

The parser also recognises superscripts such as `x^{2}` and converts them to the same form. Care should be taken to preserve `^` only for superscripts like integration limits or coordinate indices.

## Functions and Symbols
Standard functions (`\log`, `\sin`, `\exp`, etc.) are translated according to the `function_replacements` table in `latex_mappings.yml`. Greek letters and their variants are mapped to ASCII names via `symbol_replacements`. The `unicode_symbols` table provides pretty Unicode equivalents for console output.

## Allowed Macros
The parser strips common spacing and sizing commands (`\left`, `\right`, `\!`, `\,`, etc.). Unsupported macros should be avoided. Fractions `\frac{a}{b}` are converted to `(a)/(b)` automatically.

## Dictionary File
`latex_mappings.yml` groups replacements into four dictionaries:

- `symbol_replacements` – maps LaTeX symbols to safe identifiers.
- `function_replacements` – maps function names to their NumPy or SymPy equivalents.
- `macros_remove` – LaTeX commands that are stripped during sanitisation.
- `unicode_symbols` – conversions used by `latex_to_unicode` for log output.

All Greek letters—upper and lower case—and variants such as `\varphi` are included in these tables in alphabetical order.
Standard constants like `c` and `\hbar` can be loaded from `common_parameters.yml`.
