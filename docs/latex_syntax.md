# LaTeX Syntax Guide
**Last Updated:** 2025-10-30

This document describes the supported LaTeX-like syntax for cosmological model
YAML files. Expressions are parsed by `latex_utils.py` and converted to NumPy-
ready code using mappings from `latex_mappings.yml`.

`copernican_lib.statistics` now imports `latex_utils.latex_to_unicode` when it
logs acceptance fractions and fitted parameters so that diagnostics produced by
the MCMC engine display the same typographic quality as optimiser summaries.
Extending the mapping tables therefore benefits both plotting and log output.
The same helper converts ``χ²_Total`` when Stage 5 summaries print the SNe-only
chi-squared totals, keeping the glyph consistent with the BAO and CMB entries.

## YAML hygiene tips

* Prefer block scalars (`|` or `>`) when writing long expressions. Quoted
  strings interpret escape sequences, so `\beta` becomes a literal backspace.
  The updated non-ΛCDM models illustrate the folded-style approach.
* Set explicit `python_var` values when you want concise identifiers in the
  generated callables. This avoids auto-generated names such as `Omega_m_eff`
  when the expression simply needs `Omega_eff`.

## Exponentiation
Use `**` for powers:

```yaml
Hz_expression: "H(z) = H_0 * sqrt{\Omega_{m0}*(1+z)**3 + \Omega_{\Lambda0}}"
```

The parser also recognises superscripts such as `x^{2}` and converts them to
the same form as standard exponentiation notation (superscript) for display
purposes, but the right math operator is `**`. Care should be taken to
preserve
`^` only for superscripts like integration limits or coordinate indices.

## Functions and Symbols
Standard functions (`\log`, `\sin`, `\exp`, etc.) are translated according to
the `function_replacements` table in `latex_mappings.yml`. Greek letters and
their variants are mapped to ASCII names via `symbol_replacements`. The
`unicode_symbols` table provides pretty Unicode equivalents for console
output.

## Allowed Macros
The parser strips common spacing and sizing commands (`\left`, `\right`, `\!`,
`\,`, etc.). Unsupported macros should be avoided. Fractions `\frac{a}{b}` are
converted to `(a)/(b)` automatically.

## Dictionary File
`latex_mappings.yml` groups replacements into four dictionaries:

- `symbol_replacements` – maps LaTeX symbols to safe identifiers.
- `function_replacements` – maps function names to their NumPy or SymPy
  equivalents.
- `macros_remove` – LaTeX commands that are stripped during sanitisation.
- `unicode_symbols` – conversions used by `latex_to_unicode` for log output.

All Greek letters—upper and lower case—and variants such as `\varphi` are
included in these tables in alphabetical order.

### Subscripts and Superscripts

`latex_utils.py` now ships with exhaustive lookup tables covering every Latin
and Greek letter in both cases, digits and common math operators. Characters
without dedicated Unicode glyphs fall back to their original form.

## Tips for Writing Equations

- Always enclose multi-character subscripts or superscripts in braces, e.g.
  `H_{\rm 0}` or `x^{(1+z)}`. This ensures the parser interprets the entire
  group correctly.
- Use raw strings in YAML where possible to avoid accidental escape sequences.
- Avoid vendor-specific macros; if a symbol is missing from
  `latex_mappings.yml`, consider extending the mapping rather than embedding
  raw Unicode characters in the YAML file.

## Debugging Syntax Errors

If `copernican_lib.model_parser` reports a parsing failure, inspect the
generated `models/cache/` entry to see the sanitised LaTeX.  Running the
expression through `latex_utils.latex_to_unicode` can also help spot stray
characters that were not translated.  When in doubt, reduce the equation to a
minimal form and reintroduce terms gradually.
