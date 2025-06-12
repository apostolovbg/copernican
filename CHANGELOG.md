# Copernican Suite Change Log

This file summarizes notable changes across versions. Refer to `doc.json` for more technical details.

## v1.4g
- UniStra data loaders restored to the fixed-width strategy from v1.3.
- Added `cosmo_engine_1.4g.py` fixing the residual calculation bug and working with all standard model plugins.

## v1.4rc13
- Development refocused on reproducing the v1.3 parsing logic.
- Confirmed that the previous data loss came from incorrect column selection.

## v1.4rc12
- Final failed attempt under the old strategy. Still only loaded 33 supernovae.

## v1.4rc2 - v1.4rc11
- Multiple experiments to fix the pipeline resulted in cascading errors including `KeyError`, `ValueError`, `TypeError`, and improper data filtering.

## v1.4rc (Initial)
- Major refactor that broke the data pipeline.

## v1.3
- Stable reference version with correct data loaders and basic plotting.

