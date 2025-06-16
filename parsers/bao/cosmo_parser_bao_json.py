# DEV NOTE (v1.5e): General BAO JSON parser separated for modular discovery.
# DEV NOTE (v1.5f hotfix): Updated import path for ``data_loaders`` package.

import pandas as pd
import json
import os
import logging

from scripts.data_loaders import register_bao_parser


@register_bao_parser("bao_json_general_v1", "General BAO JSON format (e.g., bao1.json).")
def parse_bao_json_v1(filepath, **kwargs):
    """Parses a generic BAO JSON file into a standard DataFrame."""
    logger = logging.getLogger()
    try:
        with open(filepath, 'r') as f:
            data_json = json.load(f)

        df = pd.DataFrame(data_json['data_points'])
        required_cols = ['redshift', 'observable_type', 'value', 'error']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"BAO JSON file {filepath} missing one or more required columns: {required_cols}"); return None

        for col in ['redshift', 'value', 'error']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df.dropna(subset=required_cols, inplace=True)
        if df.empty:
            logger.error(f"No valid BAO data points after parsing {filepath}."); return None

        df.attrs['citation'] = data_json.get('citation', 'N/A')
        df.attrs['notes'] = data_json.get('notes', 'N/A')
        df.attrs['dataset_name_attr'] = data_json.get('name', f"BAO_{os.path.basename(filepath)}")
        return df
    except Exception as e:
        logger.error(f"Error reading or parsing BAO JSON file {filepath}: {e}", exc_info=True); return None
