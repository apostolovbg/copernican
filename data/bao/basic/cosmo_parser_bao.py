
import os
import pandas as pd
import json
import logging

from scripts.data_loaders import register_bao_parser


@register_bao_parser("Basic BAO testing dataset", "", data_dir=os.path.dirname(__file__))
def parse_bao_json_v1(data_dir, **kwargs):
    """Parses a generic BAO JSON file into a standard DataFrame."""
    logger = logging.getLogger()
    filepath = os.path.join(data_dir, "bao1.json")
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
