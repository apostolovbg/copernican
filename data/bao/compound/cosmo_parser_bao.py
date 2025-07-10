
"""Generic parser for BAO JSON datasets.

This parser looks for a ``*.json`` file containing the ``data_points``
array and optionally reads a ``metadata_*.json`` file for additional
information such as the dataset name and citation.  Any JSON file not
starting with ``metadata`` is treated as the data container so multiple
datasets can coexist in the same folder.
"""

import os
import pandas as pd
import json
import logging

from copernican_lib.data_loaders import register_bao_parser
from copernican_lib.utils import load_metadata_from_dir

DATA_DIR = os.path.dirname(__file__)
META = load_metadata_from_dir(DATA_DIR)

DATASET_NAME = META.get("dataset_name", "BAO dataset")
DESCRIPTION = META.get(
    "description",
    "Compilation of baryon acoustic oscillation measurements.",
)


@register_bao_parser(
    DATASET_NAME,
    DESCRIPTION,
    data_dir=DATA_DIR,
)
def parse_bao_json_v1(data_dir, **kwargs):
    """Parse a BAO dataset and attach metadata."""
    logger = logging.getLogger()
    json_files = [
        f
        for f in os.listdir(data_dir)
        if f.lower().endswith(".json") and not f.startswith("metadata")
    ]
    if not json_files:
        logger.error(f"No BAO JSON file found in {data_dir}.")
        return None
    filepath = os.path.join(data_dir, sorted(json_files)[0])
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

        meta = META

        dataset_long = meta.get('dataset_name', data_json.get('name', f"BAO_{os.path.basename(filepath)}"))
        df.attrs['citation'] = meta.get('citation', data_json.get('citation', 'N/A'))
        df.attrs['notes'] = meta.get('notes', data_json.get('notes', 'N/A'))
        df.attrs['description'] = meta.get('description', '')
        df.attrs['dataset_long_name'] = dataset_long
        df.attrs['dataset_name_attr'] = dataset_long.replace(' ', '_')
        return df
    except Exception as e:
        logger.error(f"Error reading or parsing BAO JSON file {filepath}: {e}", exc_info=True); return None
