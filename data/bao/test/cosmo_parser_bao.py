
"""Generic parser for BAO datasets stored in JSON or YAML files.

The parser searches ``data_dir`` for a file ending in ``.json`` or ``.yml``
that contains a ``data_points`` array. A companion ``metadata_*`` file is
read via :func:`copernican_lib.utils.load_metadata_from_dir` to obtain the
dataset name and citation. Any data file not starting with ``metadata`` is
treated as the measurement table so multiple datasets can coexist in the same
folder.
"""

import os
import pandas as pd
import yaml
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
    data_files = [
        f
        for f in os.listdir(data_dir)
        if f.lower().endswith((".json", ".yml", ".yaml"))
        and not f.startswith("metadata")
    ]
    if not data_files:
        logger.error(f"No BAO data file found in {data_dir}.")
        return None
    filepath = os.path.join(data_dir, sorted(data_files)[0])
    try:
        with open(filepath, 'r') as f:
            data_json = yaml.safe_load(f)

        df = pd.DataFrame(data_json['data_points'])
        required_cols = ['redshift', 'observable_type', 'value', 'error']
        if not all(col in df.columns for col in required_cols):
            logger.error(
                f"BAO data file {filepath} missing one or more required columns: {required_cols}"
            );
            return None

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
        logger.error(
            f"Error reading or parsing BAO data file {filepath}: {e}",
            exc_info=True,
        );
        return None
