
"""Parse the *compound* BAO dataset and attach metadata.

This parser is intentionally lightweight and makes no assumptions about the
cosmological model.  It simply reads the YAML table located in ``data_dir``
and returns a :class:`pandas.DataFrame` with the expected columns.  A matching
``metadata_*.yml`` file supplies the human readable dataset name and
documentation strings.  The compound dataset does **not** ship with a
covariance matrix; uncertainties are therefore treated as uncorrelated and the
engine falls back to a diagonal covariance during the :math:`\chi^2`
evaluation.  When a fiducial sound horizon ``rs_fiducial_Mpc`` is provided the
values are converted to true ``*_over_rs`` observables so the engine remains
agnostic of any fiducial scaling.
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
def parse_bao_v1(data_dir, **kwargs):
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

        # Ensure numeric columns are typed correctly.  ``rs_fiducial_Mpc`` may
        # be absent or contain ``null`` which converts to ``NaN``.
        for col in ['redshift', 'value', 'error', 'rs_fiducial_Mpc']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Some literature quotes distances multiplied by ``r_s^fid / r_s``. If
        # a fiducial sound horizon is specified we divide the measurement and
        # its uncertainty by that value to recover the dimensionless
        # ``*_over_rs`` form expected by the engine.  Entries without a fiducial
        # value are assumed to already be in the correct units.
        if 'rs_fiducial_Mpc' in df.columns:
            mask = df['rs_fiducial_Mpc'].notna()
            df.loc[mask, 'value'] /= df.loc[mask, 'rs_fiducial_Mpc']
            df.loc[mask, 'error'] /= df.loc[mask, 'rs_fiducial_Mpc']

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
