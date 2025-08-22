r"""Parse the *compound* BAO dataset.

This parser is intentionally lightweight and makes no assumptions about the
cosmological model. It simply reads the YAML table located in ``data_dir`` and
returns a :class:`pandas.DataFrame` with the expected columns. A matching
``metadata_*.yml`` file supplies the human readable dataset name and
documentation strings which are attached by
``copernican_lib.data_loaders.load_bao_data``. The compound dataset does
**not** ship with a covariance matrix; uncertainties are therefore treated as
uncorrelated and the engine falls back to a diagonal covariance during the
:math:`\chi^2` evaluation. A ``rs_fiducial_Mpc`` column may appear in some
entries but is retained only for reference—no unit conversion is performed so
that published values are used directly without risking double scaling.
"""

import logging
import os

import pandas as pd
import yaml

from copernican_lib.data_loaders import register_bao_parser

DATA_DIR = os.path.dirname(__file__)


@register_bao_parser(data_dir=DATA_DIR)
def parse_bao_v1(data_dir, **kwargs):
    """Parse a BAO dataset defined in a small YAML table."""
    logger = logging.getLogger()
    data_files = [
        f
        for f in os.listdir(data_dir)
        if f.lower().endswith((".yml", ".yaml")) and not f.startswith("metadata")
    ]
    if not data_files:
        logger.error(f"No BAO data file found in {data_dir}.")
        return None
    filepath = os.path.join(data_dir, sorted(data_files)[0])
    try:
        with open(filepath, "r") as f:
            data_yaml = yaml.safe_load(f)

        df = pd.DataFrame(data_yaml["data_points"])
        required_cols = ["redshift", "observable_type", "value", "error"]
        if not all(col in df.columns for col in required_cols):
            logger.error(
                "BAO data file %s missing one or more required columns: %s",
                filepath,
                required_cols,
            )
            return None

        # Ensure numeric columns are typed correctly. ``rs_fiducial_Mpc`` may
        # be absent or contain ``null`` which converts to ``NaN``.
        for col in ["redshift", "value", "error", "rs_fiducial_Mpc"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=required_cols, inplace=True)
        if df.empty:
            logger.error(f"No valid BAO data points after parsing {filepath}.")
            return None

        # Metadata such as dataset name, citation and notes is loaded by
        # ``load_bao_data`` and attached to the DataFrame after this function
        # returns.
        return df
    except Exception as e:
        logger.error(
            f"Error reading or parsing BAO data file {filepath}: {e}",
            exc_info=True,
        )
        return None
