# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Placeholder parser for future gravitational wave datasets."""

import logging
import os

from copernican_lib.data_loaders import register_gw_parser

DATA_DIR = os.path.dirname(__file__)


@register_gw_parser(data_dir=DATA_DIR)
def parse_gw_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    # Gravitational wave support is under development. This stub allows
    # the rest of the framework to run even though no real data is parsed yet.
    logger = logging.getLogger()
    logger.info(
        "GW parser placeholder invoked in %s. " "Feature not implemented.",
        data_dir,
    )
    return None
