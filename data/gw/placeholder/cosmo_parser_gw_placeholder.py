# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Placeholder parser for future gravitational-wave standard siren datasets."""

import logging
import os

from copernican_lib.data_loaders import register_gw_parser

DATA_DIR = os.path.dirname(__file__)


@register_gw_parser(data_dir=DATA_DIR)
def parse_gw_placeholder(data_dir, **kwargs):
    """Stub parser that logs a message and returns None."""
    # Gravitational-wave standard siren support remains under development.
    # The stub keeps the framework running until real data arrives.
    logger = logging.getLogger()
    logger.info(
        "GW placeholder invoked in %s as part of placeholder management. "
        "Feature not implemented yet.",
        data_dir,
    )
    return None
