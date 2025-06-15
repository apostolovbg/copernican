# DEV NOTE (v1.5f): Placeholder parser for future gravitational wave data formats.
import logging
from data_loaders import register_gw_parser

@register_gw_parser("gw_placeholder_v1", "Placeholder GW parser.")
def parse_gw_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"GW parser placeholder invoked for {filepath}. Feature not implemented.")
    return None
