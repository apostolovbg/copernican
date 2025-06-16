# DEV NOTE (v1.5f): Placeholder parser for future CMB data formats.
# DEV NOTE (v1.5f hotfix): Updated import path for ``data_loaders`` module.
import logging
from scripts.data_loaders import register_cmb_parser

@register_cmb_parser("cmb_placeholder_v1", "Placeholder CMB parser.")
def parse_cmb_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"CMB parser placeholder invoked for {filepath}. Feature not implemented.")
    return None
