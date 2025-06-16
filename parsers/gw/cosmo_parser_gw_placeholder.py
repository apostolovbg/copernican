# DEV NOTE (v1.6a): Placeholder parser updated for explicit registration system.
import logging
from scripts.data_loaders import BaseParser, register_parser


def parse_gw_placeholder(filepath, **kwargs):
    """Stub parser that logs a message and returns None."""
    logger = logging.getLogger()
    logger.info(f"GW parser placeholder invoked for {filepath}. Feature not implemented.")
    return None


class GWPlaceholderParser(BaseParser):
    DATA_TYPE = "gw"
    SOURCE_NAME = "placeholder"
    PARSER_NAME = "gw_placeholder_v1"
    FILE_EXTENSIONS = []

    def parse(self, filepath, **kwargs):
        return parse_gw_placeholder(filepath, **kwargs)


register_parser(
    data_type="gw",
    source="placeholder",
    name="GW Placeholder",
    parser=GWPlaceholderParser()
)

