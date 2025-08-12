"""Test package for the Copernican Suite."""

import logging

# Configure a simple root logger so tests surface informative messages.  The
# ``force`` flag ensures that duplicate handlers are not added when the test
# suite is executed multiple times.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s:%(name)s:%(message)s",
    force=True,
)

# The presence of this file allows unittest discovery to treat ``tests`` as a
# package.
