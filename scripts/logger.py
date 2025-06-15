"""Logging configuration for the Copernican Suite."""
# DEV NOTE (v1.5c): Initial logging utility carried over from Phase 0.
import logging
import os

DEFAULT_LOG_FILE = os.path.join('output', 'copernican.log')

def init_logging(log_file=DEFAULT_LOG_FILE, level=logging.INFO):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(filename=log_file, level=level,
                        format='%(asctime)s %(levelname)s %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler())
