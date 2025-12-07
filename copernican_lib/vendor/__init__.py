"""Vendored third-party helpers used by the Copernican Suite."""

import os
import sys

VENDOR_ROOT = os.path.abspath(os.path.dirname(__file__))
if VENDOR_ROOT not in sys.path:
    sys.path.insert(0, VENDOR_ROOT)
