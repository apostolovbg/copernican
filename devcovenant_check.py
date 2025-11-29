#!/usr/bin/env python3
"""
Convenience wrapper to run devcovenant checks.

Usage:
    python devcovenant_check.py              # Normal check
    python devcovenant_check.py --startup    # Startup check (AI uses this)
    python devcovenant_check.py --lint       # Lint mode
    python devcovenant_check.py --fix        # Auto-fix violations
"""

import sys
from pathlib import Path

# Import devcovenant
sys.path.insert(0, str(Path(__file__).parent))

from devcovenant.cli import main

if __name__ == "__main__":
    main()
