"""Command-line entry point that delegates to the Copernican launcher."""

import sys

from copernican import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
