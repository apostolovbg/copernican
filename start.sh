#!/bin/bash
# Launch the Copernican Suite on Linux
cd "$(dirname "$0")"
exec python3 copernican.py "$@"
