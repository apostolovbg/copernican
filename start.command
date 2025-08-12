#!/bin/bash

# This script will launch the Copernican Suite.
# Ensure the script runs from its own directory so copernican.py is found.
cd "$(dirname "$0")"

# Run the Python script using the python3 interpreter.
python3 copernican.py "$@"

# The terminal window will remain open after the script finishes
# so you can review the output. You can close it manually.
