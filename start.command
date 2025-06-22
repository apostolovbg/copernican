#!/bin/bash

# This script will launch the Copernican Suite.
# It ensures that it runs from the same directory where the script itself is located.

# Change directory to the script's location. This is crucial for it to find copernican.py.
cd "$(dirname "$0")"

# Run the Python script using the python3 interpreter.
python3 copernican.py

# The terminal window will remain open after the script finishes
# so you can review the output. You can close it manually.
