# Copernican Suite JS

This directory contains an experimental WebAssembly build powered by
[Pyodide](https://pyodide.org/). Open `index.html` in a browser to test a
simplified interface. Select a `cosmo_model_*.json` file and click **Run** to
load it. The Python code runs entirely in your browser and prints a brief
message.

Only a subset of the full Copernican Suite is available in this demo; heavy
packages such as CAMB are not compiled yet. Results are therefore limited, but
this proves the concept of a web-based CLI.
