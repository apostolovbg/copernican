# Packaging Guide

Standalone executables are no longer produced. The suite runs directly
from source using platform launchers:

- `start.bat` on Windows
- `start.command` on macOS
- `start.sh` on Linux

Each script creates or reuses a local `.venv`, upgrades `pip` and installs
all required packages automatically. Only a system-wide Python 3.11+
installation is needed. Running `copernican.py` outside the virtual
environment prompts you to relaunch with the appropriate script.

To install the suite as a package, run:

```bash
pip install .
```

Use `pip install -e .` for development.

