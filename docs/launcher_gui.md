# Launchers and GUI

The three launcher scripts (`start.sh`, `start.command` and `start.bat`) all
share the same goal: download a private Python 3.11 interpreter, build the
managed virtual environment, install pinned requirements and start either the
CLI or the GUI through the shared `copernican.py` entry point. The GUI option is
now more transparent: selecting it prints a short status line, sets
`COPERNICAN_DETACH_GUI=1` and calls `python copernican.py --gui` without
obscuring any output behind `nohup`, `start /b` or redirections. This lets the
launcher log the orchestration services while `copernican.py` spawns the
detached GUI process, honours the `pythonw` preference on Windows and returns
as soon as the handoff completes.

## How the bootstrapping works

1. A launcher ensures `.python` contains a Python 3.11 build whose bundled
   `python` binary passes the `python_in_311_series` check. Anything outside
   that window is deleted so upgrades never drift away from the supported
   interpreter.
2. The script creates `.venv`, activates it and installs the pinned
   dependencies from `requirements.lock`. The environment is rebuilt whenever
   the interpreter changes so the GUI inherits the same deterministic runtime
   as the CLI.
3. Once the launcher detects it is running inside `.venv`, it presents the
   menu. Choosing the GUI option now prints a short notice, sets the strict
   warning flag, and relies entirely on `copernican.py --gui` to detach the
   visual interface and log the shared services.

## Detach strategy

`copernican.py --gui` still detaches automatically when `COPERNICAN_DETACH_GUI`
is set. The new behaviour keeps the terminal focused on the orchestration
notification, allowing the detached GUI to start in the background without
closing the terminal silently. If `pythonw` exists (on Windows) or `pythonw`
variants are available on the current platform, the GUI uses them so the
console remains clean. When Tkinter is unavailable, the GUI gracefully falls
back to headless validation while the launcher notes the lack of a window so
contributors can update the documentation (see Law 11) and explain the
limitation in the `docs/` tree.

## Troubleshooting

- **No GUI window appears** – check whether Tkinter is installed in the
  managed environment. The GUI scaffolding silently skips rendering when Tk is
  missing, so the start script's message will now surface the headless fallback
  behaviour instead of disappearing without feedback.
- **The GUI closes immediately** – consult the console output that remains
  visible in the terminal; `copernican.py --gui` now logs errors before
  detaching, and the start script keeps that log readable for every attempt.
- **You prefer manual launch** – run `.venv/bin/python copernican.py --gui`
  (or `.venv\Scripts\python.exe` on Windows) directly so the GUI stays attached
  to the terminal.

## Documentation expansion

Law 11 in [`AGENTS.md`](../AGENTS.md) insists that documentation grows with
feature work, so every start script tweak must be mirrored in `README.md`,
`AGENTS.md` and at least one document in `docs/`. This file is the new chapter
for the launcher workflow, referencing the shared orchestration services,
exposing the `COPERNICAN_DETACH_GUI` flag and describing how the GUI now
detaches cleanly across platforms.
