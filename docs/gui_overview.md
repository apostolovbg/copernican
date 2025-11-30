# GUI Overview

The Copernican Suite GUI uses a lightweight Tkinter scaffold so it can run
inside the managed virtual environment without extra framework dependencies.
The navigation rail stays visible at all times and the Home screen shows recent
runs, pinned configurations and explicit quick actions for launching the Run
Builder, the Run Monitor, or the output directory.

## Run Builder

The Run Builder now mirrors the CLI stages with dedicated panels for the
seed, model selection, dataset selection, engine choice, plan notes and the final
confirmation. Each stage draws live data from the inventories generated during
GUI start-up so you can click to open folders, preview metadata or revalidate
parser trust statements while you configure a new run. The confirm panel lists
the choices that will go into the manifest before hitting *Start Run* so
operators never launch without verifying seeds, models or datasets.

## Data

The Data catalogue renders a scrollable list of every registered dataset along
with its badges, citations, parser METADATA and digest information. Each row
offers *Open folder*, *View metadata* and *Revalidate parser* buttons that run
the same helpers as the CLI. Filters for SNe, BAO and CMB remain available so
you can narrow the view while keeping the trust scores visible.

## Models

Models now appear in a scrollable panel with badge, license and SHA256 details.
The *Open model folder* and *View YAML* actions link the UI directly to the
underlying YAML files so you can inspect definitions without leaving the GUI.

## Engines

Every discovered engine shows its label, version, badges and digest inside its
own framed row. *Open engine folder* and *View module* buttons call the same
helpers the CLI uses, ensuring the GUI never feels like a stub even with just a
handful of engines.

## Settings

The Settings screen keeps the diagnostics frame from before while adding an
Output directory helper (entry, create/refresh buttons and an open flag) and
environment hints for variables such as `COPERNICAN_SEED`,
`COPERNICAN_STRICT_WARNINGS`, `COPERNICAN_ENABLE_STAGED_MENU` and
`COPERNICAN_DETACH_GUI`.

## Help

The Help panel now renders `README.md` inside a scrollable text widget, complete
with the `docs/banner_github.png` image so documentation stays available even
when operators prefer the GUI to the CLI.

## Launching the GUI from the Start Scripts

Selecting the GUI option from `start.sh`, `start.command` or `start.bat` still
runs the shared `copernican.py --gui` entry point, but now each launcher prints
a clear message before handing the window over to the detached process and
waits for `copernican.py` to confirm the handoff. The new
[Launchers and GUI](docs/launcher_gui.md) guide documents how the start scripts
set `COPERNICAN_DETACH_GUI=1`, rely on `copernican.py` for concurrency and keep
the terminal focused on the orchestration services log so operators quickly
see whether the GUI actually started.
