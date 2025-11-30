# GUI Overview

The Copernican Suite GUI uses a lightweight Tkinter scaffold so it can run
inside the managed virtual environment without extra framework dependencies.
The navigation rail stays visible at all times and the Home screen shows recent
runs, pinned configurations and explicit quick actions for launching the Run
Builder, the Run Monitor, or the output directory.

## Run Builder

The Run Builder now mirrors the CLI stages with dedicated panels for the seed,
model selection, dataset selection, engine choice, plan notes and the final
confirmation. Models and datasets use single-selection lists so you cannot
select more than one item at each stage, and every panel draws live entries
from the inventories generated during GUI start-up. Revalidation, metadata
previewing and folder opening remain available inside the builder, and the
confirm panel lists the new run settings alongside the usual seeds, models and
datasets before operators hit *Start Run*.

## Data

The Data catalogue renders separate scrollable lists for each dataset type so
you choose at most one SNe, BAO or CMB entry per run. Each row shows badges,
citations, parser metadata and digests while the *Open folder*, *View metadata*
and *Revalidate parser* buttons call the same helpers the CLI exposes.

## Run Settings

The builder now includes a Run Settings panel where you set the number of
walkers, burn-in steps, production steps and multiprocessing pool size before
confirming a run. These values are stored in the run manifest so downstream
replays or audits capture every execution parameter.

## Metadata dialogs

Metadata, YAML and module viewers automatically size themselves to the longest
line, enforce a sensible minimum height and include an **Open file…** button
that launches the underlying asset in the operating system's default editor.
The dialogs remain resizable, so scrolling through long YAMLs or dataset notes
feels the same as opening the files directly.

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
