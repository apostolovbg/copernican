# GUI Overview

The Copernican Suite GUI is a Tkinter scaffold that mirrors the CLI workflow
while keeping the interactive controls lean. Its layout and runtime helpers
look to [`docs/architecture.md`](docs/architecture.md) and
[`docs/orchestration_services.md`](docs/orchestration_services.md) for the shared
ordering of validation, manifest generation, and execution. The Home screen
shows recent runs, pinned manifests, and quick actions (Run Builder, Run Monitor,
output directory, **Import manifest…**) so users can resume or duplicate any
workspace with zero shell interaction.

## Navigation and Branding

The navigation rail reserves 240 px, leaving a 60 px square for the Copernican
logo above the Home button. Equal spacing to the left and top chrome keeps the
rail visually balanced without adding legacy navigation stages. The rail lists
Home, Run Builder, Run Monitor, Data, Models, Engines, Settings, and Help—
all accessible without staging menus. The Start Scripts still offer CLI/GUI
choices, and the GUI launcher prints a hand-off message before detaching the
window (`pythonw` on Windows, `nohup` on Unix) so terminals do not stay open.

## Run Builder Panels

The Run Builder walks through the Stage 1 seed input, Stage 2 model and dataset
selection, engine choice, Run Settings, plan summary, and confirmation. Each
panel reads from the refreshed inventory generated during start-up so the GUI
always matches the automated catalogues described in `docs/architecture.md`.
Models and datasets use single-selection lists, while datasets separate SNe,
BAO, and CMB into widened listboxes with dedicated scrollbars so even 100+
entries stay legible. Badges show metadata, dataset hashes, citations, and
parser diagnostics, while **Open folder**, **View metadata**, and
**Revalidate parser** tap the same helpers as the CLI.

Run Settings capture walker count, burn-in steps, production steps, and
multiprocessing pool size—mirroring the CLI Stage 2 prompt after the CMB
dataset loads. The panel describes minimum values, “quick burn-in” shortcuts,
and CPU pool hints pulled from `multiprocessing.cpu_count()`. Those settings are
saved to the manifest so `copernican_lib.run_manifest` can replay them later.
Saving a manifest writes into a temporary workspace (`output/copernican_run_NEW_CONFIG/`)
and enables the **Save and confirm** button; Cancel/Clear delete the temporary
folder so aborted drafts do not linger.

The confirmation page keeps **Start Run** disabled until the manifest exists
and the builder has recorded the selected seed/model/dataset/engine/settings.
When the run starts the workspace is renamed to `copernican-run_<timestamp>` to
match CLI conventions and `copernican_lib.gui.run_worker` hands the manifest to
`copernican.main` with `--manifest`.

## Data, Models, and Engines

The Data tab renders dataset type filters (SNe, BAO, CMB) with row badges,
parser digests, dataset metadata references, and action buttons. Selecting
datasets inside the Run Builder automatically scrolls the catalog to the active
entry, and the GUI keeps a metadata viewer sized to the longest line and
restricted to Tkinter’s 15/25 rule. Each metadata dialog exposes an **Open
file…** button that launches the OS default editor so users can inspect YAML,
metadata, or parser source with a single click.

Models appear in a scrollable catalog with badges for compatibility flags,
`rs_expression`, and hashed metadata. The *Open model folder* and *View YAML*
actions duplicate CLI helpers so the GUI is never disconnected from the
underlying source files. Engines likewise show their version, badges, and
digest, letting operators inspect or revalidate the module before launching a
run.

## Run Monitor and Diagnostics

The Run Monitor streams `stdout`/`stderr` from the CLI worker (spawned by
`copernican_lib.gui.run_worker`) into a scrollable log console that filters INFO,
WARNING, and ERROR levels. The monitoring panel shows dual progress bars for the
current batch and walker progress plus a walker-level duplicate of the CLI’s
fifty-character progress meter. Spinner glyphs and percentage text follow the
carriage-return renderer introduced in version 7.6.14, ensuring the GUI log and
console log remain visually consistent.

Cancel, Pause, and Hard Stop buttons stay greyed out until a run actually
starts; once active they send signals that terminate the CLI worker (Cancel/Hard
Stop) or request a pause/resume handshake through the shared orchestration
interfaces described in [`docs/orchestration_services.md`](docs/orchestration_services.md).

## Metadata Dialogs and Help

Metadata, YAML, and parser viewers size themselves to their longest line,
lock horizontal resizing, and obey the design-requested minimum heights. The
dialogs include an **Open file…** helper that delegates to the OS default editor
and adds vertical scrollbars so long tables remain legible.

The Help panel renders the latest `README.md` (banner included) inside a
scrollable text widget so documentation is always accessible from within the
GUI. A dedicated `docs/banner_github.png` display ensures the GUI view matches
the GitHub experience even when the launcher runs headless.

## Settings and Environment Hints

The Settings page exposes diagnostic filters plus:

- Output directory helpers (path entry, create/refresh buttons, *Open directory*).
- Environment hint toggles for `COPERNICAN_SEED`, `COPERNICAN_STRICT_WARNINGS`,
  `COPERNICAN_ENABLE_STAGED_MENU`, `COPERNICAN_DETACH_GUI`, `COPERNICAN_HEADLESS_RUN`, and `COPERNICAN_DETACH_GUI`.
- Quick links to `docs/gui_overview.md`, `docs/run_manifest.md`, and the log
  directory so operators can jump to the referenced documentation or tail the
  current run log.

Settings also surface instrumentation tips that remind users to run `pre-commit
run --all-files` before editing (the same check that enforces DevCovenant) and
to use `python -m unittest discover` when they choose the *Run the unit test
suit* option from the menu.
