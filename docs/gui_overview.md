# GUI Overview

The Copernican Suite GUI uses a lightweight Tkinter scaffold so it can run
inside the managed virtual environment without extra framework dependencies.
The navigation rail stays visible at all times and the Home screen shows recent
runs, pinned configurations and explicit quick actions for launching the Run
Builder, the Run Monitor, the output directory or the **Import manifest...**
workflow that clones an existing manifest for reuse. The navigation rail now
reserves 240 px so the padded Copernican logo square above the Home button keeps
equal spacing to the window chrome from the left border, title bar and the main
pane divider without introducing extra controls. The icon sits in a 60 px square
so the visual balance is maintained even with the tighter top/left padding.

## Run Builder

The Run Builder now mirrors the CLI stages with dedicated panels for the seed,
model selection, dataset selection, engine choice, plan notes and the final
confirmation. Models and datasets use single-selection lists so you cannot
select more than one item at each stage, and every panel draws live entries
from the inventories generated during GUI start-up. Revalidation, metadata
previewing and folder opening remain available inside the builder, and the
confirm panel lists the new run settings alongside the usual seeds, models and
datasets before operators hit *Start Run*.
Each dataset type renders in its own widened listbox with a dedicated scrollbar
so even long catalogues stay visible without re-introducing multi-select menus.
The navigation controls now grey out *Previous* on the first step and *Next* on
the last so operators always see when they can move, and the only way to launch
sampling is through the confirmation step’s **Start Run from manifest** button.
A companion **Insert manifest** button stages the generated manifest so you can
review metadata or export it before launching the worker.

The Save Manifest step now stays locked until every seed/model/data/engine
panel reports a selection. Saving writes the current manifest to
`output/copernican_run_NEW_CONFIG/run_manifest_NEW_CONFIG.yml`, leaves the
workspace editable and enables the **Save and confirm** controls. Cancel and Clear
actions purge that temporary folder so aborted builders never leave stray manifests,
and the confirmation panel keeps its **Start run** button disabled until a
manifest exists so the GUI always renames the workspace to `copernican-run_<timestamp>`
before handing it to `copernican_lib.gui.run_worker`.

## Data

The Data catalogue renders separate scrollable lists for each dataset type so
you choose at most one SNe, BAO or CMB entry per run. Each row shows badges,
citations, parser metadata and digests while the *Open folder*, *View metadata*
and *Revalidate parser* buttons call the same helpers the CLI exposes.

## Run Settings

The builder now includes a Run Settings panel where you set the number of
walkers, burn-in steps, production steps and multiprocessing pool size before
confirming a run. The panel mirrors the CLI prompts verbatim—reminding you
about minimum walkers, recommended defaults, “quick burn-in” shortcuts and the
current CPU count—so GUI launches receive the same context as the terminal
workflow. These values are stored in the run manifest so downstream replays or
audits capture every execution parameter.

## Run Execution

Pressing **Start Run** launches the full CLI workflow in a background worker
process using the current builder configuration. The GUI streams the worker's
stdout/stderr into the diagnostics panel, mirrors CLI log messages and exposes
Cancel/Hard Stop buttons that terminate the child process when you need to stop
early. (Pause/resume is still a CLI-only feature.) GUI workers now set
`COPERNICAN_HEADLESS_RUN=1` so the CLI exits cleanly after finishing instead of
prompting for another run, and the per-run log captures any unexpected
exceptions even when the run is launched from the GUI.

The Run Monitor now mirrors the CLI progress state with dual progress bars
for the current batch/iteration counts and the walker-level reports plus a
scrollable log console that tails `logs/runs/*.txt`. The filter buttons keep
INFO, WARNING or ERROR entries visible so you can follow the exact same
diagnostics the command line renders while the GUI keeps every alert anchored
for quick navigation. A new “Lock log to latest entry” checkbox beside the
filters pins the view to the most recent lines whenever you want to watch the
ensemble finish without manually scrolling back down. That log console now
drops the CLI counter rows (e.g., “Burn-in stage batch 1: 3/200 steps
completed”) so it shows only batch summaries, and the Cancel/Pause/Hard Stop
buttons stay disabled (greyed out) until a run starts, after which they return
to their normal, clickable appearance.
Run executor logs remain separate: the CLI worker still writes its full trace,
including dataset discovery, to `output/copernican-run_<timestamp>/copernican-run_<timestamp>.txt`
for reproducibility, while the Run Monitor tails the dedicated GUI stream under
`logs/runs/` so you can inspect a compact monitoring log without trimming the
scientific artefacts.
The monitor refresh loop now verifies every progress label, bar and log text
widget still exists before touching it, so switching away from the Run Monitor
or Validation tabs no longer triggers Tk errors while the background run keeps
streaming updates uninterrupted.

## Analysis

The navigation rail now includes an **Analysis** button between Engines and
Validation. Pressing it opens a 4-tab workspace that mirrors the Settings
screen’s tab bar. Each tab shows a contextual description, a row of three action
buttons and a dedicated body frame—Diagnostics remains a placeholder while Run
Summary, Posteriors and Comparisons now present concrete workflows powered by the
shared `copernican_lib.analysis` helpers.

Run Summary lets you browse to any ``output/copernican-run_*`` folder, load the
manifest/log/parameter summary-derived diagnostics and render the results inside a
scrollable text pane. The “Reload summary” button re-parses the folder,
“Export summary” saves structured ``analysis-summary_<timestamp>.yml``/``.json``
copies into the selected directory, and “Copy JSON” pours the serialised result
into the clipboard for quick reporting. The summary lists datasets, row counts,
R-hat/ESS diagnostics, model χ² components, BAO ``r_s`` values and the run’s
timestamps without forcing you to re-run the CLI workflow.

The Posteriors tab now hooks directly into ``posterior-*.nc`` outputs via
``copernican_lib.posterior_explorer`` and renders their trace/hist overview inside the
shared :class:`copernican_lib.gui.plot_viewer.PlotViewer`. The action buttons above the
tab refresh the file list, autoscale the plot and toggle drag-enabled zoom/pan while the
control row lets you pick a snapshot and load it into the viewer. Fit-to-screen and
“restore view” helpers keep the traces legible, and the zoom toggle enables a click-and-drag
pan so you can inspect any region without re-creating the plot. A second control row
offers “Show corner plot” and “Show histograms” actions that load the latest archived PNGs
from the run directory so you can quickly compare the derived Surfaces without re-opening
the sampler output.

The Comparisons tab now lets you load two run directories, view Δχ²/parameter shifts
and dataset-count deltas in a JSON panel, and export or copy the structured
comparison summary that the new `copernican_lib.analysis.compare_runs` helper produces.
Refreshing the diff re-analyses both folders via `copernican_lib.analysis.analyze_run`
so you can compare any historical results without re-running the CLI workflow.

## Validation
The navigation rail now includes a **Validation** button positioned between
Engines and Settings. When you press it, the GUI launches `python copernican.py
--run-validation`, streams the CLI stdout into the Run Monitor–style log box,
writes the NEW_CONFIG, plots and logs into
`validation/output/<manifest_stem>/validation_run_<timestamp>/`, and stores the
textual summary in the gitignored `VALIDATION.md` file so the panel can reload
the latest results even when the suite is not rerun. The manifest evaluates the
fixed reference model against Union Through UNITY 2000 SNe, BOSS DR12 BAO and
Planck 2018 Lite and declares each parameter using `fixed` priors so the sampler
still records the canonical point while keeping the values numerically locked
for regression checks. A **Cancel validation** button terminates the worker mid-run,
**Clear validation** removes every `validation/output/.../validation_run_*`
directory plus the gitignored summary, and the “Lock summary to latest entry”
checkbox keeps the log pinned to the newest lines while the GUI progress bars
mirror the CLI counter state.
The validation button stays disabled while the run is active so you cannot stack
overlapping validations.

## Metadata dialogs

Metadata, YAML and module viewers automatically size themselves to the longest
line, enforce the 15/25-line minimum and default window sizes requested by
design, and include an **Open file…** button that launches the underlying asset
in the operating system's default editor. The dialogs keep horizontal resizing
locked while allowing unlimited vertical resizing so short files stay compact
and long YAMLs remain comfortable to read.

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

The Settings screen now mirrors the Run Builder navigation: four tabs list
Logging, Datasets, GUI and Tools options in the same style as the wizard
controls above.  Logging lets operators set retention counts, severity
thresholds and console capture before purging archived diagnostics files
directly from the same panel.  The Datasets page toggles automatic discovery,
hash caching and offers a manual digest rebuild so trusted parsers stay current.
GUI settings cover automatic detachment, managed `.venv` enforcement and the
environment hints (`COPERNICAN_SEED`, `COPERNICAN_STRICT_WARNINGS`,
`COPERNICAN_DETACH_GUI`) shown on the status bar.  The Tools tab surfaces
maintenance helpers such as rebuilding the sanitized `models/cache` files,
revalidating the parser registry, and resetting the Run Builder workspace.
All preferences persist through `copernican_lib/settings.py` into the shared
`copernican_settings.yml` file so GUI and CLI launches honour the same defaults.

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
