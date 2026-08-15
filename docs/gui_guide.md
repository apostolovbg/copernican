# Copernican GUI Guide
The Copernican GUI wraps the manifest-driven workflow in a Tkinter
application so you can compose reproducible runs without touching the console.
This guide walks through the navigation rail, the Run Builder wizard, the Run
Monitor console and the inline help. Refer to `docs/cli_guide.md` if you prefer
to stay in the CLI or want to correlate GUI actions with the manifest pipeline.
## Table of Contents
- [Navigation Rail](#navigation-rail)
- [Run Builder Overview](#run-builder-overview)
- [Saving and Exporting Manifests](#saving-and-exporting-manifests)
- [Run Monitor and Diagnostics](#run-monitor-and-diagnostics)
- [Embedded Help](#embedded-help)
## Navigation Rail
The left rail is always visible and reserves space for a padded Copernican logo
plus buttons for every page:
- **Home** – Surfaces catalogue health, model/sampler badges, and quick
 actions for importing manifests, launching the Run Builder, or opening the
 output directory.
- **Run Builder** – Opens the seven-step wizard described below and keeps the
 Previous/Next/Cancel controls anchored beneath the jump buttons.
- **Run Monitor** – Streams sampler progress, logs, and run alerts. Buttons
 for cancel, pause, and hard stop share the same disabled state logic used
 elsewhere, so you always see whether the worker is running.
- **Data / Models / Sampler engines** – Expose searchable catalogues sourced
 from the cached registries. Each page includes open-folder actions, metadata
 panes, parser revalidation controls, and trust notes so you can inspect assets
 before entering the builder.
- **Validation** – Executes `python -m copernican
  --run-validation`, streams the CLI output into a Run Monitor–style log box,
 saves outputs under
 `copernican/validation/output/<manifest_stem>/validation_run_<timestamp>/`
 and writes the latest summary to `~/VALIDATION.md`. The manifest evaluates
 the fixed reference model against Union Through UNITY 2000 SNe, BOSS DR12 BAO
 and Planck 2018 Lite, declaring every parameter via `fixed` priors so the
 sampler leaves a trace and the corner plot highlights that canonical
 point even though the values remain numerically fixed for validation.
 **Cancel validation** terminates the background worker, **Clear validation**
 removes every `copernican/validation/output/.../validation_run_*` folder plus
 `~/VALIDATION.md`, and the “Lock summary to latest entry” checkbox keeps
 log pinned to the newest lines while the GUI progress bars mirror the CLI
 counter state.
- **Settings** – Provides diagnostics filters, log viewers, and
 output-directory helpers plus a recap of the `COPERNICAN_*` environment
 variables in
 effect.
- **Help** – Loads this guide or the CLI companion from `docs/` directly
 inside the GUI with a scrollable text viewer.
- **About / Exit** – Show the project overview or terminate the GUI and
 return to the launcher menu.
All navigation pages share the same bold header style used by Run Builder
and Run Monitor, keeping the typography consistent regardless of the page.
## Run Builder Overview
The builder consists of seven pages listed at the top of the panel. Jump
buttons have the same width as the Previous/Next/Cancel controls and use
native ttk
states so Manifest and Confirm grey out until prerequisites are satisfied. A
contextual two-line message under the buttons explains what needs to happen on
each page. The steps are:
1. **Seed** – Enter a numeric seed or accept the default. The GUI respects
 `COPERNICAN_SEED` environment variable and logs the final value into the run
 manifest and summary tables. Default (0), Random timestamp, Alien Invasion,
 Emoji Meteors, Constellation and the environment override buttons are
 arranged in a single vertical stack so screen-readers and keyboard users can
 tab through them predictably. Mini-game documentation lives next to the
 code: see `copernican/rng_minigames/README.md` for the API and the
 per-game READMEs under `copernican/rng_minigames/<game>/` for rules,
 accessibility notes and configuration settings. Alien Invasion exposes both
 a **Let AI take care** autopilot (which learns per workstation using cache
 files) and a Hall of Fame leaderboard so players can compare the fastest
 completions or let the AI practice on their behalf. The window also exposes
 Pause/Resume, **Let AI learn** (continuous loops) and **Let AI forget**
 controls, all documented in the alien-invasion README.
2. **Control model** – Select the comparison control from the shared model
 catalogue. `model_lcdm.yml` is selected by default, and the preview pane and
 `Load model...` action use the same validation behavior as the test page.
3. **Test model** – Select the model evaluated against the control. The page
 uses the same single-select list, metadata preview, and exact-path loading
 behavior as the control page.
4. **Data** – Three fixed-height (four-row) listboxes stack vertically for
 SNe, BAO, and CMB catalogues. Each box is 500 px wide and uses a dedicated
 scrollbar so selections remain readable.
5. **Sampler engine** – Selecting a sampler loads its capability metadata and
 renders per-parameter controls inside the Run Settings box. Integer and
 float fields use spinboxes with bounded ranges taken from
 `_ENGINE_SETTING_LIMITS`; pool
 size is capped by the detected CPU core count. Boolean settings render as
 checkboxes (for example, Display progress). Recommendations display directly
 above their associated inputs. The fixed Copernican native declared-graph CMB
 engine is shown for provenance and is not a selectable builder step.
6. **Manifest** – Displays the draft manifest in a scrollable text widget and
 surfaces reminder text if the workspace has not been saved. The buttons let
 you save, save-and-confirm, export to an external path, open the on-disk
 manifest or clear the workspace.
7. **Confirm** – Summarises the entire run, including the control/test pair,
 walker/burn-in/production settings, pool size hints, and the fixed native CMB
 engine. The **Start run** button stays disabled until
 a manifest exists so every execution renames the workspace (`copernican-
   run_<timestamp>`) before launching the CLI worker.
`Next` is blocked on the Sampler engine step until all of the first five pages
have selections. Attempting to proceed triggers a toast and a modal warning
telling
you which prerequisites are missing. The Manifest page also refuses to advance
until the manifest is saved, ensuring the Confirm page always opens with a real
workspace on disk.
## Saving and Exporting Manifests
The builder creates a temporary workspace under `output/` named
`copernican_run_NEW_CONFIG`. Saving writes `run_manifest_NEW_CONFIG.yml` into
that directory, updates the summary metadata and unlocks the Confirm step. You
can then:
- Export the manifest to another folder via **Save to external folder...**
- Open the existing manifest with the OS default application
- Clear the configuration, which deletes the workspace and resets every field
- Save and jump straight to Confirm without manually clicking `Next`
The GUI and CLI share
`copernican.lib.run_executor.execute_run_from_manifest`, so every saved
manifest flows through the exact same orchestration code.
## Run Monitor and Diagnostics
The Run Monitor page exposes:
- A status label describing the orchestration phase
- Batch and walker progress bars with live percentages from Stage 2
- Manifest metadata lines so you can confirm which assets are running
- A log viewer with filter buttons (Info/Warnings/Errors) and shortcuts to view
 or open the active log file
- Alerts that jump directly to recorded anchors inside the log
- Buttons to pause, cancel, hard stop or open the output directory once the run
 starts producing files
The child worker owns the single canonical run log. Its structured console
transport supplies the in-memory monitor with the original event severity;
the monitor and progress snapshot channel never write back into that file.
The GUI resolves one output directory, timestamp, and log name before launch,
and the CLI manifest executor consumes that same identity.
All warnings and system messages also appear in the footer status bar, which
shows the Copernican version on the left and the Python/venv summary on
the right.
## Embedded Help
The Help page mirrors the Run Builder control bar. Two buttons (GUI guide
and CLI guide) sit under the introductory paragraph, and the header updates to
“Help: GUI guide” or “Help: CLI guide” as you switch between them.
Markdown is
rendered with heading, bold, italic and code styling so the guides remain easy
to read inside the GUI, complete with the project banner and scrollbars. Use
this page to keep documentation open beside the builder or monitor without
leaving the application.
