# Copernican Suite GUI Guide
The Copernican Suite GUI wraps the manifest-driven workflow in a Tkinter
application so you can compose reproducible runs without touching the console.
This guide walks through the navigation rail, the Run Builder wizard, the Run
Monitor console and the inline help. Refer to `docs/cli_guide.md` if you prefer
to stay in the CLI or want to correlate GUI actions with the manifest pipeline.

## Navigation Rail
The left rail is always visible and reserves space for a padded Copernican logo
plus buttons for every page:

- **Home** – Surfaces catalogue health, model/engine badges, and quick actions
  for importing manifests, launching the Run Builder or opening the output
  directory.
- **Run Builder** – Opens the six-step wizard described below and keeps the
  Previous/Next/Cancel controls anchored beneath the jump buttons.
- **Run Monitor** – Streams sampler progress, logs and run alerts. Buttons for
  cancel, pause and hard stop share the same disabled state logic used
  elsewhere, so you always see whether the worker is running.
- **Data / Models / Engines** – Expose searchable catalogues sourced from the
  cached registries. Each page includes open-folder actions, metadata panes,
  parser revalidation controls and trust notes so you can inspect assets before
  entering the builder.
- **Validation** – Executes `python copernican.py --run-validation`,
  streams the CLI output into a Run Monitor–style log box, saves outputs under
  `validation/output/<manifest_stem>/copernican-run_<timestamp>/` and writes the
  latest summary to the gitignored `VALIDATION.md`. The manifest evaluates the
  fixed reference model against Union Through UNITY 2000 SNe, BOSS DR12 BAO and
  Planck 2018 Lite, declaring every parameter via `fixed` priors so the sampler
  still leaves a trace and the corner plot highlights that canonical point even
  though the values remain numerically fixed for validation. **Cancel validation**
  terminates the background worker, **Clear validation** removes every
  `validation/output/...` folder plus `VALIDATION.md`, and the “Lock summary to
  latest entry” checkbox keeps the log pinned to the newest lines while the GUI
  progress bars mirror the CLI counter state.
- **Settings** – Provides diagnostics filters, log viewers and output-directory
  helpers plus a recap of the `COPERNICAN_*` environment variables currently in
  effect.
- **Help** – Loads this guide or the CLI companion from `docs/` directly inside
  the GUI with a scrollable text viewer.
- **About / Exit** – Show the project overview or terminate the GUI and return
  to the launcher menu.

All navigation pages now share the same bold header style used by Run Builder
and Run Monitor, keeping the typography consistent regardless of the page.

## Run Builder Overview
The builder consists of six pages listed at the top of the panel. Jump buttons
have the same width as the Previous/Next/Cancel controls and use native ttk
states so Manifest and Confirm grey out until prerequisites are satisfied. A
contextual two-line message under the buttons explains what needs to happen on
each page. The steps are:

1. **Seed** – Enter a numeric seed or accept the default. The GUI respects the
  `COPERNICAN_SEED` environment variable and logs the final value into the run
  manifest and summary tables. Default (0), Random timestamp, Alien Invasion,
  Emoji Meteors, Constellation and the environment override buttons are
  arranged in a single vertical stack so screen-readers and keyboard users can
  tab through them predictably. Mini-game documentation lives next to the code:
  see [`rng_minigames/README.md`](../rng_minigames/README.md) for the API and the
  per-game READMEs under `rng_minigames/<game>/` for rules, accessibility notes
  and configuration settings. Alien Invasion exposes both a **Let AI take care**
  autopilot (which learns per workstation using cache files) and a Hall of Fame
  leaderboard so players can compare the fastest completions or let the AI
  practice on their behalf. The window also exposes Pause/Resume, **Let AI
  learn** (continuous loops) and **Let AI forget** controls, all documented in
  the alien-invasion README.
2. **Models** – Single-select list with quick metadata access. The preview pane
   stays pinned above the footer and shortens automatically so dataset controls
   remain visible.
3. **Data** – Three fixed-height (four-row) listboxes stack vertically for SNe,
   BAO and CMB catalogues. Each box is 500 px wide and uses a dedicated
   scrollbar so selections remain readable.
4. **Engine** – Selecting an engine loads its capability metadata and renders
   per-parameter controls inside the Run Settings box. Integer and float fields
   use spinboxes with bounded ranges taken from `_ENGINE_SETTING_LIMITS`; pool
   size is capped by the detected CPU core count. Boolean settings render as
   checkboxes (for example, Display progress). Recommendations display directly
   above their associated inputs.
5. **Manifest** – Displays the draft manifest in a scrollable text widget and
   surfaces reminder text if the workspace has not been saved. The buttons let
   you save, save-and-confirm, export to an external path, open the on-disk
   manifest or clear the workspace.
6. **Confirm** – Summarises the entire run, including walker/burn-in/production
   settings and pool size hints. The **Start run** button stays disabled until a
   manifest exists so every execution renames the workspace
   (`copernican-run_<timestamp>`) before launching the CLI worker.

`Next` is blocked on the Engine step until all of the first four pages have
selections. Attempting to proceed triggers a toast and a modal warning telling
you which prerequisites are missing. The Manifest page also refuses to advance
until the manifest is saved, ensuring the Confirm page always opens with a real
workspace on disk.

## Saving and Exporting Manifests
The builder creates a temporary workspace under `output/` named
`copernican_run_NEW_CONFIG`. Saving writes
`run_manifest_NEW_CONFIG.yml` into that directory, updates the summary metadata
and unlocks the Confirm step. You can then:

- Export the manifest to another folder via **Save to external folder...**
- Open the existing manifest with the OS default application
- Clear the configuration, which deletes the workspace and resets every field
- Save and jump straight to Confirm without manually clicking `Next`

The GUI and CLI now share `copernican_lib.run_executor.execute_run_from_manifest`,
so every saved manifest flows through the exact same orchestration code.

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

All warnings and system messages also appear in the footer status bar, which
shows the Copernican Suite version on the left and the Python/venv summary on
the right.

## Embedded Help
The Help page now mirrors the Run Builder control bar. Two buttons (GUI guide
and CLI guide) sit under the introductory paragraph, and the header updates to
“Help: GUI guide” or “Help: CLI guide” as you switch between them. Markdown is
rendered with heading, bold, italic and code styling so the guides remain easy
to read inside the GUI, complete with the project banner and scrollbars. Use
this page to keep documentation open beside the builder or monitor without
leaving the application.
