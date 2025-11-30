# GUI Overview

The Copernican Suite GUI uses a Tkinter scaffold so it can run inside the
managed virtual environment without extra dependencies. The left rail remains
visible across all screens and exposes the following sections with keyboard
shortcuts noted in parentheses:

- Home (`Ctrl+1`)
- Run Builder (`Ctrl+2`)
- Data (`Ctrl+3`)
- Models (`Ctrl+4`)
- Engines (`Ctrl+5`)
- Settings (`Ctrl+6`)
- Help (`Ctrl+7`)

## Home
The Home panel lists recent runs, pinned configurations and quick actions. Each
entry is focusable to support screen readers and keyboard navigation.

## Run Builder
The Run Builder follows the seed, models, data, engine, plan and confirm steps.
Users can jump between stages via the step buttons, cancel at any time or save
a draft to revisit later without losing progress.

## Run Monitor
The Run Monitor shows a status strip and progress meter for the active run.
Cancel and stop controls pause or terminate processing while keeping the
monitor visible.

## Summary
After a run completes, the Summary screen surfaces output links and manifest
reuse actions so follow-on runs can launch from the same configuration.
