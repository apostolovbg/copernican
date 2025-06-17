# DEV NOTE (v1.5h)
Planning notes updated after implementing phases 0–3 with a basic Flask web interface and session handling.

# Copernican Suite Web Transition Plan
This document replaces the previous refactoring roadmap. All earlier phases were completed in version 1.5f. Phases 0–3 are now implemented in version 1.5h with a minimal Flask server. The legacy CLI remains for testing but will be deprecated once the web UI is fully functional.

After each phase is finished, append a note to the **Progress Tracking** section at the end of this file.

## Phase 0 – Framework Selection & Setup
1. **Pick a lightweight framework** – Use Flask for the initial implementation. It keeps dependencies minimal and is easy to deploy with a WSGI server.
2. **Create `webapp/` package** – Contains the Flask application, HTML templates and static assets.
3. **Refactor CLI logic** – Move the current run workflow in `copernican.py` into reusable functions so the web server can call them directly.

## Phase 1 – Basic Web Interface
1. **Landing Page** – Displays a header with the *Copernican Suite* name, a short description and the current version.
2. **New Run / Abort Run Button** on top of the landing page, below the header, always visible. Starts or cancels a run. When a run is active the button text changes to *Abort run* and turns red.
3. **Compile Model Button** – beside the previous button, again, always visible. Placeholder for the future model compiler. For now it simply shows a stub message.

*Phase 0 implemented in v1.5h: Flask app created and CLI refactored.*
## Phase 2 – Tabbed Workflow
1. **Tab Layout** – Below the main buttons add a tab bar with the following tabs:
   - **Alternative model** – Initially contains a file upload control for `cosmo_model_*.json` and a Test button which selects cosmo_model_lcdm.json which will continue to live under models/. Other tabs stay disabled until a file is chosen. When a file is chosen, a model summary will appear under the buttons - it shows the model equations in formatted math plus metadata from the JSON file.
   - **Computational engine** – Lists all engines found in the `engines/` directory. Each engine is a clickable button. A confirmation button *Employ selected engine* activates the choice.
   - **Datasets** – Presents available data sources per type (e.g. SNe options: University of Strasbourg or Pantheon+). Selecting a source automatically picks the correct data files and parser - h2_unistra will be deprecated. There will be only one data parser per source - unistra will always use tablef3.dat, read with h1_unistra, and pantheon will always use Pan.dat + Pancm.cov, read with pantheon parser. Thus, data file selection and parser selection will be deprecated. The user will only select the preferred data source per every data type. The current only BAO data source will be called just Basic. More will be added in the future - sources for BAO and SNe, and other data types and sources for them.
   - **Run** – Contains the console and a *Test your model* button that begins the analysis.
   - **Hubble diagram**, **BAO**, and future data tabs – Display plots generated after a run completes.
   - **Export results** – Provides a *Download all results of the last run in a .zip* button.
2. **Tab Activation** – Tabs become active in order: uploading a model enables *Model summary* and *Computational engine*, confirming dataset choices enables *Run*, finishing the run enables the plot tabs and export tab.

*Phase 2 implemented in v1.5h: Tab layout and placeholders added to web interface.*
## Phase 3 – Running Analyses
1. **Session Management** – Starting a run creates a timestamped session folder under `output/` to store plots, CSV files and the log.
2. **Abort Logic** – Clicking *Abort run* stops the current process and marks the run as cancelled. Cached files persist until the user starts a new run.
3. **Result Download** – After a run completes, users can fetch a ZIP archive of all outputs via the export tab.
4. **Prompt Before Discarding** – If cached results exist, a new run request prompts: "Do you really want to discard results from the last run? Please make sure you have downloaded them or you don't really need them, because they will vanish into the vacuum of space!" with options **Space them out** or **Cancel**.
*Phase 3 implemented in v1.5h: Sessions and downloads available via Flask UI.*

## Phase 4 – Future Enhancements
1. **Model Compiler Module** – A form-driven tool for creating new JSON models directly in the browser - appears just as the tab bar and box, below the New run and Compile model buttons, when compile model is clicked
2. **Additional Data Types** – Extend dataset selection and plotting tabs for CMB, gravitational waves and other observations.
3. **Deployment** – Package the Flask app with a WSGI server like Gunicorn. Document how to run locally and deploy on hosting providers.
4. **Deprecate CLI** – Once the web UI is stable and tested, remove CLI prompts from `copernican.py` and route all interactions through the web interface.

---
### Progress Tracking
- *2025-06-20* – Initial plan created for web interface transition.
- *2025-06-21* – Phases 0–3 completed with Flask skeleton and session management.
