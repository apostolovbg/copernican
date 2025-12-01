# About Copernican Suite

**Version:** 10.9.6  
**License:** See `LICENSE.md` (Copernican Suite developers, 2025)

The Copernican Suite is a cross-platform framework for evaluating
cosmological models against astronomical observations.  It runs on a
dedicated Python 3.11 interpreter, isolates every dependency inside `.venv`,
and ships a GUI shell that mirrors the CLI workflow so analysts can rapidly
compare models, engines and datasets while keeping their results
reproducible.

## Key Features

- **Model agnostic pipelines.** Models describe their equations, priors and
  dataset compatibility inside `models/cosmo_model_*.yml`, while the suite
  constructs engine plugins that stay picklable and multiprocessing-friendly.
- **Dataset parity.** SNe, BAO and CMB sources live under `data/` with
  metadata, parsers and SHA256 digests; the registry enforces `dataset_id`
  alignment so every run manifest names the exact files used.
- **Engine transparency.** Engines such as `engines/cosmo_engine_mcmc.py`
  expose `ENGINE_LABEL`, `ENGINE_VERSION` and `ENGINE_KIND` so the GUI and CLI
  always report the sampler configuration before it starts.
- **Reproducible output.** Every run writes a `copernican-run_YYYYMMDD_HHMMSS`
  directory with plots, NetCDF chains and `run_manifest_*.yml` files describing
  the models, datasets, Git state and dataset hashes used for that run.
- **Diagnostics-first.** Launchers keep a `logs/` directory for program and
  run-level logs, the GUI mirrors the live logs inside its Run Monitor, and
  both CLI and GUI feed the same audited log streams.
- **Live monitoring improvements.** The new Run Monitor navigation button
  exposes live progress bars, filtered run logs with a dedicated viewer, an
  “Open run output” link, Cancel/Pause/Hard Stop controls, and the Exit Suite
  action so GUI sessions mimic the CLI exit routine while the diagnostics pane
  still provides View/Open/Flush buttons for its log stream.

## Citation

If you use the Copernican Suite in a publication, cite the project as:

```bibtex
@software{copernican-suite,
  author = {Apostolov, Apostol and Copernican Development Team},
  title = {Copernican Suite},
  version = {10.9.6},
  year = {2025},
  url = {https://github.com/apostolovbg/copernican},
}
```

## Getting Help

- Review `README.md` for quick-start steps, launcher guidance and CI notes.
- Inspect `docs/` for the GUI, packaging and dataset overview guides.
- Launch `.venv/bin/python copernican.py --gui` once the managed environment
  is ready, then use the Run Monitor to watch the live sampler progress.

Need more help? Open an issue on the GitHub repository or consult the
`docs/launcher_gui.md` chapter for launcher diagnostics and workflows.
