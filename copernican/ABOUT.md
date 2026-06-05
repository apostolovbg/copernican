# About Copernican
**Doc ID:** ABOUT
**Doc Type:** repo-about
**Project Version:** 12.0.26
**Last Updated:** 2026-06-05
**DevCovenant Version:** 1.0.1b6

## Table of Contents
- [Overview](#overview)
- [Package Surface](#package-surface)
- [Documentation Model](#documentation-model)
- [Where To Start](#where-to-start)

## Overview
Copernican is a package-first Python toolkit for testing cosmological models
against SNe Ia, BAO, and CMB observations. The repository keeps the runtime,
the dataset registry, the GUI, and the validation helpers under one governed
workflow so analysts and maintainers can trace every run back to a manifest,
selected datasets, and a logged execution path.

The project is built around explicit package surfaces rather than ad hoc
script entry points. The `copernican` package owns the runtime entry points,
the bundled assets, and the mirrored package-root documentation that users
see first when they open the repository or launch the GUI.

The bundled engines, models, and validation helpers now live under
`copernican/`, and validation writes its local summary marker to
`~/VALIDATION.md` so package installs and source checkouts follow the same
path layout.

The CLI and GUI share one application logger, and each run keeps its own
run logs inside the generated `~/copernican_output/copernican-run_*`
folder.

The model chooser now offers `Load model...` so any valid `.yml` or
`.yaml` file can load by exact path in the CLI or GUI.

The engine modules dropped the `cosmo_` prefix, so the docs and package
surface now refer to `engine_*` files. The MCMC initializer uses a
tolerance cutoff for tiny singular values to keep walker startup stable
across platforms.

`copernican/workflow.py` owns the launch flow for the package entry points,
and `copernican/lib/global_settings/copernican_settings.yml` carries the
GUI-facing defaults that shape that flow.

The GitHub Actions governance job also bootstraps the repo-local `.venv`
before it invokes DevCovenant, keeping CI aligned with local managed
environment runs.

## Package Surface
The package root is part of the public contract. It includes the README,
policy-facing metadata, and the mirrored support documents that describe how
the project behaves in practice.

- `copernican/README.md` gives the launch and workflow summary.
- `copernican/ABOUT.md` explains the project shape at a higher level.
- `copernican/SECURITY.md` describes how to report security issues.
- `copernican/SUPPORT.md` explains where to get help and what to include.
- `copernican/CITATION.cff` carries the citation metadata for the package.
- `copernican/engines/` contains the bundled cosmology engines.
- `copernican/models/` contains the bundled model definitions.
- `copernican/validation/` contains the manifest runner and validation docs.

`copernican/workflow.py` and
`copernican/lib/global_settings/copernican_settings.yml` sit behind those
front-door docs, so changes in the launch path or GUI defaults should be
reflected there first.

Those files are mirrored from the root copies so the package can ship the same
documentation surface without inventing a second narrative.

## Documentation Model
Copernican keeps two documentation tracks deliberately separate.

The root `README.md`, `ABOUT.md`, `SECURITY.md`, `SUPPORT.md`, and
`CITATION.cff` form the package-facing doc set. They are the authored source
files. `package-doc-sync` mirrors them into `copernican/` so the package can
ship the same content without manual duplication.

The `docs/` tree is the long-form manual set. It stays synchronized with
`copernican/docs/` and covers the deeper runtime, GUI, dataset, and packaging
guides. That separation keeps the short front-door docs concise while the
manual docs stay detailed enough for operators who need to work through the
workflow step by step.

## Where To Start
If you are new to Copernican, read `README.md` first, then open the manual
docs that match your task. Use the GUI guide for interactive runs, the CLI
guide for scripted execution, and the dataset or manifest guides when you need
to inspect the trusted inputs that drive a run.

If you are maintaining the repository, keep the package-root docs mirrored,
keep the manual docs identical between `docs/` and `copernican/docs/`, and
use the DevCovenant workflow to keep the docs, policies, and changelog in
sync.
