# About Copernican
**Doc ID:** ABOUT
**Doc Type:** repo-about
**Project Version:** 12.0.26
**Last Updated:** 2026-06-22
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

The CMB surface now includes a declared-math graph engine for
`standard: false` contracts. It evolves one declared graph per `k` mode, then
projects the solved graph through declared transfer components and
line-of-sight integration instead of routing models through a theory-family
compatibility layer. Non-standard contracts now declare variables,
equations, constraints, closures, sources, initial conditions, observable
mappings, and numerical requirements in one immutable graph; unsupported
symbols, unsolved variables, missing initial conditions, missing
observables, incompatible projection-role bindings, and unsupported
projection kernels fail loudly. The implementation now lives in the
`copernican/lib/likelihoods/cmb/` package, where `cmb.py` owns the public
likelihood surface, `camb_solver.py` owns the standard backend route, and
`copernican_cmb_solver.py` owns the native internal orchestration layer.
`model_coder.py`
compiles the static native runtime once and `engine_adapter.py` hands that
runtime to the likelihood package directly, so the native path no longer
rebuilds a CAMB-style contract before every prediction. That runtime now
carries compiled background and reionization evaluator plans, and the
declared perturbation compiler now stores picklable expression programs plus
ordered graph metadata so the native solver can reuse dense slot plans
instead of re-parsing expressions and rescanning unresolved mappings inside
repeated solver stages. The native solver now batches projection kernels
across `ell`, reuses cached background and recombination products when the
declared background inputs are unchanged, and keeps runtime-response
behavior tests on lighter helper numerics while the reference-backed
scientific checks stay unchanged in the same governed suite. The background
helper consumes the declared
background graph, computes a
Peebles-style recombination history, integrates the declared reionization
ODE, and builds the visibility, optical-depth, and residual-ionization
curves before the transfer-function projection runs. Declared background
outputs now feed native density, pressure, equation-of-state, and curvature
quantities directly; the perturbation runtime can mix `tau`, `eta`, `a`,
`z`, or other declared monotonic background coordinates on equation
left-hand sides; and end-anchored boundary conditions can drive the native
shooter when they replace the missing start-state slots. Declared
observables may now target TT,
TE, EE, BB, lensing-potential, or custom transfer components when their
required graph quantities and projection roles are present. Transfer
components keep named source-term roles separate from reviewed projection
kernels, and `custom_line_of_sight` can project declared source sums through
explicit kernels without hiding unsupported BB or lensing inputs.
`spin2_b_mode` requires a declared `polarization_b` source, and
`line_of_sight_lensing_potential` requires a declared `potential` source. The
manifest layer records the compiled graph summary, background and
recombination provenance, and the selected production CMB execution route so
operators can tell whether a run used backend-standard CAMB perturbations or
the native declared graph. Synthetic runtime-response checks and
reference-backed CMB comparisons both stay inside the normal governed test
suite, while `copernican/validation/` remains the separate
publication-style LCDM reference runner built on the same manifest executor.
The front-door README mirrors that summary so package readers see the same
custom CMB surface from the repository root.

The GUI launches directly from the managed `.venv`, and each run keeps
its own run logs inside the generated `~/copernican_output/copernican-run_*`
folder.

The model chooser now offers `Load model...` so any valid `.yml` or
`.yaml` file can load by exact path in the CLI or GUI.

The engine modules dropped the `cosmo_` prefix, so the docs and package
surface now refer to `engine_*` files. The MCMC initializer uses a
tolerance cutoff for tiny singular values to keep walker startup stable
across platforms.

`copernican/workflow.py` owns the launch flow for the package entry
points, and `copernican/lib/global_settings/defaults.yml` carries the
GUI-facing defaults that shape that flow through
`copernican/lib/settings.py`.

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
`copernican/lib/global_settings/defaults.yml` sit behind those
front-door docs, so changes in the launch path or GUI defaults should be
reflected there first through `copernican/lib/settings.py`.

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
