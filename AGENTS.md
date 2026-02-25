# Copernican Suite Development Guide
**Last Updated:** 2025-12-14

Development notes were previously kept at the top of this file. That history
now lives in `CHANGELOG.md`. New modifications must update the changelog, and
legacy `dev_note` headers embedded in source files have been fully phased out.


## 1. Program Overview
The helper modules previously stored under `scripts/` now live in the
`copernican_lib/` package. CLI-specific helpers sit under `copernican_lib/cli/`
so dependency validation and menu rendering stay modular while the launcher
loads only the lightest prerequisites at startup. The dependency scanner now
skips relative imports inside bundled packages so Copernican's own likelihood
helpers never trigger false missing module reports. The suite evaluates
cosmological models against SNe Ia, BAO and CMB data. Support for additional
observations such as gravitational-wave standard sirens is being prepared
alongside ongoing placeholder management. Users interact with `copernican.py`,
choose a model from `./models/`, pick a computational engine from `./engines/`
and choose data sources. Parsers reside alongside their data but are imported
only when their SHA256 digest matches a vetted list to block untrusted files.
Results are saved under `./output/`, each run in a dedicated `copernican-
run_YYYYMMDD_HHMMSS` subdirectory. Each plot carries a centered footer with
three lines: the model comparison, dataset details and the citation. The first
and third lines are bold, while the dataset name on the second line is bolded
using Matplotlib's standard text rendering. Dataset names retain their original
spacing and the second line wraps after 190 characters when necessary. During
the current refactor the suite is forward-only: legacy staged menus and
backward-compatibility shims are intentionally absent to keep the interactive
shell lean while the GUI evolves. Avoid reintroducing fallbacks unless the
roadmap later requests them explicitly. GUI launchers should use the
`copernican_lib/orchestration` service map rather than duplicating CLI helpers;
the `copernican.py --gui` shim lists the available modules. The staged menu has
been retired—there is no `COPERNICAN_ENABLE_STAGED_MENU` flag or `--enable-
legacy-stage-menu` CLI option—so the Tkinter scaffold under
`copernican_lib/gui/` preserves the navigation rail, Run Builder and monitor
shells even when the renderer falls back to headless mode for automated
environments. The root window now displays the version from
`copernican_lib/VERSION`, Home quick actions open the output directory, Run
Builder and Run Monitor, and the builder itself walks through seed, model,
data, engine and plan steps with live selectors connected to the refreshed
catalogues. Models and datasets limit the operator to a single selection per
panel, with data wired into separate SNe, BAO and CMB menus so each choice
stays confined to its type. A new Run Settings panel captures walkers, burn-in,
production and pool-size hints before a run is started. Data, Models and
Engines panes render scrollable catalogues with working folder, metadata and
parser revalidation buttons, Settings exposes diagnostics filters, output-
directory helpers and environment hints, and Help renders `README.md` (banner
and all) inside a scrollable text widget so the documentation is available
without leaving the GUI.
Full details live in `docs/gui_overview.md`.
The Save Manifest page only enables once the seed, model, dataset and engine
panels hold selections; saving writes the active manifest to
`output/copernican_run_NEW_CONFIG/run_manifest_NEW_CONFIG.yml`, updates the
summary metadata and unlocks the confirmation step. Starting the run renames
that workspace to `copernican-run_<timestamp>` so the CLI worker always reads
the timestamped manifest while Cancel/Clear removes the temporary folder so no
drafts linger. Metadata/YAML dialogs size themselves to the longest line, add
an **Open file…** action that launches the source asset in the OS default
editor, and Start Run now delegates to `copernican_lib.gui.run_worker`,
which invokes the real CLI workflow in a child process using the builder
selections.
The worker’s stdout/stderr feed the diagnostics pane while
Cancel and Hard Stop terminate the child so runs remain interruptible from the
GUI.
CLI manifest launches now share the same
`copernican_lib.run_executor.execute_run_from_manifest` helper as the GUI so
the orchestration logic stays centralized and the manifest runner can be reused
by other frontends without duplicating the workflow.
The navigation rail itself
now reserves 240 px so a padded Copernican logo square rendered from
`img/logogui.png` sits above the Home button with equal spacing to the
surrounding chrome before the other navigation controls begin. The icon now
shares a 60 px square to stay centered with the lighter left/top padding.
`copernican.py` now accepts `--gui`, `--cli` and `--no-gui` flags plus
`--manifest` and `--output-dir` overrides so CI can direct manifests to
deterministic paths. GUI invocations detach automatically (``pythonw`` on
Windows, `nohup` on Unix) so terminals close once the handoff completes; the
start launchers must preserve that behaviour and defer to the shared launcher
instead of reviving legacy menu stacks. Each run directory also includes a
`run_manifest_*.yml` file listing the selected models, engine, dataset hashes
and Git state to aid reproducibility. The data loaders compute and log SHA256
digests for all non-parser files in each dataset directory and store these
hashes on the returned DataFrames. The manifest copies this mapping verbatim.
Parsers must register under the `dataset_id` stated in their metadata so the
loaders can locate them directly without discovery.

Posterior NetCDF files store provenance on both the inference-data root and
inside the posterior group so callers opening only the posterior dataset still
recover the model name, dataset identifier and other metadata without reading
the top-level attributes.

Stage 5 now tolerates legacy corner-plot validators that only return flattened
samples and labels. Custom tooling should adopt the newer three-value signature
so thinning statistics remain explicit, but the fallback keeps archival plugins
functional while developers migrate. Version 7.4.4 retains
``_validate_corner_inputs`` as a shim over the canonical
``_prepare_corner_inputs`` helper so the legacy import path stays alive without
triggering linter redefinition warnings.

Corner plots must now obey the deepened dual-clearance policy introduced in
7.6.8. The layout helper enforces both a fixed padding between the axes and
footer, keeps the lowest footer line above a dedicated clearance floor and
raises the grid so no combination of footer lines can overlap the axes.  The
Stage 5 suptitle now sits lower to mirror the rest of the plotting suite.
Contributors tweaking Stage 5 visuals should keep the `_CORNER_FOOTER_PADDING`
and `_CORNER_FOOTER_CLEARANCE` constants in their tests and update the shared
documentation whenever the guard bands or title anchor move.

A ``COPERNICAN_SEED`` environment variable overrides the interactive seed
prompt.
When unset, the program asks users to accept the default ``0``, enter their own
value or generate a random seed.
The final choice is stored in the run manifest and logged so analyses can be
reproduced.
The launcher keeps a blank spacer after logging
initialisation—replacing the retired "Copernican has initialised" banner—so
the Stage 1 configuration menu aligns with historical spacing without
repeating redundant text.
GUI users can also forge deterministic seeds via the mini-games described under
`rng_minigames/`.
The top-level README covers the embedding API, while each folder (Emoji
Meteors, Constellation, Alien Invasion, etc.) contains its own README with
gameplay notes, accessibility tips and configuration files.
Because the project is
vendorable, contributors must keep those README files and
`rng_minigames/CHANGELOG.md` current whenever they add content, tune the
autopilot or edit the visuals.
The Copernican README/AGENTS entries now only reference the bundle, so all
authoritative documentation lives beside the code. Alien Invasion’s autopilot
(**Let AI take care**, **Let AI learn**, **Let AI forget**) and modal controls
(**Pause/Resume**, Hall of Fame) are documented in
`rng_minigames/alien_invasion/README.md`.

The program enables Python's ``faulthandler`` at startup and registers
``SIGILL``, ``SIGSEGV`` and ``SIGFPE`` handlers. When triggered, they dump
stack traces to both the console and the active log file before exiting.
Immediately after logging initialises the suite records the Python version,
operating system, CPU model and key package versions. A short summary is shown
on the console while the log captures full details. Progress messages print to
``stdout`` and flush on every update so lengthy optimisations still display
activity on Linux terminals.

All Python warnings are forwarded to the central logger. Set
``COPERNICAN_STRICT_WARNINGS=1`` to elevate warnings to errors during CI runs.

Before any heavy computation, a tiny NumPy/SciPy calculation checks that the
installed binaries match the CPU. If this fails the log explains possible CPU
feature mismatches and suggests reinstalling with suitable wheels.

The default engine is `engines/cosmo_engine_mcmc.py`. Model plugins are now
constructed via `copernican_lib.plugins.build_engine_plugin` which produces a
picklable dataclass describing bounds, priors, transforms and dataset
compatibility. Posterior evaluation is handled by
`copernican_lib.posterior.make_logposterior`, ensuring every engine shares the
same prior, transform and bounds logic while remaining multiprocessing safe.
The BAO χ² helper accepts pre-extracted arrays so callers can convert data
frames once outside optimisation loops. Joint likelihoods use
`copernican_lib.likelihoods.JointLike` so Stage 2 evaluates SNe, BAO and CMB
data simultaneously, recording per-dataset χ² values in the sampler output.
When both models reference the same YAML file the Stage 2 workflow compares
`MODEL_FILENAME` values and reuses the initial posterior so BAO and CMB
overlays align exactly during ΛCDM self-consistency checks. The engine emits
step-by-step counter summaries (e.g., “Burn-in stage batch 1: 3/200 steps
completed (1%)”) for both burn-in and production phases, displays percentage
indicators and continues to return ``-np.inf`` whenever a proposal falls
outside declared parameter bounds or yields a non-finite chi-squared so the
sampler rejects invalid walkers deterministically.

Version 7.6.3 removes the retired runtime estimator entirely.
Stage 2 now streams per-walker updates into the fifty-character progress bars
so operators see continuous movement without speculative timing extrapolations.
The release also reiterates that ArviZ remains a hard dependency: convergence
diagnostics must succeed for every batch, and provisioning fails fast when the
package is missing.
When both model plugins resolve to the same YAML file the helper reuses the
ΛCDM measurement directly instead of executing the alternative branch a second
time.

Version 7.6.11 raised that standard further by routing Stage 2 progress
through `tqdm`.
Version 7.6.12 locked the smooth animation in place by disabling the
library's adaptive throttling and mirroring the Unicode glyphs inside the live
display.
Version 7.6.13 added a dedicated walker-progress meter plus an animated
spinner, and Version 7.6.14 retires `tqdm` entirely in favour of a native
carriage-return renderer so macOS, Linux and Windows terminals repaint every
walker update on a single console line while the logged Unicode glyphs remain
identical to the interactive output.
Recent refactors now replace that renderer with line-based counter updates,
keeping the same ``batch_start``, ``progress_update`` and ``batch_finish``
records for the GUI while eliminating the spinner pump and carriage-return
artifacts from the console logs.
the spinner pump and carriage-return artifacts from the console logs.

Version 7.1.1 standardises every runtime timestamp on Coordinated Universal
Time (UTC) so log files, manifests and output directories match across
developer machines and CI runners. Version 7.1.0 adds an interactive Stage 2
sampler menu. After the CMB dataset loads, the launcher now prompts for
production steps, burn-in length, walker count and multiprocessing pool size,
suggesting minimum values derived from the selected models. The chosen
configuration is logged and written to the parameter summary files so trimmed
exploratory runs and CPU-optimised batches remain auditable. Version 7.3.1
builds on that interaction by replacing terse confirmation and shutdown
questions with numbered menus that spell out when an operator is accepting a
plan, restarting the questionnaire, returning to the defaults summary or
exiting the suite entirely. Parameter priors must now declare an explicit
`type`; legacy `distribution` aliases are rejected.  The parser canonicalises
every mapping, injects `type: fixed` for bounds whose endpoints coincide and
surfaces deterministic constants via `plugin.FIXED_PARAMS` so downstream
utilities can consume them without re-reading the YAML cache. Non-ΛCDM sample
YAMLs (`cosmo_model_cfsc.yml`, `cosmo_model_cpc.yml`, `cosmo_model_qauc.yml`
and `cosmo_model_usmf{3..7}.yml`) are maintained as parseable exemplars. Treat
their `python_var` assignments and folded block scalars as the canonical style
when authoring new theories. After burn-in any walkers that drift into ``nan``
coordinates are reseeded around the ensemble mean with progressively smaller
jitter, eliminating the `RuntimeWarning: invalid value encountered in scalar
subtract` messages that appeared in archived logs. Starting with version 1.11.4
the test suite no longer runs automatically. Launchers offer a *Run the unit
test suite* option which delegates to `python -m unittest discover` and exits
cleanly even when Matplotlib has not yet been imported.

## 2. Directory Layout
```
models/           - YAML model definitions with embedded theory text and
                    equations.
engines/          - Computational backends (SciPy CPU by default)
data/             - Observation files under ``data/<type>/<source>/``. Each
                    dataset directory includes a `metadata_*.yml` file with
                    `dataset_name`, `description`, `citation`, `license`, the
                    full `author` list and BibTeX keys such as `title`,
                    `volume`, `journal` and `DOI`. Metadata is loaded
                    exclusively by
                    `copernican_lib/dataset_registry.py` after each parser
                    runs.
  cmb/planck2018lite/ - Planck 2018 lite TT/TE/EE spectra and covariance
output/           - Per-run folders with plots, tables and NetCDF chains
AGENTS.md         - Development specification and contributor rules
CHANGELOG.md      - Release history
copernican_lib/optim_utils.py - Shared optimisation helpers used by engines
copernican_lib/latex_utils.py - LaTeX translation helpers using
                                latex_mappings.yml
copernican_lib/console_output.py - Central console output helpers
```
Installing the suite with `pip` produces a `copernican_suite.egg-info`
directory containing build metadata. This folder can be safely removed and
should not be edited manually. The `/data` directory is read-only. Preserve
bundled datasets, parsers and metadata files exactly as released unless a human
request explicitly calls for an update.

The current plotting style and algorithms are considered stable. Do not alter
them without explicit instruction.

Multiprocessing is enabled across several engines. To guarantee a clean
environment for each worker, the program sets Python's multiprocessing start
method to `spawn` at entry. Model YAML files are validated via `jsonschema`
**only** in the main process; child processes read the sanitized cache without
re-validating to prevent occasional plugin failures under multiprocessing.

All engines should remain purely computational. Shared utilities such as
evaluation counters now live in ``copernican_lib/optim_utils.py`` and are
imported by the engines instead of being reimplemented inside each backend.

The ``_eval_safe`` helper in ``engine_plugin_validation`` caps recursion depth
and AST node count when parsing expressions for ``get_camb_params`` to block
runaway evaluation on malicious or overly complex inputs.

## 3. Dependency Installation
`copernican.py` scans all project files for imported modules using Python's AST
parser to avoid false positives from comments. It writes the discovered imports
to `.cache/dependency_scan.json` alongside the absolute path, size and
modification time of every parsed file. When those details match a previous run
the check loads the cached list immediately, keeping the menu snappy even on
large worktrees. The `.cache/` directory is created on demand and must remain
untracked so contributors keep private dependency metadata. Set
`COPERNICAN_DEP_CACHE_DIR` to direct the cache to a custom location when the
default path is read-only. The `start.*` launchers always download a private
Python 3.11 interpreter into ``.python`` and build ``.venv`` from that
interpreter, ignoring any system-wide Python. If the download fails they exit
with guidance. When packages are missing the program now fails fast and
instructs the operator to rerun the launcher to rebuild the managed environment
instead of invoking `pip` from inside ``copernican.py``. Running outside
``.venv`` prompts the user to restart via the appropriate launcher. This
lightweight approach works across Windows, macOS and Linux while allowing new
engines to introduce additional dependencies without manual updates to the
documentation. The launchers delete bundled interpreters that fall outside the
Python 3.11 series, recreate `.venv` when its Python drifts beyond that window
and print a notice before invoking `sudo`, `brew` or `winget` so users know any
password prompt originates from the package manager and is never read or
stored. `sudo -k` and explicit prompts ensure the operating system handles all
credential entry. ArviZ ships as the released `0.22.0` build, which already
supports NumPy 2 so the dependency set no longer relies on a pinned commit
archive.

`requirements.lock` pins exact versions for all runtime packages, and
`[project].dependencies` in `pyproject.toml` mirrors these pins. Any dependency
change must regenerate both files and update `THIRD_PARTY_LICENSES.md` to keep
license records current (see law 22). To install the suite as a package, run
`pip install .` at the repository root. Use `pip install -e .` if you intend to
develop the code. The start scripts install pinned dependencies from
`requirements.lock` before running `pip install --no-deps .`. They delete any
`build/` directory before and after installing the project to prevent stale
build artifacts and recreate `.venv` once when the activation script is
missing.

Pull requests run a GitHub Actions workflow named ``Tests`` that executes pre-
commit checks and the full unit suite on Ubuntu, macOS and Windows. Only pull
requests trigger the workflow to avoid duplicate push builds.

## 4. YAML Model System
As of version 2.0 every cosmological model is described by a single YAML file
`cosmo_model_*.yml`. All theory text, equations and parameters reside in this
file. Markdown files may mirror the YAML for readability, but models are
distributed only as YAML. No permanent Python plugins exist in the repository.
Models are automatically discovered by scanning for `cosmo_model_*.yml` files
in the `models/` directory. All expressions inside these YAML files must be
written in LaTeX math form; raw Python code is not permitted.

### 4.1 YAML Model File
The schema requires `model_name`, `version`, `parameters`, `equations`,
`abstract` and `description`. Optional fields such as `unit` and `latex_name`
provide additional context. `copernican_lib/model_spec_validator.py` validates
the YAML and writes a sanitized copy to `models/cache/`.
`copernican_lib/model_coder.py` transforms the equations into NumPy callables.
These callables are validated by `copernican_lib/engine_plugin_validation.py`
before being passed to the chosen engine. `model_spec_validator.py` ignores
unrecognized keys and copies them to the cache, so new metadata can be added
without breaking older YAML files.

Treat the `description` block as the journal article for the theory. Write at
least ten pages of Markdown and LaTeX covering assumptions, derivations,
observational comparisons, parameter motivation and reproducibility guidance.
When a human or AI contributor revises a YAML model, increment the internal
`version` field even if Copernican's overall release version does not change.
Only `models/cosmo_model_lcdm.yml` is required for the suite to run; all other
models ship as exemplars and may evolve or be replaced as their manuscripts
improve.

## 5. Creating a New Model
1. Copy an existing `cosmo_model_*.yml` file such as `cosmo_model_lcdm.yml`.
2. Edit the YAML fields to describe your model, following the schema above.
3. *(Optional)* Create a Markdown file with the same base name if you want a
   human-readable summary. The YAML file remains the single source of truth.
   See `cosmo_model_template.yml` for a complete template.

### 5.1 YAML Template
Use the following structure when creating new models:

```yaml
model_name: My Model
version: "1.0"
parameters:
  - name: H0
    bounds: [50, 100]
    definition: N/A
    latex_name: H_0
equations:
  sne:
    - "d_L(z) = (1+z) \int_0^z \frac{c\,dz'}{H(z')}"
    - "\mu(z) = 5\log_{10}[d_L(z)/{\rm Mpc}] + 25"
  bao:
    - "D_M(z) = \int_0^z \frac{c}{H(z')}"
    - "D_H(z) = \frac{c}{H(z)}"
    - "D_V(z) = [D_M(z)^2 D_H(z)]^{1/3}"
```
Initial guesses are computed automatically as the midpoint of each parameter's
bounds.

`model_spec_validator.py` and `model_coder.py` handle validation and code
generation automatically; no manual Python implementation is required. The
parser keeps unknown keys intact, ensuring the DSL stays backward compatible as
new fields are introduced.

Models that advertise BAO support must now declare an explicit
``rs_expression``. The former numerical fallback has been removed because it
double-counted photon densities whenever the model's ``H(z)`` already included
radiation terms. Tests must cover every new integral to confirm the provided
sound horizon matches the declared background dynamics.

### 4.2 Dataset compatibility flags

Generated model plugins include boolean attributes
`valid_for_distance_metrics`, `valid_for_bao` and `valid_for_cmb`. All default
to `True` and signal which datasets the model supports. When `valid_for_cmb` is
`False` the engine does not require the optional `compute_cmb_spectrum`
function during validation. Models that can compute a CMB power spectrum should
also define a `cmb.param_map` object describing how standard CAMB parameters
such as `H0` and `ombh2` are derived from the model's variables or constants.

# DEVELOPMENT POLICY (DevCovenant and Laws)

**IMPORTANT: READ FROM HERE TO THE END OF THE DOCUMENT AT THE BEGINNING OF
EVERY DEVELOPMENT SESSION**

**Copernican development must run inside the repository's `.venv`.** Always
launch via `start.sh` / `start.command` / `start.bat` so the managed
environment is active before editing code or running tests.

The suite uses **DevCovenant**, a self-enforcing policy system that keeps the
plain-English rules below synchronized with automated checks. Whenever a policy
block changes, set `updated: true`, update the corresponding script in
`devcovenant/policy_scripts/`, add/adjust tests, and clear the flag once the
hashes are refreshed.

Even with DevCovenant enforcing most rules automatically, contributors must
keep the following expectations in mind:

1. **`AGENTS.md` is the canonical source of truth.** Other docs should link
   here rather than duplicating the rules, and edits to this section require
   explicit human approval.
2. **Re-read these expectations at the start of every session.** Staying
   aligned with the latest guidance is part of the workflow.
3. **Preserve human-authored edits.** Respect the structure, tone and metadata
   of human contributions unless you are correcting an objective mistake or a
   human explicitly asks for a rewrite.

Following these expectations, DevCovenant's policies and all good coding and
documentation practices is not optional; it is essential for the long-term
viability and success of the Copernican Suite. Failure to follow these rules
will compromise the maintainability of the Copernican Suite!!!

## DEVCOVENANT DEVELOPMENT POLICY MANAGEMENT AND ENFORCEMENT SYSTEM

### How DevCovenant Works

1. **Policies are defined here** in plain English with machine-readable
   metadata
2. **Python scripts check compliance** automatically during development
3. **Hash verification ensures sync** between policy text and implementation
4. **AI maintains the scripts** when policies are updated
5. **Pre-commit hooks enforce** policies before code is committed

### DevCovenant Ignore List

Repository-wide exclusions now live under `ignore.patterns` inside
`devcovenant/config.yaml`. Each entry accepts gitignore-style globbing (e.g.,
`copernican_lib/vendor/**`, `tests/**`, `**/tests/**`) and automatically feeds
the engine so policy scripts can skip those paths without duplicating
hard-coded lists. Adjust the configuration whenever a new path must be ignored
for every policy; per-policy overrides still live with their specific metadata.

### When to Run DevCovenant (AI Agents)

**CRITICAL**: AI agents must run DevCovenant at specific checkpoints:

#### 1. **At the START of Every Work Session** (REQUIRED)

**Before beginning any work on the repository**, immediately run:
`pre-commit run --all-files` followed by `python tools/run_tests.py`.

DevCovenant runs as part of pre-commit. The repository should pass the other
pre-commit hooks anyway if everything has been clean on last commit.

Running the full hook suite and tests at the beginning of work ensures:
- All policies are synchronized with their implementation scripts
- Any updated policies trigger immediate script updates
- The AI is aware of all current policies before proceeding
- The end-to-end test suite exercises the current state before changes begin

**What happens:**
- DevCovenant parses all policy definitions from this file
- Checks for hash mismatches (policy text updated but script hasn't been)
- Reports sync issues with clear, actionable instructions
- **AI MUST update any out-of-sync scripts BEFORE proceeding with user's
  request**

**Example workflow:**
```bash
# 1. AI starts work session
$ git status
$ cat AGENTS.md  # Read policies (standard practice)
$ python devcovenant_check.py check --mode=startup

# 2. If sync issues detected:
🔄 POLICY SYNC REQUIRED

Policy 'changelog-coverage' has been updated.
The policy script is out of sync and must be updated FIRST.

[Policy text and instructions shown]

# 3. AI updates the script
$ vi devcovenant/policy_scripts/changelog-coverage.py
$ vi devcovenant/tests/test_policies/test_changelog-coverage.py
$ pytest devcovenant/tests/test_policies/test_changelog-coverage.py -v

# 4. Re-run to verify and update hash
$ python devcovenant_check.py check --mode=startup

# 5. Now proceed with user's request
✅ All policies are in sync!
```

#### 2. **Before Committing Code** (AUTOMATIC)

DevCovenant runs automatically via pre-commit hook. AI can also run manually:

```bash
python devcovenant_check.py check --mode=pre-commit
```

This checks only changed files for faster performance.

#### 3. **At the END of a Work Session** (RECOMMENDED)

**Before finishing work**, run a full check:

```bash
python devcovenant_check.py check --mode=lint
```

This performs comprehensive validation of all files to ensure nothing was
missed.

#### 4. **When Updating a Policy** (REQUIRED)

After editing a policy in this file:

1. Set `updated: true` in the policy metadata
2. Run `python devcovenant_check.py check --mode=startup`
3. Follow instructions to update the corresponding script
4. Update tests
5. Run tests: `pytest devcovenant/tests/test_policies/test_<policy_id>.py -v`
6. Re-run DevCovenant (hash updates automatically, `updated` flag clears)

**Important Notes:**
- DevCovenant violations **MUST** be fixed before commit
- Policy sync issues have **HIGHEST PRIORITY** - fix before user's request
- Never skip DevCovenant checks to "save time"
- Read all violation messages carefully - they guide you to the fix
- Use `--fix` flag to auto-fix when available: `python devcovenant_check.py
  check --fix`

#### 5. **Before responding to ANY user request** (REQUIRED)

Re-run `pre-commit run --all-files` **and** `python tools/run_tests.py`
immediately before crafting the reply, even if the user only asked a question
or the change touched documentation only. This full sweep re-runs
DevCovenant across the entire repository and exercises the entire test suite so
global policies (e.g., `no-future-dates`) re-check historical files and tests
cannot silently regress.

### Policy Format

Each policy has a `policy-def` block with these flags:

- **id**: Unique identifier (lowercase-with-hyphens)
- **status**: `new`, `active`, `updated`, `deprecated`, or `deleted`
- **severity**: `critical` (blocks always), `error` (blocks at error
  threshold), `warning` (blocks at warning threshold), or `info` (informational
  only)
- **auto_fix**: `true` if automatic fixing is available, `false` otherwise
- **updated**: `true` when policy text changes (triggers AI script update)
- **applies_to**: File patterns (optional, e.g., `*.py`, `devcovenant/**/*`)
- **hash**: Automatically maintained hash of policy + script
- Additional `key: value` rows inside the block become policy-specific options
  that are exposed to scripts via `PolicyCheck.get_option()`. Use them for
  knobs such as file suffixes, directory allowlists and required commands so
  scopes stay declarative.

Selectors in these metadata blocks should prefer the shared naming scheme
(`include_suffixes`, `exclude_prefixes`, `force_include_globs`,
`watch_files`, etc.) documented in `devcovenant/README.md` so contributors can
predict how policies interpret prefixes, suffixes and glob filters.

#### Standard selector metadata

Policies that scope work by path should stick to the canonical selector keys
so developers get the same mental model everywhere:

- `include_prefixes`, `include_suffixes`, `include_globs` — primary filters
  that opt files into the rule.
- `exclude_prefixes`, `exclude_suffixes`, `exclude_globs` — negative filters
  that carve out directories, generated code or vendor trees.
- `force_include_globs` — re-introduces specific globs when they would
  otherwise be excluded (for example, dataset parsers that live under a
  read-only directory).
- `watch_files`, `watch_dirs` — explicit lists of files or directories that
  should always be examined, even if they fall outside the other selectors.
- `protected_globs` / `exempt_globs`, `guarded_paths`, `user_visible_dirs` —
  policy-specific selectors that follow the same gitignore/glob semantics to
  describe read-only, security or documentation scopes.

The shared `SelectorSet` helper in `devcovenant/selectors.py` enforces
`force-include → exclude → include` precedence so each policy inherits the same
matching rules. When adding new selectors, follow this naming style and update
`devcovenant/tests/test_selectors.py` so downstream repositories can reason
about the matcher without reading every policy script.

#### Repository overrides and migrations

Defaults declared in AGENTS.md describe Copernican’s needs, but every selector
can be overridden in `devcovenant/config.yaml`. Place repository-specific
overrides under `policies.<policy-id>` so forks can point the same policy at a
different directory tree without touching any Python. Global exclusions that
used to live in `devcovignore.md` now belong to `ignore.patterns` in
`config.yaml`, and read-only directories—including parser exemptions—are
tracked via `include_globs` / `exclude_globs` metadata instead of ad-hoc text.
Keep AGENTS.md authoritative for the default scopes, and document any local
override in `config.yaml` so the configuration stays self-describing.

### Adopting DevCovenant in other repositories

1. Copy the `devcovenant/` directory into the destination repository and
   install the pre-commit hook (see `devcovenant/README.md#installation`).
2. Point `paths.policy_definitions` in `devcovenant/config.yaml` at the host
   project’s canonical policy document (for example `CONTRIBUTING.md`) and
   adjust `paths.registry_file`, `engine.file_suffixes` and `ignore.patterns`
   to reflect the new layout.
3. Define each policy’s selector metadata directly in that policy’s
   `policy-def` block using the standard keys listed above. These defaults are
   recorded in the documentation so humans understand the scope even before
   reading `config.yaml`.
4. Add any repository-specific tweaks under `policies.<policy-id>` inside
   `config.yaml`. These overrides take precedence over AGENTS.md so forks can
   retarget policies without editing the original prose.
5. Run `pre-commit run --all-files` (or `python devcovenant_check.py check
   --mode=startup`) to regenerate the registry and confirm that the metadata,
   scripts and configuration are in sync. Update hashes/tests whenever policy
   text changes, exactly as described earlier in this document.

Once these steps pass, DevCovenant enforces the new repository with the same
consistency guarantees as Copernican while keeping all knobs declarative.

### Development Policies

## Policy: Changelog Coverage

```policy-def
id: changelog-coverage
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
main_changelog: CHANGELOG.md
collections: rng_minigames/:rng_minigames/CHANGELOG.md:true
```

All changed files must be documented in the appropriate changelog.
Paths that match a prefix listed under `collections` go to the companion log
for that collection, and everything else belongs in `main_changelog`.
Entries listed in `skipped_files` (lockfiles, existing changelog files, etc.)
remain exempt.
Compare `git diff --name-only` against the newest entry in the relevant
changelog before every commit and **explicitly enumerate every changed
file**—the lint hook fails whenever a touched path is missing from the
changelog summary.

---

## Policy: No Git Conflict Markers

```policy-def
id: no-git-conflict-markers
status: active
severity: critical
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
```

Never insert Git conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) in any
file. All merge conflicts must be resolved before committing.

---

## Policy: Line Length Limit

```policy-def
id: line-length-limit
status: active
severity: warning
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
max_length: 79
include_suffixes: .py,.md,.rst,.txt
exclude_prefixes: copernican_lib/vendor,data
force_include_globs: data/*/cosmo_parser_*.py
```

Keep individual lines under 79 characters to maintain readability. This policy
targets the suffixes declared in `include_suffixes`, skips anything under
`exclude_prefixes`, and re-introduces specific globs through
`force_include_globs` even when those paths would otherwise be excluded.
Adjust `max_length` or the metadata lists to match another repository’s
requirements.

---

## Policy: Last Updated Marker Placement

```policy-def
id: last-updated-placement
status: active
severity: warning
auto_fix: true
updated: false
applies_to: *
enforcement: active
waiver: false
```

Refresh documentation and `Last Updated` markers only when editing the
allowlisted files: `AGENTS.md`, `CITATION.cff`, `copernican.py`, `start.sh`,
`start.bat` and `start.command`. Remove these markers from every other surface
and do not introduce new ones elsewhere. When touching an allowlisted file,
update its `Last Updated` marker within the first three lines using an ISO-8601
date with no time component.

---

## Policy: DevCovenant Self-Enforcement

```policy-def
id: devcov-self-enforcement
status: active
severity: error
auto_fix: false
updated: false
applies_to: devcovenant/**/*
enforcement: active
waiver: false
```

DevCovenant enforces its own policies on itself. All policy scripts must:
- Have corresponding tests in `devcovenant/tests/test_policies/`
- Achieve at least 80% code coverage
- Follow the PolicyCheck base class interface
- Include comprehensive docstrings
- Pass all tests before being registered

---

## Policy: DevFlow Run Gates

```policy-def
id: devflow-run-gates
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
test_status_file: devcovenant/test_status.json
required_commands: pytest,python -m unittest discover
code_extensions: .py,.pyi,.rs,.c,.cpp,.h,.hpp
```

Every coding session begins and ends with `pre-commit run --all-files`, and the
session must also record a full `python tools/run_tests.py` invocation after
code edits. The DevFlow gate enforces that cadence by inspecting
`devcovenant/test_status.json`, confirming the recorded commands include every
entry from `required_commands` and asserting the timestamp post-dates the most
recent code change touching any extension listed in `code_extensions`. The
policy metadata exposes those knobs so other repositories can redirect the
status file, tailor the commands or expand the definition of “code” without
editing the policy script.

---

## Policy: Version Synchronization

```policy-def
id: version-sync
status: updated
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
version_file: copernican_lib/VERSION
readme_file: README.md
citation_file: CITATION.cff
pyproject_file: pyproject.toml
runtime_entrypoints: copernican.py
runtime_roots: copernican_lib,engines
```

The project follows Semantic Versioning (`MAJOR.MINOR.PATCH`). Record the
active version in every file listed under `version_file`, `readme_file`,
`citation_file` and `pyproject_file`, and keep those declarations in sync.
Runtime code in the `runtime_entrypoints` and `runtime_roots` must import the
shared helper instead of embedding ad-hoc strings. DevCovenant validates the
canonical string and compares it against the previous commit’s recorded
version. Any mismatch, invalid SemVer format, or non-increasing bump causes a
violation so version numbers never regress or drift from the documented
sources.

---

## Policy: Semantic Version Scope

```policy-def
id: semantic-version-scope
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: true
version_file: copernican_lib/VERSION
changelog_file: CHANGELOG.md
ignored_prefixes: devcovenant,rng_minigames
override_file: .devcovenant/waivers/semantic-version-scope.txt
```

Whenever the file listed under `version_file` changes, the release bump must
follow SemVer’s MAJOR.MINOR.PATCH rules. The newest changelog section (`##
Version X.Y.Z`) must declare at least one `[semver:patch]`, `[semver:minor]` or
`[semver:major]` tag describing the most significant change in that release,
and those tags must reflect the stated scope:

- **Patch** – backwards-compatible bug fix, documentation, or build change.
- **Minor** – backwards-compatible feature, dataset, or CLI enhancement.
- **Major** – any backwards-incompatible change to CLI flows, YAML schemas,
  parsers, manifests, or output formats.

The checker compares the newest version to the previous entry in the metadata-
defined `changelog_file` and fails when the bump is smaller than the tagged
scope. Changes limited to any prefix listed in `ignored_prefixes` do not force
a release and are ignored by this policy. If an extraordinary release needs a
different scope, add `.devcovenant/waivers/semantic-version-scope.txt` (the
default `override_file`) with `override=<level>` (`patch`, `minor` or `major`)
and remove it immediately
after the release. Every release entry must use a single `[semver:*]` scope and
the bump recorded in `copernican_lib/VERSION` must match the declared
scope—logging a scope in the changelog without bumping the version file is
rejected.

---

## Policy: No Future Dates

```policy-def
id: no-future-dates
status: active
severity: error
auto_fix: true
updated: false
applies_to: *
enforcement: active
waiver: false
```

`Last Updated` timestamps and date fields must never extend into the future.
Future dates indicate dating errors or premature commits. Validate every date
against the current day before recording it. Always run `pre-commit run
--all-files` (or `python devcovenant_check.py check --mode=lint`) before
responding so the policy scans the entire changelog, citation metadata and
every other file, catching stale or future timestamps even when those files
were not edited in the current diff.
Running `devcovenant_check.py check --fix` rewrites any offending date to the
current UTC day so mechanical typos are corrected automatically.

---

## Policy: New Modules Need Tests

```policy-def
id: new-modules-need-tests
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
include_prefixes: copernican_lib,engines
include_suffixes: .py
watch_dirs: tests
```

New Python modules under any path listed in `include_prefixes` (and matching
`include_suffixes`) must be accompanied by new or updated tests rooted under
the metadata-defined `watch_dirs`. This prevents untested code
from entering the repository and maintains code quality standards. Tests should
evolve with the code. No code should be tailored to satisfy tests—instead,
tests should follow in unison as the implementation changes. When removing a
module, the tests or parts of tests associated with it should be removed or
amended accordingly. Customize the include/selectors and `watch_dirs` to match
another repository’s layout.

---

## Policy: No Print in Library

```policy-def
id: no-print-in-library
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
include_prefixes: copernican_lib,engines
exclude_prefixes: copernican_lib/vendor
allowed_files: copernican_lib/console_output.py
```

Library and engine code must use the managed console output helper listed in
`allowed_files` instead of bare `print()` calls. This keeps diagnostics
consistent across platforms and properly routes output through the shared
utilities. Paths registered under `exclude_prefixes` remain exempt so vendored
code retains its upstream behavior.

---
## Policy: Read-Only Directories

```policy-def
id: read-only-directories
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
include_globs: data/**
exclude_globs: data/**/cosmo_parser_*.py
```

Read-only paths are defined by the `include_globs` metadata. Paths matching
`exclude_globs` remain editable so code-bearing assets can still be linted and
tested even when they sit under a protected directory. Editing a protected path
now fails immediately so metadata remains the single source of truth for
read-only coverage.

---

## Policy: Docstring and Comment Coverage

```policy-def
id: docstring-and-comment-coverage
status: active
severity: error
auto_fix: false
updated: false
applies_to: *.py
enforcement: active
waiver: false
include_suffixes: .py
exclude_prefixes: copernican_lib/vendor
exclude_globs: tests/**,**/tests/**
```

Every targeted Python module should include a descriptive docstring or an
adjacent explanatory comment for modules, classes and functions.
The checker uses the `all_files` snapshot when available, so every file
matching `include_suffixes` (minus any `exclude_prefixes`/`exclude_globs`)
gets evaluated even before it is staged.
The policy accepts short docstrings or inline comments positioned immediately
before each definition so the team can grow coverage gradually.
Missing documentation now triggers an error-level violation so that gaps in
coverage are addressed promptly.
Running DevCovenant in a non-`pre-commit` mode (e.g., `lint` or `startup`)
virtually inspects *all* matching files so the policy uncovers gaps beyond just
the staged files.
Tune `include_suffixes`, `exclude_prefixes` and `exclude_globs` inside the
policy definition whenever a repository needs different boundaries.
definition whenever a repository needs different boundaries.

---

## Policy: Raw String Escapes

```policy-def
id: raw-string-escapes
status: active
severity: warning
auto_fix: true
updated: false
applies_to: *.py
enforcement: active
waiver: false
include_suffixes: .py
exclude_prefixes: copernican_lib/vendor
exclude_globs: tests/**,**/tests/**
```

String literals inside the selector-defined scope must either be prefixed with
`r` or use explicit escape sequences for each backslash. The policy scans every
matching file (respecting the global DevCovenant ignore list) and warns when a
bare backslash appears without being part of a known escape sequence,
encouraging raw strings or double-escaped paths. The severity is warning-level
so commits block until the literal is fixed or intentionally waived.
Auto-fix mode doubles the offending backslashes so the literal is safe to
commit without hand-editing.

---

## Policy: Start Script Parity

```policy-def
id: start-script-parity
status: active
severity: error
auto_fix: true
updated: false
applies_to: start.sh,start.command,start.bat
enforcement: active
waiver: false
watch_files: start.sh,start.command,start.bat
```

Changes touching any launcher listed in `watch_files` must consider the others.
If one launcher is updated while its siblings remain untouched, the policy
raises an error reminding the maintainer to mirror the changes so the GUI
handoff remains consistent across platforms.
Auto-fix mirrors the edited launcher across the remaining start scripts so
platform shims stay synchronized automatically.

---

## Policy: Name Clarity

```policy-def
id: name-clarity
status: active
severity: warning
auto_fix: false
updated: false
applies_to: *.py
enforcement: active
waiver: false
```

New Python symbols should avoid placeholder or overly short names (e.g., `foo`,
`tmp`, `var`, `data`). The check scans the staged files (and all files during
`lint`/`startup`) and reminds authors whenever an identifier is either
blacklisted or shorter than three characters outside conventional loop
counters; add a `# name-clarity: allow` comment to suppress intentional
exceptions. Paths excluded via `ignore.patterns` in `config.yaml` are skipped
so vendored code keeps its original identifiers.

---

## Policy: Dependency License Sync

```policy-def
id: dependency-license-sync
status: active
severity: error
auto_fix: true
updated: false
applies_to: *
enforcement: active
waiver: false
dependency_files: requirements.in,requirements.lock,pyproject.toml
third_party_file: THIRD_PARTY_LICENSES.md
licenses_dir: licenses
report_heading: ## License Report
```

Every dependency addition, removal or version change must update the files
listed under `dependency_files`, refresh the aggregated table in
`third_party_file`, and ensure the `licenses_dir/` tree includes the refreshed
license texts. The `report_heading` section inside `third_party_file` must
mention each modified dependency manifest so reviewers can trace the changes.
The policy inspects the tracked dependency inputs, the license table, and the
license directory so CI always captures the dependency list and any new license
obligations in lockstep.
Running `--fix` appends a dated entry to the license report for each changed
dependency manifest and drops a marker inside `licenses/` so the directory and
table both reflect the refresh.

---

## Policy: Documentation Growth Tracking

```policy-def
id: documentation-growth-tracking
status: active
severity: info
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
```

When files inside the metadata-defined `user_visible_dirs` or individual
`user_visible_files` change, the documentation corpus must “strictly grow”
by adding a new paragraph, subsection or example that explains the updated
behavior, workflow or configuration.
This active info-level reminder simply surfaces the policy text and points
editors at the relevant docs so they remember to expand the prose before we
raise the severity.

---

## Policy: Security Compliance Notes

```policy-def
id: security-compliance-notes
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
```

Security-critical files listed under `guarded_paths` (launchers, security docs,
etc.) must be accompanied by a short entry in `log_path` describing the
compliance or mitigation rationale. The policy ensures the security log grows
whenever those guarded files change so reviewers can confirm the latest
requirements were considered before a change lands. When `log_path` is
untouched while a guarded file changes, DevCovenant blocks the commit and
requests the missing entry.

---

## Policy: Security Scanner

```policy-def
id: security-scanner
status: active
severity: error
auto_fix: false
updated: false
applies_to: *.py
enforcement: active
waiver: false
```

Automated scanning flags known compliance risks—`eval`, `exec`,
`pickle.loads` or
`subprocess.run(..., shell=True)`—inside repository modules.
The policy runs on every Python file (excluding tests/vendor code) and halts
the commit with a clear message whenever a risky construct is found, nudging
contributors to pick a safer implementation before the change lands.

---

## Policy: Start Script Guardrails

```policy-def
id: start-script-guardrails
status: active
severity: error
auto_fix: true
updated: false
applies_to: start.sh,start.command,start.bat
enforcement: active
waiver: false
```

Every launcher keeps essential guardrails intact. Each entry in the `scripts`
metadata lists the file path and the required snippets (separated by `|`). If
any script loses a required snippet, the policy raises an error so the security
guidance cannot disappear silently.
When the fixer runs, it injects the canonical guardrail snippets (including
`pkg_notice` and the `sudo -k` helper) or reintroduces the `PKG_NOTICE`
definition in `start.bat`, keeping every launcher aligned automatically.

---

## Policy: Test Status Tracking

```policy-def
id: test-status-tracking
status: active
severity: error
auto_fix: false
updated: false
applies_to: devcovenant/test_status.json
enforcement: active
waiver: false
test_status_file: devcovenant/test_status.json
```

Every commit that touches Copernican’s code, tests or orchestration helpers
must record the most recent full-suite run in `devcovenant/test_status.json`.
Run `python tools/run_tests.py` so the default suites (`pytest` followed by
`python -m unittest discover`) execute and `tools/update_test_status.py`
records the UTC timestamp, git SHA and executed command.
When DevCovenant sees relevant files change without a corresponding status
update, the policy blocks the commit so reviewers always know when the suite
last passed. The lists under `watch_dirs` and `watch_files` control which paths
trigger the reminder.

---

## Policy: Managed Virtual Environment

```policy-def
id: managed-venv
status: active
severity: error
auto_fix: false
updated: false
applies_to: *
enforcement: active
waiver: false
expected_virtualenvs: .venv
```

DevCovenant verifies it is running from the repository’s managed virtual
environment. If the current interpreter or `VIRTUAL_ENV` does not resolve
inside one of the entries listed under `expected_virtualenvs`, the policy fails
with instructions to relaunch via the appropriate start script. This prevents
accidental edits from system interpreters and keeps the dependency set
consistent across contributors.
