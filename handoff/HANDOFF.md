# CCMBS handoff

## Purpose and current status

This is a temporary transfer package for the ongoing Slice Seven work in
`PLAN.md`. The human authorized the experimental modified CAMB source in
`handoff/` for this transfer. Copernican's installed CAMB dependency remains
stock. Use the experimental copy only from a temporary directory and only
for explicitly input-matched independent reference work.

Slice Seven is **in progress**. A green `gate --verify`, focused tests and a
completion preflight passed on the previous machine. The preflight did not
solve a CCMBS spectrum. The full `devcovenant run` has not passed for these
changes, and full-window matched CCMBS–CAMB scientific acceptance remains
open. The human runs `devcovenant run` unless they instruct otherwise.

The checked evidence includes stock-CAMB local PPF history relations across
phantom crossing; 105 focused contract, reference and acceptance tests; and
bounded wCDM and w0wa surface solves. In one bounded wCDM comparison, the
batched and BDF surface-peak-normalized differences were at most 3.8e-5 in
TT, 8.2e-6 in TE and 2.7e-6 in EE. These checks do not certify full spectra,
curved-universe parity or other theories. See Slice Seven in `PLAN.md` for
recorded conditions, reference provenance and the outstanding work.

## Package contents

`tmp/ccmbs-reference-handoff.tar.gz` contains:

- The experimental CAMB 1.6.0 source and the forutils source it used,
  including their licenses. Build products and the previous machine's
  compiled libraries are excluded.
- The original CAMB and forutils source archives named in the fixtures.
- The exact PPF reference generators at the archive root and the earlier
  no-stress generators under `ccmbs-smooth-reference-scripts/`.
- The original coarse and fine recombination and reionization input tables
  for the nine current experimental reference cases, plus assembly scripts.

The committed reference fixture retains candidate spectra, numerical input
arrays and provenance. The archived raw tables retain the byte identities
recorded for the original reference generations. The experimental source,
generator scripts and source archives match the recorded source hashes.
The archive itself has a separate checksum in `tmp/SHA256SUMS`.

## Prepare the new machine

1. Check out the pushed branch and read `AGENTS.md`, `PLAN.md`, and the
   repository workflow. Do not assume the previous machine's local gate,
   virtual environment or temporary directory was transferred by Git.
2. Follow the corrected Python 3.11 bootstrap, virtual environment and
   locked-dependency instructions in `README.md`. Activate `.venv` before
   tests or DevCovenant commands. Verify that the normal import is stock:

   ```sh
   source .venv/bin/activate
   python -c 'import camb; print(camb.__version__, camb.__file__)'
   ```

   This should resolve inside `.venv`, not the experimental source tree.
3. Provide the macOS command-line developer tools, `make` and `gfortran`.
   Check `xcode-select -p`, `make --version` and `gfortran --version` first.
   On a machine with Homebrew, `brew install gcc` supplies GNU Fortran.
   If Homebrew, developer tools, permissions or dependencies are missing,
   resolve them with the human as needed; report installation failures and
   ask for missing authorization rather than silently substituting tools.
4. From the repository root, verify and extract the transfer archive to a
   fresh temporary directory. Keep the printed directory for this session:

   ```sh
   (cd handoff/tmp && shasum -a 256 -c SHA256SUMS)
   CCMBS_HANDOFF_DIR=$(mktemp -d "${TMPDIR:-/tmp}/ccmbs-handoff.XXXXXX")
   export CCMBS_HANDOFF_DIR
   tar -xzf handoff/tmp/ccmbs-reference-handoff.tar.gz \
       -C "$CCMBS_HANDOFF_DIR"
   printf '%s\n' "$CCMBS_HANDOFF_DIR"
   ```

   Do not overwrite an existing temporary CAMB tree. The temporary directory
   may be cleared by the operating system; the archive is the transfer copy.
5. Check the extracted source, both generator sets and all current raw input
   tables against the committed reference fixture. Run this from the repo
   root with `.venv` active:

   ```sh
   python - <<'PY'
   import gzip
   import hashlib
   import json
   import os
   from pathlib import Path

   root = Path(os.environ["CCMBS_HANDOFF_DIR"])
   path = Path("tests/project/fixtures/ccmbs_comparable_camb.json.gz")
   fixture = json.loads(gzip.decompress(path.read_bytes()))

   def check(path, expected):
       actual = hashlib.sha256(path.read_bytes()).hexdigest()
       if actual != expected:
           raise ValueError(f"Hash mismatch: {path}")

   count = 0
   for group in ("references", "superseded_experimental_references"):
       for case, reference in fixture[group].items():
           inputs = reference.get("experimental_inputs")
           if not inputs:
               continue
           model, point = case.split(":")
           smooth = (model == "model_lcdm_mnu.yml" or
                     group == "superseded_experimental_references")
           scripts = root
           if smooth:
               scripts = root / "ccmbs-smooth-reference-scripts"
           for name, expected in inputs["source_sha256"].items():
               if name.endswith(".f90"):
                   path = root / "CAMB-1.6.0" / "fortran" / name
               else:
                   path = scripts / name
               check(path, expected)
           check(root / "ccmbs-camb-1.6.0.tar.gz",
                 inputs["stock_archive_sha256"])
           check(root / "ccmbs-forutils.tar.gz",
                 inputs["forutils_archive_sha256"])
           if group == "references":
               prefix = "ccmbs-final-" if smooth else "ccmbs-ppf-"
               stem = prefix + model.removesuffix(".yml") + "-" + point
               candidates = reference["reference_refinement"]
               for level, candidate in (
                   candidates["candidate_payloads"].items()
               ):
                   for kind in ("recombination", "reionization"):
                       name = f"{stem}-{level}-{kind}.dat"
                       expected = candidate["experimental_inputs"]
                       check(root / name, expected[kind + "_table_sha256"])
           count += 1
   print(f"Verified source provenance for {count} experimental references.")
   PY
   ```

6. Build the experimental library in the extracted temporary tree. This uses
   the included forutils source. Do not install it over stock CAMB:

   ```sh
   (cd "$CCMBS_HANDOFF_DIR/CAMB-1.6.0" && python setup.py make)
   PYTHONPATH="$CCMBS_HANDOFF_DIR/CAMB-1.6.0" \
       python -c 'import camb; print(camb.__version__, camb.__file__)'
   python -c 'import camb; print(camb.__version__, camb.__file__)'
   ```

   The first import must resolve under the temporary source tree and the
   second under `.venv`. A newly built library has its own binary hash; it is
   not expected to match the old machine's compiler and architecture. The
   fixture's recorded original binary hash remains historical provenance.
   If compilation or loading fails, inspect the compiler and build output;
   do not alter the frozen references or silently switch implementations.

## Continue the work

Read the Slice Seven work and acceptance criteria in `PLAN.md`. The current
fixture contains stock and experimental reference histories; its experimental
PPF cases use the PPF generator set at the archive root. The earlier no-stress
set is retained for provenance and the massive-neutrino references. Their
original assembly scripts contain a previous macOS temporary path; inspect
and adapt the invocation to the extracted location before regenerating any
reference. Preserve the recorded originals and do not assume regenerated
floating-point arrays or binary hashes will match across machines.

Use the normal stock CAMB import for stock comparisons. Only an explicitly
scoped `PYTHONPATH` should select the experimental tree. Before starting a
long comparison, validate inputs and use the completion preflight:

```sh
source .venv/bin/activate
python -m tests.project.lib.scientific_acceptance --completion-preflight
```

A passing preflight only checks bindings; it does not accept scientific
spectra. Continue the actual matched comparisons, numerical checks and
repairs required by Slice Seven without weakening their tolerances or
assuming an LCDM-only solver. Report real results and remaining limitations.
