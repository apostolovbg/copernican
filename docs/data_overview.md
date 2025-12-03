# Data Directory and Provenance Overview

The `data/` tree holds every vetted observation the Copernican Suite ships with,
alongside dedicated parsers and metadata that describe authorship, licensing,
and scientific provenance. The directory is read-only by default to preserve
auditable datasets; parser `.py` files and `metadata_*.yml` files may be
updated when a dataset revision is needed, but the original tables remain
untouched unless a release explicitly adds a new version. Data folders are
replicated under the `docs/` narrative and referenced in each changelog entry
that touches them per the `changelog-coverage` policy.

```
data/
  sne/        - Supernovae Type Ia datasets (JLA, Pantheon+, Union3 UNITY)
  bao/        - Baryon Acoustic Oscillation tables (BOSS DR12, compound)
  cmb/        - Cosmic Microwave Background \(\ell > 2\) spectra (Planck 2018 lite)
  gw/         - Gravitational-wave standard sirens (placeholder stubs)
```

Each dataset folder includes:

* **Parser** `cosmo_parser_<source>.py` – registers via
  `copernican_lib.dataset_registry.register_parser`, returns a
  `pandas.DataFrame`, and decorates observations with metadata such as
  `covariance_matrix_inv` or `diag_errors_for_plot`.
* **Metadata** `metadata_<source>.yml` – lists `dataset_name`, `dataset_id`,
  `description`, `citation`, `license`, authors, BibTeX fields, and any
  `independence_assumptions` that run configurations record in
  `copernican_lib/config_schemas/run_config.yml`.
* **Trusted digest** in `copernican_lib.dataset_registry.TRUSTED_PARSER_DIGESTS`
  – the launcher refuses to load a parser whose SHA256 hash does not match the
  recorded value, protecting against tampering and ensuring reproducibility.
* **Data tables** – the raw tables are read when parsing and their SHA256
  digests (excluding parser scripts) are computed, logged, and stored on
  `df.attrs['file_hashes']` so manifests note every exact byte that influenced a
  run.

The loaders append `dataset_version` and `data_path` to the DataFrame attributes
and log the dataset name plus digest sequence so the manifest consistently
replays the same inputs for downstream auditors. When a dataset is re-used by
multiple models (for example, `pantheon` as both SNe and in a joint BAO/CMB run)
the digests guarantee the CLI and GUI manifest builder still point to the same
files.

## Dataset Catalogs

### Supernovae (SNe Ia)

1. **JLA Betoule+2014**
   *Location:* `data/sne/jla2014/`
   *Parser:* `cosmo_parser_jla2014.py` ingests `tablef3.dat` and systematics
   from `tablef4.fit`, builds the SALT2 nuisance vector (\(M_B, \alpha, \beta\)),
   and exposes the inverted covariance via `df.attrs['covariance_matrix_inv']`.
   The diagonal errors used for plotting are stored as
   `df.attrs['diag_errors_for_plot']`.
2. **Pantheon+SH0ES 2022**
   *Location:* `data/sne/pantheon/`
   *Parser:* `cosmo_parser_pantheon.py` locates `.dat` and `.cov` files, sorts
   by redshift, attaches diagonal errors, and inverts the full covariance
   matrix. When matrix inversion fails, it gracefully falls back to tolerances
   inferred from the diagonal uncertainties and logs the fallback path.
3. **Union3 UNITY (Rubin et al. 2025)**
   *Location:* `data/sne/union3/`
   *Parser:* `cosmo_parser_union3.py` loads the UNITY compressed FITS table,
   extracts the 22 redshift nodes, distance moduli, and the inverse covariance
   block used in likelihood calculations. The release includes a README
   explaining how to regenerate the UNITY tables from the provided Stan models.

### BAO (Baryon Acoustic Oscillation)

1. **BOSS DR12 Consensus (Alam et al. 2017)**
   *Location:* `data/bao/bossdr12/`
   *Parser:* `cosmo_parser_bossdr12.py` combines \(D_M/rs\), \(D_H/rs\), and
   \(D_V/rs\) observables, builds a block-diagonal covariance, and stores the
   inverse matrix plus diagonal errors in the DataFrame attributes for later
   χ² evaluations. The parser also records the fiducial sound horizon used to
   normalise the transformation.
2. **Compound BAO Dataset**
   *Location:* `data/bao/compound/`
   *Parser:* `cosmo_parser_compound.py` loads YAML-defined `data_points`,
   coerces numeric columns, and discards incomplete rows. No covariance matrix
   is provided; the engine instead applies a diagonal covariance derived from
   the uncertainties in the YAML, and the parser documents this assumption in
   `metadata_compound.yml`.

### CMB (Cosmic Microwave Background)

1. **Planck 2018 Lite TT/TE/EE**
   *Location:* `data/cmb/planck2018lite/`
   *Parser:* `cosmo_parser_cmb_planck2018lite.py` splits the `.dat` TT/TE/EE
   blocks by looking for drops in \(\ell\), transforms to \(D_\ell\), reads the
   Fortran binary covariance (`c_matrix_plik_v22.dat`), adjusts endianness,
   converts to \(D_\ell\), and stores the inverse matrix on `df.attrs`.

### Gravitational Waves (GW)

Placeholder directories under `data/gw/` log stub messages and return `None`.
They exist so future gravitational-wave standard siren data can join the GUI
catalogue once the parser and metadata are ready. The loader skips directories
naming themselves `placeholder` unless a human flips them to an active state.

## Adding or Updating Datasets

1. Create `data/<type>/<source>/`.
2. Add `cosmo_parser_<source>.py` that decorates a loader function with
   `dataset_registry.register_parser`. The function must return a
   `pandas.DataFrame` and attach metadata items and covariance helpers under
   `df.attrs`.
3. Supply `metadata_<source>.yml` with `dataset_name`, `dataset_id`, `description`, `citation`, `license`, `authors`, `BibTeX`, `independence_assumptions`, and any dataset-specific notes.
4. Update `copernican_lib.dataset_registry.TRUSTED_PARSER_DIGESTS` with the new
   parser’s SHA256 digest (normalize line endings to `\n` before hashing), then
   log the change in `CHANGELOG.md` and expand this file to describe the update.

The registry automatically discovers every parser whose folder is not named
`placeholder`, so renaming a directory to remove the staging marker instantly
makes it visible in the interactive menus.

## Parser Hash Verification

The loader refuses to import a parser whose digest does not match an entry in
`TRUSTED_PARSER_DIGESTS`. To refresh a hash:

```sh
python - <<'PY'
import hashlib
from pathlib import Path

path = Path("data/sne/jla2014/cosmo_parser_jla2014.py")
content = path.read_bytes().replace(b"\r\n", b"\n")
print(hashlib.sha256(content).hexdigest())
PY
```

Update the digest mapping, bump the changelog, and record the hash change in
this document so every hash edit is auditable.
