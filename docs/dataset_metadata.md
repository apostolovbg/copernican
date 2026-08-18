# Dataset Metadata Fields
Each dataset folder contains a `metadata_*.yml` file that describes the source.
All fields are optional except for `dataset_name`, `dataset_id`, `description`
and `license`.
Run ``python -m tools.check_meta`` after updating any of these metadata files
or their documentation headers. The helper measures "today" in Coordinated
Universal Time so both the command-line report and the regression tests catch
future timestamps consistently across time zones.
Example skeleton:
```yaml
dataset_name: Example Dataset
dataset_id: example_set
description: Short human readable blurb
citation: FirstAuthor et al. 2024 - Journal 12 (2024) 34-56 - DOI: 10.x/y
license: Free to use with attribution
```
- `dataset_name` -- Short human-readable identifier used in logs, plot footers
 and CSV headers.
- `dataset_id` -- Short identifier used in filenames. It must omit spaces and
 forbidden characters: `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>` and `|`, yet
 convey which dataset is referenced.
- `description` -- Brief explanation of the dataset origin.
- `citation` -- Formatted as "FirstAuthor et al. - J. Vol (Year) Pages - DOI:
 URL".
- `license` -- Usage terms or license under which the dataset is released.
- `author` -- Full author list from the publication.
- `title` -- Publication title.
- `article` -- BibTeX citation key.
- `volume` -- Journal volume.
- `ISSN` -- International Standard Serial Number.
- `arXiv` -- Link to the arXiv abstract.
- `url` -- DOI URL.
- `DOI` -- Plain DOI identifier.
- `number` -- Journal issue number.
- `journal` -- Full journal name.
- `publisher` -- Publishing house.
- `year` -- Publication year.
- `month` -- Publication month (three-letter abbreviation).
- `pages` -- Page range or article number.
- `notes` -- Additional free-form comments displayed on the second line of plot
 footers.
The metadata file is loaded automatically by the data loaders via
`copernican.lib.utils.load_metadata_from_dir` after the parser returns and
attached to the parsed `DataFrame` through the ``.attrs`` dictionary. Loaders
store both the human-readable `dataset_name` and the filename friendly
`dataset_id`. Plot footers render the dataset name in bold, followed by `:
description notes` on the second line and the citation on a third line. Custom
fields are preserved and can be used by new samplers or analysis scripts.
### Best Practices
- Keep descriptions short yet informative; the second footer line wraps at 190
 characters, so overly long notes may span several lines.
- Use the full author list to ensure proper attribution in publications that
 derive from the suite's outputs.
- Unknown fields are preserved by the loader, making it safe to add experiment-
 specific keys for downstream tools.
- BAO datasets expose a `model_prediction` column during analysis. The
 prediction remains identical for control and test roles when their adapters
 match because the sampler reuses the shared posterior.
- Stage 5 summary files include `parameter-summary_*.yml/json`.
 Supernova-only MCMC runs copy the SNe chi-squared into ``χ²_Total`` so
 both sides of a self-consistency test report the same totals when models
 share an adapter.
### Union3 metadata example
- Include the Unity citation, author list and MIT license in the metadata file
 so the loaders attach them once the compressed FITS is parsed.
- Use the `notes` field to explain that `mu_mat_union3_cosmo=2_mu.fits`
 contains the redshift nodes, compressed distance modulus column and inverse
 covariance block consumed by `dataset_parser_union3.py`. If the dataset
 requires an additive SNe intercept treatment, say so explicitly so the
 likelihood can marginalize it before residual comparison.
- Point readers to [`licenses/Union3-MIT.txt`](../licenses/Union3-MIT.txt)
 whenever spelling out the dataset's usage terms in release notes or
 documentation.
### Model Parameter Priors
Model YAML files support a `prior` block for each parameter. Priors carry
their `type` and relevant numeric fields: Gaussian priors require `mean` and
`sigma`, uniform priors use `lower` and `upper`, and log-uniform priors demand
strictly positive `lower`/`upper` pairs. Parsed models expose these details so
samplers can apply them during optimisation while
`copernican.lib.priors.LogUniformPrior` injects the accompanying log-space
transform automatically.
