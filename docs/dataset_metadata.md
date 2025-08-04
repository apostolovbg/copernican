# Dataset Metadata Fields

Each dataset folder contains a `metadata_*.yml` file that describes the
source. All fields are optional except for `dataset_name` and
`description`.

- `dataset_name` -- Short human-readable identifier used in logs and plot
  footers.
- `description` -- Brief explanation of the dataset origin.
- `citation` -- Formatted as "FirstAuthor et al. - J. Vol (Year) Pages - DOI: URL".
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
- `notes` -- Additional free-form comments.

The metadata file is loaded automatically by
`copernican_lib.utils.load_metadata_from_dir` and attached to the parsed
`DataFrame` through the ``.attrs`` dictionary. Parsers store the original
`dataset_name` along with a sanitized variant, `dataset_name_sanitized`,
where spaces are replaced by underscores for safe filenames. Custom fields
are preserved and can be used by new engines or analysis scripts.
