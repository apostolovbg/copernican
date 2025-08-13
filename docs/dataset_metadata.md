# Dataset Metadata Fields

Each dataset folder contains a `metadata_*.yml` file that describes the
source. All fields are optional except for `dataset_name`, `dataset_id`
and `description`.

- `dataset_name` -- Short human-readable identifier used in logs, plot
  footers and CSV headers.
- `dataset_id` -- Short identifier used in filenames. It must omit spaces
  and forbidden characters: `/`, `\`, `:`, `*`, `?`, `"`, `<`, `>` and
  `|`, yet still convey which dataset is referenced.
- `description` -- Brief explanation of the dataset origin.
- `citation` -- Formatted as "FirstAuthor et al. - J. Vol (Year) Pages - DOI:
  URL".
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
- `notes` -- Additional free-form comments displayed on the second line of
  plot
  footers.

The metadata file is loaded automatically by the data loaders via
`copernican_lib.utils.load_metadata_from_dir` after the parser returns and
attached to the parsed `DataFrame` through the ``.attrs`` dictionary.
Loaders store both the human-readable `dataset_name` and the filename
friendly `dataset_id`. Plot footers render the dataset name in bold,
followed by `: description notes` on the second line and the citation on
a third line. Custom fields are preserved and can be used by new engines
or analysis scripts.

### Best Practices

- Keep descriptions short yet informative; the second footer line wraps at
  190 characters, so overly long notes may span several lines.
- Use the full author list to ensure proper attribution in publications that
  derive from the suite's outputs.
- Unknown fields are preserved by the loader, making it safe to add
  experiment-specific keys for downstream tools.
