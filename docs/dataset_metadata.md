# Dataset Metadata Fields

Each dataset folder contains a `metadata_*.yml` file that describes the
source. All fields are optional except for `dataset_name` and
`description`.

- `dataset_name` -- Short human-readable identifier used in logs and plot
  footers.
- `description` -- Brief explanation of the dataset origin.
- `citation` -- Reference string cited in plots and CSV headers.
- `notes` -- Additional free-form comments.
- `authors_all` -- List of authors for provenance tracking.

The metadata file is loaded automatically by
`copernican_lib.utils.load_metadata_from_dir` and attached to the parsed
`DataFrame` through the ``.attrs`` dictionary. Custom fields are preserved
and can be used by new engines or analysis scripts.
