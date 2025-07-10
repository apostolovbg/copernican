# BAO JSON Dataset Format

Each BAO dataset is stored in a folder under `data/bao/<source>/`. The JSON file
contains a single object with a `data_points` array. The parser also looks for an
optional `metadata_*.json` file in the same folder which provides a dataset name
and citation.

Example `bao1.json`:
```json
{
  "data_points": [
    {"name": "6dFGS z=0.106", "redshift": 0.106,
     "observable_type": "DV_over_rs", "value": 2.976,
     "error": 0.133, "rs_fiducial_Mpc": null},
    ...
  ]
}
```

Example `metadata_bao1.json`:
```json
{
  "dataset_name": "Compound BAO dataset",
  "description": "Compilation of BAO distance measurements from multiple surveys",
  "citation": "Reference string with survey citations",
  "notes": "Any extra comments"
}
```

All observable types use the naming convention `DV_over_rs`, `DM_over_rs` or
`DH_over_rs` to indicate $D_V$, $D_M$ or $D_H$ divided by the sound horizon. The
parser converts the JSON to a Pandas `DataFrame` and attaches the metadata to the
`.attrs` attribute. This `metadata_*.json` with `dataset_name`, `description`,
`citation` and optional `notes` is used for **all** datasets so plot footers
can display consistent source information.
