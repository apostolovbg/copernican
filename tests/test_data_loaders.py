"""Tests for ``copernican_lib.data_loaders`` registration helpers."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml

from copernican_lib import data_loaders


class DataLoaderRegistryTestCase(unittest.TestCase):
    """Exercise parser registration and metadata handling."""

    def test_register_and_load_sne_parser(self):
        """A temporary SNe parser should load data and metadata."""
        prev = data_loaders.SNE_PARSERS.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                meta = {
                    "dataset_name": "Dummy SNe",
                    "dataset_id": "dummy_sne",
                    "description": "test set",
                }
                with open(
                    os.path.join(tmp, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    yaml.safe_dump(meta, fh)

                @data_loaders.register_sne_parser(
                    name="dummy_sne",
                    data_dir=tmp,
                )
                def _parser(_dir, **_kwargs):
                    return pd.DataFrame(
                        {
                            "zcmb": [0.1],
                            "mu_obs": [1.0],
                            "e_mu_obs": [0.1],
                        }
                    )

                df = data_loaders.load_sne_data("dummy_sne")
                self.assertEqual(df.attrs["dataset_name"], "Dummy SNe")
                self.assertEqual(df.attrs["dataset_id"], "dummy_sne")
                self.assertEqual(len(df), 1)
        finally:
            data_loaders.SNE_PARSERS = prev

    @mock.patch("copernican_lib.data_loaders.console.ask", return_value="1")
    def test_select_source_uses_dataset_name(self, ask_mock):
        """Interactive selection should display names and return the id."""
        registry = {
            "dummy_sne": {
                "dataset_name": "Dummy SNe",
                "description": "test set",
                "data_dir": None,
                "function": lambda *_: None,
            }
        }
        captured = []
        with mock.patch(
            "copernican_lib.data_loaders.console.write",
            lambda msg: captured.append(msg),
        ):
            ds_id = data_loaders._select_source(registry, "SNe")
        self.assertEqual(ds_id, "dummy_sne")
        output = "".join(captured)
        self.assertIn("Dummy SNe", output)
        self.assertNotIn("dummy_sne", output)


if __name__ == "__main__":
    unittest.main()
