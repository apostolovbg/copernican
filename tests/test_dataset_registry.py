# Last Updated: 2025-11-24
# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Tests for ``copernican_lib.dataset_registry`` registration helpers."""

import os
import tempfile
import unittest
from unittest import mock

import pandas as pd
import yaml

from copernican_lib import dataset_registry
from copernican_lib.utils import load_metadata_from_dir


class DataLoaderRegistryTestCase(unittest.TestCase):
    """Exercise parser registration and metadata handling."""

    def test_register_and_load_sne_parser(self):
        """A temporary SNe parser should load data and metadata."""
        prev = dataset_registry.SNE_PARSERS.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                meta = {
                    "dataset_name": "Dummy SNe",
                    "dataset_id": "dummy_sne",
                    "description": "test set",
                    "version": "1.0-test",
                }
                with open(
                    os.path.join(tmp, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    yaml.safe_dump(meta, fh)

                @dataset_registry.register_sne_parser(
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

                df = dataset_registry.load_sne_data("dummy_sne")
                self.assertEqual(df.attrs["dataset_name"], "Dummy SNe")
                self.assertEqual(df.attrs["dataset_id"], "dummy_sne")
                self.assertEqual(len(df), 1)
                self.assertEqual(df.attrs["dataset_version"], "1.0-test")
                self.assertEqual(df.attrs["data_path"], tmp)
                self.assertEqual(
                    df.attrs["independence_assumptions"],
                    dataset_registry.INDEPENDENCE_ASSUMPTIONS["sne"],
                )
        finally:
            dataset_registry.SNE_PARSERS = prev

    def test_hash_mismatch_skips_parser(self):
        """Hash mismatches prevent parser import."""
        prev_parsers = dataset_registry.SNE_PARSERS.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_HASHES.copy()
        try:
            with tempfile.TemporaryDirectory() as base:
                sne_dir = os.path.join(base, "sne", "rogue2")
                os.makedirs(sne_dir)
                meta = {
                    "dataset_name": "Rogue2 SNe",
                    "dataset_id": "rogue2_sne",
                }
                with open(
                    os.path.join(sne_dir, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    yaml.safe_dump(meta, fh)
                parser_path = os.path.join(sne_dir, "cosmo_parser_rogue2.py")
                code = (
                    "from copernican_lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='rogue2_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as fh:
                    fh.write(code)
                rel_path = os.path.relpath(parser_path, base)
                dataset_registry.TRUSTED_PARSER_HASHES[rel_path] = "0" * 64
                dataset_registry.SNE_PARSERS = {}
                dataset_registry._discover_parsers(base_dir=base)
                self.assertNotIn("rogue2_sne", dataset_registry.SNE_PARSERS)
        finally:
            dataset_registry.SNE_PARSERS = prev_parsers
            dataset_registry.TRUSTED_PARSER_HASHES = prev_hashes

    def test_windows_separators_are_normalized(self):
        """Trusted modules load even when ``relpath`` returns backslashes."""
        prev_parsers = dataset_registry.SNE_PARSERS.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_HASHES.copy()
        try:
            with tempfile.TemporaryDirectory() as base:
                sne_dir = os.path.join(base, "sne", "trusted")
                os.makedirs(sne_dir)
                meta = {
                    "dataset_name": "Trusted SNe",
                    "dataset_id": "trusted_sne",
                }
                with open(
                    os.path.join(sne_dir, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    yaml.safe_dump(meta, fh)
                parser_path = os.path.join(sne_dir, "cosmo_parser_trusted.py")
                code = (
                    "from copernican_lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='trusted_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as fh:
                    fh.write(code)
                rel_key = os.path.relpath(parser_path, base).replace("\\", "/")
                digest = dataset_registry._file_sha256(parser_path)
                dataset_registry.TRUSTED_PARSER_HASHES[rel_key] = digest
                dataset_registry.SNE_PARSERS = {}
                orig_relpath = os.path.relpath

                def fake_relpath(path, start):
                    return orig_relpath(path, start).replace("/", "\\")

                with mock.patch("os.path.relpath", side_effect=fake_relpath):
                    dataset_registry._discover_parsers(base_dir=base)
                self.assertIn("trusted_sne", dataset_registry.SNE_PARSERS)
        finally:
            dataset_registry.SNE_PARSERS = prev_parsers
            dataset_registry.TRUSTED_PARSER_HASHES = prev_hashes

    @mock.patch(
        "copernican_lib.dataset_registry.console.ask", return_value="1"
    )
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
            "copernican_lib.dataset_registry.console.write",
            lambda msg: captured.append(msg),
        ):
            ds_id = dataset_registry._select_source(registry, "SNe")
        self.assertEqual(ds_id, "dummy_sne")
        output = "".join(captured)
        self.assertIn("Dummy SNe", output)
        self.assertNotIn("dummy_sne", output)

    def test_invalid_metadata_raises_yaml_error(self):
        """Invalid YAML metadata should raise ``YAMLError``."""
        with tempfile.TemporaryDirectory() as tmp:
            bad = os.path.join(tmp, "metadata_bad.yml")
            with open(bad, "w", encoding="utf-8") as fh:
                fh.write("dataset_name: bad\nitems: [1, 2\n")
            with self.assertRaises(yaml.YAMLError):
                load_metadata_from_dir(tmp)

    def test_untrusted_parser_is_skipped(self):
        """Modules not whitelisted should never be imported."""
        prev = dataset_registry.SNE_PARSERS.copy()
        try:
            with tempfile.TemporaryDirectory() as base:
                sne_dir = os.path.join(base, "sne", "rogue")
                os.makedirs(sne_dir)
                meta = {
                    "dataset_name": "Rogue SNe",
                    "dataset_id": "rogue_sne",
                }
                with open(
                    os.path.join(sne_dir, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as fh:
                    yaml.safe_dump(meta, fh)
                parser_path = os.path.join(sne_dir, "cosmo_parser_rogue.py")
                code = (
                    "from copernican_lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='rogue_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as fh:
                    fh.write(code)
                dataset_registry.SNE_PARSERS = {}
                dataset_registry._discover_parsers(base_dir=base)
                self.assertNotIn("rogue_sne", dataset_registry.SNE_PARSERS)
        finally:
            dataset_registry.SNE_PARSERS = prev


if __name__ == "__main__":
    unittest.main()
