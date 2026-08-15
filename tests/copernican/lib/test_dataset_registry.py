"""Tests for dataset registry and parser discovery."""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import pandas
import yaml

from copernican.lib import dataset_registry, utils
from copernican.lib.utils import load_metadata_from_dir


class ParserDiscoverySecurityTestCase(unittest.TestCase):
    """Exercise discovery against trusted and rogue parser modules."""

    def test_only_whitelisted_modules_imported(self) -> None:
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                base = Path(temporary_dir)
                good_dir = base / "sne" / "trusted"
                good_dir.mkdir(parents=True)
                meta = {
                    "dataset_name": "Trusted SNe",
                    "dataset_id": "trusted",
                }
                with open(
                    good_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                parser_path = good_dir / "cosmo_parser_trusted.py"
                code = (
                    "from copernican.lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='trusted', data_dir=r'"
                    f"{good_dir}')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                parser_path.write_text(code, encoding="utf-8")
                rel_key = os.path.relpath(parser_path, base).replace("\\", "/")
                digest = dataset_registry._file_sha256(parser_path)
                dataset_registry.TRUSTED_PARSER_DIGESTS[rel_key] = digest

                bad_dir = base / "sne" / "rogue"
                bad_dir.mkdir()
                meta_bad = {
                    "dataset_name": "Rogue SNe",
                    "dataset_id": "rogue",
                }
                with open(
                    bad_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as file_handle:
                    yaml.safe_dump(meta_bad, file_handle)
                bad_parser = bad_dir / "cosmo_parser_rogue.py"
                sentinel = base / "malicious_imported.txt"
                bad_code = (
                    "import os\n"
                    "with open(os.environ['MAL_SENTINEL'], 'w', "
                    "encoding='utf-8') as file_handle:\n"
                    "    file_handle.write('imported')\n"
                )
                bad_parser.write_text(bad_code, encoding="utf-8")
                os.environ["MAL_SENTINEL"] = str(sentinel)

                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(
                    base_dir=temporary_dir, force=True
                )
                self.assertIn("trusted", dataset_registry.SNE_PARSER_REGISTRY)
                self.assertNotIn("rogue", dataset_registry.SNE_PARSER_REGISTRY)
                self.assertFalse(sentinel.exists())
                self.assertNotIn(
                    "data.sne.rogue.cosmo_parser_rogue", sys.modules
                )
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev_parsers
            dataset_registry.TRUSTED_PARSER_DIGESTS = prev_hashes
            os.environ.pop("MAL_SENTINEL", None)

    def test_symlinked_paths_are_skipped(self) -> None:
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                base = Path(temporary_dir)
                outside = base.parent / f"outside_{uuid.uuid4().hex}"
                outside.mkdir()
                meta = {"dataset_name": "Link SNe", "dataset_id": "link"}
                with open(
                    outside / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                bad_parser = outside / "cosmo_parser_bad.py"
                sentinel = outside / "symlink_imported.txt"
                code = (
                    "import os\n"
                    "with open(os.environ['LINK_SENTINEL'], 'w', "
                    "encoding='utf-8') as file_handle:\n"
                    "    file_handle.write('imported')\n"
                )
                bad_parser.write_text(code, encoding="utf-8")
                os.environ["LINK_SENTINEL"] = str(sentinel)

                sne_dir = base / "sne"
                sne_dir.mkdir()
                (sne_dir / "linked_dir").symlink_to(
                    outside, target_is_directory=True
                )
                real_dir = sne_dir / "real"
                real_dir.mkdir()
                with open(
                    real_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                (real_dir / "cosmo_parser_real.py").symlink_to(bad_parser)

                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(
                    base_dir=temporary_dir, force=True
                )
                self.assertFalse(sentinel.exists())
                self.assertEqual(dataset_registry.SNE_PARSER_REGISTRY, {})
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev_parsers
            dataset_registry.TRUSTED_PARSER_DIGESTS = prev_hashes
            os.environ.pop("LINK_SENTINEL", None)


class DataLoaderRegistryTestCase(unittest.TestCase):
    """Exercise parser registration and metadata handling."""

    def test_register_and_load_sne_parser(self):
        prev = dataset_registry.SNE_PARSER_REGISTRY.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary_dir:
                meta = {
                    "dataset_name": "Dummy SNe",
                    "dataset_id": "dummy_sne",
                    "description": "test set",
                    "version": "1.0-test",
                }
                with open(
                    os.path.join(temporary_dir, "metadata_dummy.yml"),
                    "w",
                    encoding="utf-8",
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)

                @dataset_registry.register_sne_parser(
                    name="dummy_sne",
                    data_dir=temporary_dir,
                )
                def _parser(_dir, **_kwargs):
                    return pandas.DataFrame(
                        {
                            "zcmb": [0.1],
                            "mu_obs": [1.0],
                            "e_mu_obs": [0.1],
                        }
                    )

                dataframe = dataset_registry.load_sne_data("dummy_sne")
                self.assertEqual(dataframe.attrs["dataset_name"], "Dummy SNe")
                self.assertEqual(dataframe.attrs["dataset_id"], "dummy_sne")
                self.assertEqual(len(dataframe), 1)
                self.assertEqual(
                    dataframe.attrs["dataset_version"], "1.0-test"
                )
                self.assertEqual(dataframe.attrs["data_path"], temporary_dir)
                self.assertEqual(
                    dataframe.attrs["independence_assumptions"],
                    dataset_registry.OBSERVATION_INDEPENDENCE_NOTES["sne"],
                )
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev

    def test_hash_mismatch_skips_parser(self):
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
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
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                parser_path = os.path.join(sne_dir, "cosmo_parser_rogue2.py")
                code = (
                    "from copernican.lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='rogue2_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(code)
                rel_path = os.path.relpath(parser_path, base)
                dataset_registry.TRUSTED_PARSER_DIGESTS[rel_path] = "0" * 64
                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(
                    base_dir=base, force=True
                )
                self.assertNotIn(
                    "rogue2_sne", dataset_registry.SNE_PARSER_REGISTRY
                )
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev_parsers
            dataset_registry.TRUSTED_PARSER_DIGESTS = prev_hashes

    def test_windows_separators_are_normalized(self):
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
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
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                parser_path = os.path.join(sne_dir, "cosmo_parser_trusted.py")
                code = (
                    "from copernican.lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='trusted_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(code)
                rel_key = os.path.relpath(parser_path, base).replace("\\", "/")
                digest = dataset_registry._file_sha256(parser_path)
                dataset_registry.TRUSTED_PARSER_DIGESTS[rel_key] = digest
                dataset_registry.SNE_PARSER_REGISTRY = {}
                orig_relpath = os.path.relpath

                def fake_relpath(path, start):
                    return orig_relpath(path, start).replace("/", "\\")

                with mock.patch("os.path.relpath", side_effect=fake_relpath):
                    dataset_registry.discover_trusted_parsers(
                        base_dir=base, force=True
                    )
                self.assertIn(
                    "trusted_sne", dataset_registry.SNE_PARSER_REGISTRY
                )
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev_parsers
            dataset_registry.TRUSTED_PARSER_DIGESTS = prev_hashes

    @mock.patch(
        "copernican.lib.dataset_registry.console.ask", return_value="1"
    )
    def testprompt_dataset_selection_uses_dataset_name(self, ask_mock):
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
            "copernican.lib.dataset_registry.console.write",
            lambda msg: captured.append(msg),
        ):
            ds_id = dataset_registry.prompt_dataset_selection(registry, "SNe")
        self.assertEqual(ds_id, "dummy_sne")
        output = "".join(captured)
        self.assertIn("Dummy SNe", output)
        self.assertNotIn("dummy_sne", output)

    def test_invalid_metadata_raises_yaml_error(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            bad = os.path.join(temporary_dir, "metadata_bad.yml")
            with open(bad, "w", encoding="utf-8") as file_handle:
                file_handle.write("dataset_name: bad\nitems: [1, 2\n")
            with self.assertRaises(yaml.YAMLError):
                load_metadata_from_dir(temporary_dir)

    def test_untrusted_parser_is_skipped(self):
        prev = dataset_registry.SNE_PARSER_REGISTRY.copy()
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
                ) as file_handle:
                    yaml.safe_dump(meta, file_handle)
                parser_path = os.path.join(sne_dir, "cosmo_parser_rogue.py")
                code = (
                    "from copernican.lib.dataset_registry "
                    "import register_sne_parser\n"
                    "@register_sne_parser(name='rogue_sne', data_dir=r'"
                    + sne_dir
                    + "')\n"
                    "def load(_dir, **_kwargs):\n"
                    "    return None\n"
                )
                with open(parser_path, "w", encoding="utf-8") as file_handle:
                    file_handle.write(code)
                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(
                    base_dir=base, force=True
                )
                self.assertNotIn(
                    "rogue_sne", dataset_registry.SNE_PARSER_REGISTRY
                )
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev


class CompoundBaoHashRegressionTest(unittest.TestCase):
    """Verify compound BAO file-hash bookkeeping and logging."""

    def test_compound_bao_file_hash_is_attached_and_logged(self) -> None:
        parser_module = importlib.import_module(
            "copernican.datasets.bao.compound.cosmo_parser_compound"
        )
        registry_entry = dataset_registry.BAO_PARSER_REGISTRY[
            "compound_bao_set"
        ]
        original_parser = registry_entry["function"]
        parser_calls = 0

        def counted_parser(*args, **kwargs):
            nonlocal parser_calls
            parser_calls += 1
            return parser_module.parse_bao(*args, **kwargs)

        registry_entry["function"] = counted_parser
        try:
            with (
                mock.patch.object(
                    dataset_registry,
                    "collect_dataset_hashes",
                    wraps=dataset_registry.collect_dataset_hashes,
                ) as hash_mock,
                self.assertLogs(level="INFO") as log,
            ):
                bao_dataframe = dataset_registry.load_bao_data(
                    dataset_id="compound_bao_set"
                )
        finally:
            registry_entry["function"] = original_parser

        hashes = bao_dataframe.attrs.get("file_hashes", {})
        compound_path = Path("copernican/datasets/bao/compound/compound.yml")
        expected = utils.compute_sha256(str(compound_path))

        self.assertEqual(hashes.get("compound.yml"), expected)
        self.assertTrue(any(expected in message for message in log.output))
        self.assertEqual(parser_calls, 1)
        self.assertEqual(hash_mock.call_count, 1)
        self.assertEqual(bao_dataframe.attrs["covariance_model"], "diagonal")
        self.assertFalse(
            any("falling back" in message.lower() for message in log.output)
        )
        self.assertEqual(
            sum(
                "declared diagonal covariance" in message
                for message in log.output
            ),
            1,
        )


class PublicSymbolCoverageTestCase(unittest.TestCase):
    """Expose the dataset registry API to the coverage policy."""

    def test_public_symbols_are_exposed(self) -> None:
        self.assertTrue(callable(dataset_registry.collect_dataset_hashes))
        self.assertTrue(callable(dataset_registry.discover_trusted_parsers))
        self.assertTrue(callable(dataset_registry.get_parser_registries))
        self.assertTrue(callable(dataset_registry.get_parser_registry))
        self.assertTrue(callable(dataset_registry.load_bao_data))
        self.assertTrue(callable(dataset_registry.load_cmb_data))
        self.assertTrue(callable(dataset_registry.load_gw_data))
        self.assertTrue(callable(dataset_registry.register_bao_parser))
        self.assertTrue(callable(dataset_registry.register_cmb_parser))
        self.assertTrue(callable(dataset_registry.register_gw_parser))
        self.assertTrue(callable(dataset_registry.register_sne_parser))

    def test_decorator_factory_is_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            decorator = dataset_registry.register_sne_parser(
                name="coverage_demo",
                data_dir=temporary_dir,
            )
            self.assertTrue(callable(decorator))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
