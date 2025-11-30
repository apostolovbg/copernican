# Copyright (c) 2025 Copernican Suite developers.
# See LICENSE.md in the repository root for details.

"""Security tests for parser discovery.

Only parser modules whose hashes are registered should be imported.
Malicious files must be ignored.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
import uuid
from pathlib import Path

import yaml

from copernican_lib import dataset_registry

class ParserDiscoverySecurityTestCase(unittest.TestCase):
    """Exercise parser discovery against trusted and rogue modules."""

    def test_only_whitelisted_modules_imported(self) -> None:
        """Ensure discovery loads only vetted modules and skips others."""
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                # --- Trusted parser setup ---
                good_dir = base / "sne" / "trusted"
                good_dir.mkdir(parents=True)
                meta = {"dataset_name": "Trusted SNe", "dataset_id": "trusted"}
                with open(
                    good_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as fh:
                    yaml.safe_dump(meta, fh)
                parser_path = good_dir / "cosmo_parser_trusted.py"
                code = (
                    "from copernican_lib.dataset_registry "
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

                # --- Malicious parser setup ---
                bad_dir = base / "sne" / "rogue"
                bad_dir.mkdir()
                meta_bad = {"dataset_name": "Rogue SNe", "dataset_id": "rogue"}
                with open(
                    bad_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as fh:
                    yaml.safe_dump(meta_bad, fh)
                bad_parser = bad_dir / "cosmo_parser_rogue.py"
                sentinel = base / "malicious_imported.txt"
                bad_code = (
                    "import os\n"
                    "with open(os.environ['MAL_SENTINEL'], 'w', "
                    "encoding='utf-8') as fh:\n"
                    "    fh.write('imported')\n"
                )
                bad_parser.write_text(bad_code, encoding="utf-8")
                os.environ["MAL_SENTINEL"] = str(sentinel)

                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(base_dir=tmp)
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
        """Symlinked directories and parsers must never be imported."""
        prev_parsers = dataset_registry.SNE_PARSER_REGISTRY.copy()
        prev_hashes = dataset_registry.TRUSTED_PARSER_DIGESTS.copy()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                outside = base.parent / f"outside_{uuid.uuid4().hex}"
                outside.mkdir()
                meta = {"dataset_name": "Link SNe", "dataset_id": "link"}
                with open(
                    outside / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as fh:
                    yaml.safe_dump(meta, fh)
                bad_parser = outside / "cosmo_parser_bad.py"
                sentinel = outside / "symlink_imported.txt"
                code = (
                    "import os\n"
                    "with open(os.environ['LINK_SENTINEL'], 'w', "
                    "encoding='utf-8') as fh:\n"
                    "    fh.write('imported')\n"
                )
                bad_parser.write_text(code, encoding="utf-8")
                os.environ["LINK_SENTINEL"] = str(sentinel)

                sne_dir = base / "sne"
                sne_dir.mkdir()
                # Symlinked dataset directory pointing outside ``base``.
                (sne_dir / "linked_dir").symlink_to(
                    outside, target_is_directory=True
                )
                # Real dataset with symlinked parser file.
                real_dir = sne_dir / "real"
                real_dir.mkdir()
                with open(
                    real_dir / "metadata_dummy.yml", "w", encoding="utf-8"
                ) as fh:
                    yaml.safe_dump(meta, fh)
                (real_dir / "cosmo_parser_real.py").symlink_to(bad_parser)

                dataset_registry.SNE_PARSER_REGISTRY = {}
                dataset_registry.discover_trusted_parsers(base_dir=tmp)
                self.assertFalse(sentinel.exists())
                self.assertEqual(dataset_registry.SNE_PARSER_REGISTRY, {})
        finally:
            dataset_registry.SNE_PARSER_REGISTRY = prev_parsers
            dataset_registry.TRUSTED_PARSER_DIGESTS = prev_hashes
            os.environ.pop("LINK_SENTINEL", None)

if __name__ == "__main__":  # pragma: no cover
    unittest.main()
