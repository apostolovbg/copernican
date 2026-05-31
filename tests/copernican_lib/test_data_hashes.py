import importlib
import unittest
from pathlib import Path

from copernican_lib import dataset_registry, utils


class CompoundBaoHashRegressionTest(unittest.TestCase):
    """Verify compound BAO file-hash bookkeeping and logging."""

    def test_compound_bao_file_hash_is_attached_and_logged(self):
        importlib.import_module(
            "copernican.datasets.bao.compound.cosmo_parser_compound"
        )
        with self.assertLogs(level="INFO") as log:
            bao_dataframe = dataset_registry.load_bao_data(
                dataset_id="compound_bao_set"
            )

        hashes = bao_dataframe.attrs.get("file_hashes", {})
        compound_path = Path("copernican/datasets/bao/compound/compound.yml")
        expected = utils.compute_sha256(str(compound_path))

        self.assertEqual(hashes.get("compound.yml"), expected)
        self.assertTrue(any(expected in message for message in log.output))
