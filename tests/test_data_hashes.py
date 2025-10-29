import importlib
import os
import unittest

from copernican_lib import data_loaders, utils


class DataHashingTest(unittest.TestCase):
    """Verify SHA256 hashes are stored and logged for dataset files."""

    def test_bao_compound_hashes(self):
        importlib.import_module("data.bao.compound.cosmo_parser_compound")
        with self.assertLogs(level="INFO") as log:
            df = data_loaders.load_bao_data(dataset_id="compound_bao_set")
        hashes = df.attrs.get("file_hashes", {})
        fname = os.path.join("data", "bao", "compound", "compound.yml")
        expected = utils.compute_sha256(fname)
        self.assertEqual(hashes.get("compound.yml"), expected)
        self.assertTrue(any(expected in msg for msg in log.output))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
