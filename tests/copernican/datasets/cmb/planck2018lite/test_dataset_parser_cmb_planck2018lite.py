"""Parser test for the Planck 2018 lite TT/TE/EE sample."""

from __future__ import annotations

import importlib
import unittest
from pathlib import Path


class TestPlanck2018LiteParser(unittest.TestCase):
    """Exercise the bundled Planck 2018 lite parser."""

    def test_loader_returns_cmb_power_spectra(self) -> None:
        parser_module = importlib.import_module(
            "copernican.datasets.cmb.planck2018lite."
            "dataset_parser_cmb_planck2018lite"
        )
        data_dir = (
            Path(__file__).resolve().parents[5]
            / "copernican"
            / "datasets"
            / "cmb"
            / "planck2018lite"
        )
        self.assertTrue(callable(parser_module.parse_planck2018lite))
        self.assertEqual(
            parser_module.parse_planck2018lite.__name__,
            "parse_planck2018lite",
        )
        cmb_dataframe = parser_module.parse_planck2018lite(str(data_dir))
        self.assertIsNotNone(cmb_dataframe)
        self.assertEqual(len(cmb_dataframe), 215)
        self.assertTrue(cmb_dataframe.attrs.get("is_cmb"))
        self.assertIn("covariance_matrix_inv", cmb_dataframe.attrs)
        self.assertEqual(
            cmb_dataframe.attrs["covariance_matrix_inv"].shape,
            (215, 215),
        )
        self.assertIn("param_names", cmb_dataframe.attrs)
        self.assertIn("H0", cmb_dataframe.attrs["param_names"])
        self.assertIn("Dl_obs", cmb_dataframe.columns)


if __name__ == "__main__":
    unittest.main()
