import unittest
import importlib
from pathlib import Path
import numpy as np

from scripts import model_parser, model_coder, engine_interface, data_loaders
import engines.cosmo_engine_1_4b as engine


class FunctionalTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(__file__).resolve().parents[1]
        models_dir = base / 'models'
        json_path = models_dir / 'cosmo_model_lcdm.json'
        cache_dir = models_dir / 'cache'
        cache_path = model_parser.parse_model_json(json_path, cache_dir)
        funcs, parsed = model_coder.generate_callables(cache_path)
        cls.plugin = engine_interface.build_plugin(parsed, funcs)
        engine_interface.validate_plugin(cls.plugin)

    def test_plugin_validation(self):
        self.assertTrue(hasattr(self.plugin, 'distance_modulus_model'))

    def test_engine_routines(self):
        sne_df = data_loaders.load_sne_data('University of Strassbourg dataset')
        self.assertIsNotNone(sne_df)
        sne_df = sne_df.head(3)

        bao_df = data_loaders.load_bao_data('Basic BAO testing dataset')
        self.assertIsNotNone(bao_df)
        bao_df = bao_df.head(3)

        cmb_df = data_loaders.load_cmb_data('planck2018lite_v1')
        self.assertIsNotNone(cmb_df)

        params = self.plugin.INITIAL_GUESSES
        chi2_sne = engine.chi_squared_sne_h1_fixed_nuisance(
            params,
            self.plugin.distance_modulus_model,
            sne_df
        )
        self.assertTrue(np.isfinite(chi2_sne))

        pred_df, rs_mpc, _ = engine.calculate_bao_observables(bao_df, self.plugin, params)
        chi2_bao = engine.chi_squared_bao(bao_df, self.plugin, params, rs_mpc)
        self.assertTrue(np.isfinite(chi2_bao))

        camb_params = self.plugin.get_camb_params(params)
        chi2_cmb = engine.chi_squared_cmb(camb_params, cmb_df)
        spec = engine.compute_cmb_spectrum(camb_params, cmb_df['ell'].values)
        self.assertTrue(np.isfinite(chi2_cmb))
        self.assertEqual(len(spec), len(cmb_df))


if __name__ == '__main__':
    unittest.main()
