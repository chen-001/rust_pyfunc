import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
from rust_pyfunc.factor_production import factor_production_runtime


class ProductionPortability(unittest.TestCase):
    def test_hidden_names_need_no_historical_raw_files(self):
        env = {**os.environ, 'RUST_PYFUNC_LEVEL2_ROOT': '/nonexistent/hmokay4_test'}
        code = 'import rust_pyfunc as rp; n=rp.py_hidden_arrange_names(); assert len(n)==len(set(n))==8772'
        subprocess.run([sys.executable, '-c', code], env=env, check=True)

    def test_minute_dates_use_explicit_h5_root_and_full_window(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = [20150105, 20150106, 20150107, 20150108, 20150109, 20150112, 20150113]
            pd.DataFrame({'date_min': dates}).to_csv(root / 'calendar_map.csv', index=False)
            with h5py.File(root / 'volume.h5', 'w') as f:
                values = np.ones((7 * 240, 2)); values[6 * 240:] = np.nan
                f['data'] = values
            runtime = factor_production_runtime(dict(base_factor_ver='test_base', base_hdf5_dir=str(root / 'base'),
                calendar_root=str(root / 'calendar'), vars_root=str(root / 'vars'), minute_root=str(root),
                level2_root=str(root / 'no_level2'), pipeline_groups=[dict(pipeline='switch_moment', names=[], dir=None)]))
            self.assertEqual(runtime.raw_dates(20150105, 20150113), [20150109, 20150112])
            self.assertEqual(runtime.raw_dates(20150105, 20150108), [])
            self.assertEqual(runtime.raw_dates(20150109, 20150109), [20150109])


if __name__ == '__main__':
    unittest.main()
