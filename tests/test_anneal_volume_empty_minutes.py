"""真实数据回归：早期研究为 NaN，2026 年补算曾误把空分钟填零。"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rust_pyfunc as rp


class EmptyMinuteRegression(unittest.TestCase):
    @unittest.skipUnless(Path('/ssd_data/stock/20241231/transaction').is_dir(), 'requires local L2 fixture')
    def test_empty_minutes_and_worker_use_same_definition(self):
        names = rp.py_anneal_volume_names()
        index = names.index('min_mixed_F2_posD_ratio_iqr')
        expected = {(20241231, '000001'): .013999998569488525,
                    (20241231, '600000'): .021700024604797363,
                    (20260908, '000001'): .02180001139640808,
                    (20260908, '600000'): .03839999437332153}
        values = {}
        for (date, code), reference in expected.items():
            value = np.asarray(rp.py_anneal_volume(code, date), dtype=np.float32)
            self.assertEqual(len(value), 5975)
            self.assertEqual(float(value[index]), reference)
            np.testing.assert_array_equal(value, rp.py_anneal_volume(code, date))
            values[date, code] = value
        # 显式验证真实 pipeline worker，防止只更新 Python 扩展而遗留旧 worker。
        with tempfile.TemporaryDirectory() as directory:
            worker = str(Path(rp.__file__).parent / 'rust_pyfunc_worker')
            previous = os.environ.get('RUST_PYFUNC_WORKER_BIN')
            os.environ['RUST_PYFUNC_WORKER_BIN'] = worker
            try:
                rp.run_factor_pipeline(pipeline='anneal_volume',
                    tasks=[[date, code] for date, code in expected], n_jobs=2,
                    backup_file='', expected_result_length=len(names),
                    trading_days=rp.td.trading_days.tolist(), bind_cores=False,
                    store_dir=directory, store_factor_names=names, export_n_jobs=2)
                frame = rp.read_factor_from_colblk(directory, names[index], 20241231, 20260908)
                frame.index = [int(str(x).replace('-', '')[:8]) for x in frame.index]
                frame.columns = [str(x).split('.')[0].zfill(6) for x in frame.columns]
                for key, reference in expected.items():
                    self.assertEqual(float(frame.loc[key[0], key[1]]), reference)
            finally:
                if previous is None:
                    os.environ.pop('RUST_PYFUNC_WORKER_BIN', None)
                else:
                    os.environ['RUST_PYFUNC_WORKER_BIN'] = previous


if __name__ == '__main__':
    unittest.main()
