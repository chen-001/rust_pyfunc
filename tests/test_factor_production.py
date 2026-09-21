"""生产协议回归：完整截面 fold、收尾缺失填补，以及失败时保留中间产物。"""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import rust_pyfunc as rp
from rust_pyfunc.factor_production import _ProductionRuntime, _runtime


class ProductionRegression(unittest.TestCase):
    def test_fold_includes_nonstock_before_output_filter(self):
        runtime = _runtime("test_base", "/unused/base", "/unused", "/unused",
                           names_save=["a_fold"])
        raw = pd.DataFrame([[1., 3., 101.]], columns=["000001", "600000", "510300"])
        with patch.object(runtime, "prepare_output_axis") as axis, \
                patch.object(rp, "factor_store_v5_template", return_value={"stocks": raw.columns}), \
                patch.object(rp, "read_factor_from_colblk", return_value=raw), \
                patch.object(rp, "save_factor") as save:
            runtime.write_selected_base({"a": "store"}, 20241231, 20241231)
        self.assertEqual(axis.call_args.args[2], set(raw.columns))
        np.testing.assert_array_equal(save.call_args.args[0].iloc[0], [34., 32., 66.])
        self.assertEqual(save.call_args.args[2], "a_fold")

    def test_rank_fill_before_rolling_matches_python_reference(self):
        raw = np.array([[1, 3, 100], [np.nan, 4, 50], [4, 2, 0]], dtype=np.float32)
        restrict = np.array([[0, 0, np.nan], [0, 0, np.nan], [0, 0, np.nan]], dtype=np.float32)
        ranked = pd.DataFrame(raw).rank(axis=1).astype(np.float32)
        filled = ranked.mask(ranked.isna() & (restrict == 0), ranked.median(axis=1), axis=0)
        actual = rp.tail_v5_rank_fill_roll_block_f32(raw, restrict, [3])
        np.testing.assert_array_equal(actual[:, :, 0], filled)
        np.testing.assert_allclose(actual[:, :, 1], filled.rolling(3, min_periods=1).mean(),
                                   rtol=1e-6, equal_nan=True)

    def test_colblk_removed_only_after_successful_base_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "work"
            work.mkdir()
            (work / "keep.txt").write_text("unrelated")
            store = work / "factor_store_test"
            args = dict(base_factor_ver="test_base", base_hdf5_dir=str(root / "output/base"),
                        colblk_store_dir="factor_store_test", level2_root=str(root / "inputs/l2"),
                        calendar_root=str(root / "inputs/calendar"), vars_root=str(root / "inputs/vars"))
            previous = Path.cwd()
            try:
                os.chdir(work)
                for fail in (True, False):
                    store.mkdir(exist_ok=True)
                    (store / "unfinished").write_text("preserve on failure")
                    def compute(runtime, *args):
                        self.assertEqual(Path(runtime.colblk_store_dir), store)
                        if fail:
                            raise RuntimeError("write failed")
                    with patch.object(_ProductionRuntime, "limit_resources", return_value=30), \
                            patch.object(_ProductionRuntime, "run_base", compute):
                        if fail:
                            with self.assertRaisesRegex(RuntimeError, "write failed"):
                                rp.write_selected_factor_base([], [], 20241231, 20241231, **args)
                        else:
                            rp.write_selected_factor_base([], [], 20241231, 20241231, **args)
                    self.assertEqual(store.exists(), fail)
                    self.assertEqual((work / "keep.txt").read_text(), "unrelated")
            finally:
                os.chdir(previous)

    def test_cleanup_refuses_working_directory_data_and_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "output/base"
            data.mkdir(parents=True)
            (data / "keep.h5").touch()
            link = root / "factor_store_link"
            link.symlink_to(data, target_is_directory=True)
            for path in (Path.cwd(), data, link):
                runtime = _runtime("test_base", str(data), str(root / "calendar"),
                                   str(root / "vars"), colblk_store_dir=str(path))
                with self.assertRaises(ValueError):
                    runtime.cleanup_colblk()
                self.assertTrue((data / "keep.h5").exists())


if __name__ == "__main__":
    unittest.main()
