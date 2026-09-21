"""准备前向收益缓存：r5 = gap5(20160101..20251231)，r1 = gap1(...)。
输出 returns_cache.npz：dates(int64), codes(str), r5(float32), r1(float32)。
"""
import sys, numpy as np

sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as dr

OUT = "/home/chenzongwei/rust_pyfunc/tmp_mono_metric/empirics/returns_cache.npz"

r5 = dr.gap5(20160101, 20251231)
r1 = dr.gap1(20160101, 20251231)
assert list(r5.index) == list(r1.index) and list(r5.columns) == list(r1.columns)

dates = np.asarray(r5.index, dtype=np.int64)
codes = np.asarray([str(c) for c in r5.columns])
a5 = np.ascontiguousarray(r5.to_numpy(dtype=np.float32))
a1 = np.ascontiguousarray(r1.to_numpy(dtype=np.float32))

np.savez(OUT, dates=dates, codes=codes, r5=a5, r1=a1)
print("saved", OUT, a5.shape, "NaN frac r5 =", float(np.isnan(a5).mean()))
