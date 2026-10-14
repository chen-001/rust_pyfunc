import os, time, json
os.environ['RAYON_NUM_THREADS'] = '10'
import numpy as np
import rust_pyfunc as rp
print('module', rp.__file__, flush=True)
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
# 1) 只看上游：FAM 裁剪后应该明显变快（改前 10 并行 185.5 秒）
t0 = time.time()
try:
    rp.py_yupei_pair(20241231, [])
except Exception:
    pass
print('UPSTREAM_NEW_SECONDS %.2f' % (time.time() - t0), flush=True)
# 2) 默认 15 张：正确性 + 总时长
t0 = time.time()
codes, names, vals = rp.py_yupei_pair(20241231)
dt = time.time() - t0
V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
print('DEFAULT_NEW_SECONDS %.2f  shape=%s' % (dt, (V.shape,)), flush=True)
np.save(OUT + '/patched_vals.npy', V)
old = np.load(OUT + '/vals.npy', mmap_mode='r')
meta = json.load(open(OUT + '/meta.json'))
same_names = (len(names) == len(meta['names'])) and all(a == b for a, b in zip(names, meta['names']))
same_codes = (len(codes) == len(meta['codes'])) and all(a == b for a, b in zip(codes, meta['codes']))
print('names 一致:', same_names, ' codes 一致:', same_codes, flush=True)
if same_names and same_codes:
    new = np.load(OUT + '/patched_vals.npy')
    olda = np.asarray(old)
    diff = np.abs(new.astype(np.float64) - olda.astype(np.float64))
    bad = np.isfinite(diff) & (diff != 0)
    nan_mismatch = np.isnan(new) != np.isnan(olda)
    print('逐位比对：不同格数 %d / %d，NaN 位置不一致 %d' % (int(bad.sum()), new.size, int(nan_mismatch.sum())), flush=True)
    if bad.any():
        idx = np.argwhere(bad)[:5]
        for i, j in idx:
            print('   行%d 列%d 旧=%r 新=%r' % (i, j, olda[i, j], new[i, j]))
