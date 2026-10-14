import os, time, json, sys
os.environ['RAYON_NUM_THREADS'] = os.environ.get('YP_THREADS', '10')
import numpy as np
import rust_pyfunc as rp
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
print('module', rp.__file__, flush=True)
# 单张 cnt_t1：与改前存档按列名切片逐位比（3580007 个值）
t0 = time.time(); c1, n1, v1 = rp.py_yupei_pair(20241231, ['cnt_t1']); dt1 = time.time() - t0
print('ONE_CNT_SECONDS %.2f  n=%d F=%d' % (dt1, len(c1), len(n1)), flush=True)
V1 = np.asarray(v1, dtype=np.float32).reshape(len(c1), len(n1))
meta = json.load(open(OUT + '/meta.json')); old = np.load(OUT + '/vals.npy', mmap_mode='r')
assert len(c1) == len(meta['codes']) and all(a == b for a, b in zip(c1, meta['codes'])), 'codes 不一致'
oi = {n: i for i, n in enumerate(meta['names'])}
base = np.asarray(old)[:, [oi[n] for n in n1]]
d = np.abs(V1.astype(np.float64) - base.astype(np.float64))
print('单张逐位：不同 %d / %d，NaN 位置不一致 %d'
      % (int((np.isfinite(d) & (d != 0)).sum()), V1.size, int((np.isnan(V1) != np.isnan(base)).sum())), flush=True)
if os.environ.get('YP_FULL') == '1':
    t0 = time.time(); codes, names, vals = rp.py_yupei_pair(20241231); dt = time.time() - t0
    V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
    print('DEFAULT_SECONDS %.2f  shape=%s' % (dt, (V.shape,)), flush=True)
    ok = (len(codes) == len(meta['codes']) and all(a == b for a, b in zip(codes, meta['codes']))
          and len(names) == len(meta['names']) and all(a == b for a, b in zip(names, meta['names'])))
    print('codes/names 一致:', ok, flush=True)
    if ok:
        base = np.asarray(old); d = np.abs(V.astype(np.float64) - base.astype(np.float64))
        print('全量逐位：不同 %d / %d，NaN 位置不一致 %d'
              % (int((np.isfinite(d) & (d != 0)).sum()), V.size, int((np.isnan(V) != np.isnan(base)).sum())), flush=True)
