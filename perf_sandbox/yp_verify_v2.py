import os, time, json
os.environ['RAYON_NUM_THREADS'] = '10'
import numpy as np
import rust_pyfunc as rp
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
print('module', rp.__file__, flush=True)
# 1) 单张：先看泛化后的裁剪有没有生效（改前同条件 310.8 秒）
t0 = time.time(); c1, n1, v1 = rp.py_yupei_pair(20241231, ['cnt_t1']); dt1 = time.time() - t0
print('ONE_CNT_SECONDS %.2f  F=%d' % (dt1, len(n1)), flush=True)
# 2) 默认 15 张：正确性 + 总时长（改前同条件 1852.28 秒）
t0 = time.time(); codes, names, vals = rp.py_yupei_pair(20241231); dt = time.time() - t0
V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
print('DEFAULT_SECONDS %.2f  shape=%s' % (dt, (V.shape,)), flush=True)
old = np.load(OUT + '/vals.npy', mmap_mode='r')
meta = json.load(open(OUT + '/meta.json'))
ok_n = (len(names) == len(meta['names'])) and all(a == b for a, b in zip(names, meta['names']))
ok_c = (len(codes) == len(meta['codes'])) and all(a == b for a, b in zip(codes, meta['codes']))
print('names/codes 一致:', ok_n, ok_c, flush=True)
if ok_n and ok_c:
    new = V.astype(np.float64); olda = np.asarray(old).astype(np.float64)
    d = np.abs(new - olda)
    bad = np.isfinite(d) & (d != 0)
    nanmis = np.isnan(V) != np.isnan(np.asarray(old))
    print('逐位比对：不同 %d / %d，NaN 位置不一致 %d' % (int(bad.sum()), V.size, int(nanmis.sum())), flush=True)
