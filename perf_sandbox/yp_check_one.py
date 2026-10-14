import os, time, json
os.environ['RAYON_NUM_THREADS'] = '10'
import numpy as np
import rust_pyfunc as rp
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
print('module', rp.__file__, flush=True)
t0=time.time(); codes, names, vals = rp.py_yupei_pair(20241231, ['cnt_t1']); dt=time.time()-t0
V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
print('ONE_SECONDS %.2f  n=%d F=%d' % (dt, len(codes), len(names)), flush=True)
meta = json.load(open(OUT+'/meta.json'))
old = np.load(OUT+'/vals.npy', mmap_mode='r')
oi = {n:i for i,n in enumerate(meta['names'])}
assert len(codes)==len(meta['codes']) and all(a==b for a,b in zip(codes, meta['codes'])), 'codes 不一致'
idx = [oi[n] for n in names]
base = np.asarray(old)[:, idx]
d = np.abs(V.astype(np.float64) - base.astype(np.float64))
bad = np.isfinite(d) & (d != 0)
nanmis = np.isnan(V) != np.isnan(base)
print('单张逐位比对：不同 %d / %d，NaN 位置不一致 %d' % (int(bad.sum()), V.size, int(nanmis.sum())), flush=True)
