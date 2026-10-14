import os, time, json
os.environ['RAYON_NUM_THREADS'] = '50'
import numpy as np
import rust_pyfunc as rp
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
print('module', rp.__file__, flush=True)
t0=time.time(); codes, names, vals = rp.py_yupei_pair(20241231); dt=time.time()-t0
V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
print('FIFTY_DEFAULT_SECONDS %.2f  n=%d F=%d' % (dt, len(codes), len(names)), flush=True)
meta=json.load(open(OUT+'/meta.json')); old=np.load(OUT+'/vals.npy', mmap_mode='r')
ok = (len(codes)==len(meta['codes']) and all(a==b for a,b in zip(codes,meta['codes']))
      and len(names)==len(meta['names']) and all(a==b for a,b in zip(names,meta['names'])))
print('codes/names 一致:', ok, flush=True)
if ok:
    base=np.asarray(old)
    d=np.abs(V.astype(np.float64)-base.astype(np.float64))
    bad=np.isfinite(d)&(d!=0)
    nanmis=np.isnan(V)!=np.isnan(base)
    print('全量逐位：不同 %d / %d，NaN 位置不一致 %d' % (int(bad.sum()), V.size, int(nanmis.sum())), flush=True)
