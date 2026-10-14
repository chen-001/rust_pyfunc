import os, time, json
os.environ['RAYON_NUM_THREADS'] = '25'
os.environ['RAYON_RS_NUM_CPUS'] = '25'
import numpy as np
import rust_pyfunc as rp

OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
os.makedirs(OUT, exist_ok=True)

print('LOAD_START', open('/proc/loadavg').read().strip(), flush=True)
t0 = time.time()
codes, names, vals = rp.py_yupei_pair(20241231)
dt = time.time() - t0
print('DEFAULT_SECONDS %.2f  X=%d F=%d' % (dt, len(codes), len(names)), flush=True)

a = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
np.save(OUT + '/vals.npy', a)
with open(OUT + '/meta.json', 'w') as f:
    json.dump({'date': 20241231, 'seconds': dt, 'n_codes': len(codes),
               'n_names': len(names), 'rayon': 25, 'codes': codes, 'names': names}, f)
print('SAVED', a.shape, a.dtype, flush=True)
print('LOAD_END', open('/proc/loadavg').read().strip(), flush=True)
