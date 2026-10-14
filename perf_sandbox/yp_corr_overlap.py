import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
import numpy as np
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)
X = np.load(OUT + '/vals.npy')
log('load ok')
mas = np.isfinite(X).astype(np.float32)
del X
log('mask ok')
Cn = mas.T @ mas
log('M^T M ok')
np.save(OUT + '/overlap.npy', np.rint(Cn).astype(np.int32))
log('overlap saved', Cn.shape, 'min', float(Cn.min()), 'median', float(np.median(Cn)))
