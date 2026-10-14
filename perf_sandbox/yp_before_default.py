import os, time, json
os.environ['RAYON_NUM_THREADS'] = '10'
import numpy as np
import rust_pyfunc as rp
print('module', rp.__file__, flush=True)
t0 = time.time()
codes, names, vals = rp.py_yupei_pair(20241231)
dt = time.time() - t0
print('BEFORE_DEFAULT_SECONDS %.2f  n=%d F=%d' % (dt, len(codes), len(names)), flush=True)
