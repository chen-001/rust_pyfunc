import os, time
os.environ['RAYON_NUM_THREADS'] = '10'
import rust_pyfunc as rp
print('module', rp.__file__, flush=True)
t0 = time.time(); c, n, v = rp.py_yupei_pair(20241231, ['cnt_t1']); dt = time.time() - t0
print('PROBE_ONE_SECONDS %.2f  n=%d F=%d' % (dt, len(c), len(n)), flush=True)
