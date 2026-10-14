import os, time
os.environ['RAYON_NUM_THREADS'] = '50'
os.environ['YUPEI_PROF'] = '1'
import rust_pyfunc as rp
print('module:', rp.__file__, flush=True)
t0 = time.time()
c, n, v = rp.py_yupei_pair(20241231, ["cnt_t1"])
print('TOTAL_SECONDS %.2f  X=%d F=%d' % (time.time()-t0, len(c), len(n)), flush=True)
