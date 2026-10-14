import os, time
os.environ['RAYON_NUM_THREADS'] = '50'
os.environ['YUPEI_PROF'] = '1'
import rust_pyfunc as rp
print('module', rp.__file__, flush=True)
t0=time.time()
try:
    rp.py_yupei_pair(20241231, [])
except Exception:
    pass
print('OLD_UPSTREAM_SECONDS %.2f' % (time.time()-t0), flush=True)
