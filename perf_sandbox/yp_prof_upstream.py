import os, time
os.environ['RAYON_NUM_THREADS'] = '50'
os.environ['YUPEI_PROF'] = '1'
import rust_pyfunc as rp
t0 = time.time()
try:
    rp.py_yupei_pair(20241231, [])
except Exception as e:
    print('(预期报错: %s)' % str(e)[:60])
print('UPSTREAM_SECONDS %.2f' % (time.time()-t0))
