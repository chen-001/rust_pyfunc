import os, time
os.environ['RAYON_NUM_THREADS'] = '50'
import rust_pyfunc as rp
print('load check then start', flush=True)
t0 = time.time(); c, n, v = rp.py_yupei_pair(20241231, ["cnt_t1"]); dt1 = time.time() - t0
print('mats=[cnt_t1]  X=%d F=%d  %.2f s' % (len(c), len(n), dt1), flush=True)
t0 = time.time(); c, n, v = rp.py_yupei_pair(20241231, ["cnt_t1", "vol_t1"]); dt2 = time.time() - t0
print('mats=[cnt_t1,vol_t1]  X=%d F=%d  %.2f s' % (len(c), len(n), dt2), flush=True)
print('=> 上游 U ≈ %.2f s, 每张矩阵 C ≈ %.2f s' % (2*dt1 - dt2, dt2 - dt1), flush=True)
