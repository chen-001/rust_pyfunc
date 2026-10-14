import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
import numpy as np
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = meta['names']; RUNTIME = meta['seconds']
X = np.load(OUT + '/vals.npy')
N, F = X.shape
log('加载完成 N=%d F=%d' % (N, F))
fin = np.isfinite(X)
A = np.where(fin, X, 0.0).astype(np.float64)
del X
M = fin.astype(np.float64)
log('A/M 就绪')
AT = np.ascontiguousarray(A.T); MT = np.ascontiguousarray(M.T)
D = A * A
log('转置就绪')
Sxy = AT @ A;   log('Sxy = A^T A 完成')
Sx  = AT @ M;   log('Sx  = A^T M 完成')
Sxx = D.T.copy() @ M; log('Sxx = D^T M 完成')
del D
Cn  = MT @ M;   log('Cn  = M^T M 完成')
del AT, MT, A, M
np.save(OUT + '/overlap.npy', np.rint(Cn).astype(np.int16))
log('overlap 已保存')

corr = np.empty((F, F), np.float32)
SxT = Sx.T; SxxT = Sxx.T
for s in range(0, F, 1024):
    e = min(s + 1024, F)
    nij = Cn[s:e]; sxij = Sx[s:e]; syij = SxT[s:e]
    sxxij = Sxx[s:e]; syyij = SxxT[s:e]
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = Sxy[s:e] - sxij * syij / nij
        vx = sxxij - sxij * sxij / nij
        vy = syyij - syij * syij / nij
        corr[s:e] = (cov / np.sqrt(vx * vy)).astype(np.float32)
    if s % 4096 == 0: log('  corr 行 %d / %d' % (s, F))
np.fill_diagonal(corr, np.nan)
np.save(OUT + '/corr.npy', corr)
log('corr 已保存')
iu = np.triu_indices(F, 1); v = np.abs(corr[iu])
res = {'abs_mean': float(np.nanmean(v)), 'abs_median': float(np.nanmedian(v)),
       'abs_p99': float(np.nanpercentile(v, 99)), 'abs_p999': float(np.nanpercentile(v, 99.9)),
       'abs_max': float(np.nanmax(v)),
       'share_gt095': float((v > 0.95).mean()), 'share_gt099': float((v > 0.99).mean()),
       'share_gt0999': float((v > 0.999).mean()), 'share_gt09999': float((v > 0.9999).mean()),
       'signed_gt099': float((corr[iu] > 0.99).mean()), 'signed_gt095': float((corr[iu] > 0.95).mean())}
log('全局: %s' % json.dumps(res, ensure_ascii=False))
json.dump(res, open(OUT + '/corr_global.json', 'w'), ensure_ascii=False, indent=1)
log('DONE')
