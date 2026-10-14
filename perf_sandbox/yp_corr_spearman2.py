import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
import numpy as np
from scipy.stats import rankdata
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)
X = np.load(OUT + '/vals.npy')
N, F = X.shape
Rk = rankdata(X, method='average', axis=0, nan_policy='omit')
del X
log('排名完成')
fin = np.isfinite(Rk)
A = np.where(fin, Rk, 0.0).astype(np.float64); del Rk
M = fin.astype(np.float64); del fin
AT = np.ascontiguousarray(A.T); MT = np.ascontiguousarray(M.T)
D = A * A
Sxy = AT @ A; log('Sxy')
Sx = AT @ M; log('Sx')
Sxx = D.T.copy() @ M; log('Sxx'); del D
Cn = MT @ M; log('Cn'); del AT, MT, A, M
C = np.empty((F, F), np.float32)
SxT = Sx.T; SxxT = Sxx.T
for s in range(0, F, 1024):
    e = min(s + 1024, F)
    nij = Cn[s:e]; sxij = Sx[s:e]; syij = SxT[s:e]
    sxxij = Sxx[s:e]; syyij = SxxT[s:e]
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = Sxy[s:e] - sxij * syij / nij
        vx = sxxij - sxij * sxij / nij
        vy = syyij - syij * syij / nij
        C[s:e] = (cov / np.sqrt(vx * vy)).astype(np.float32)
np.fill_diagonal(C, np.nan)
np.save(OUT + '/corr_spearman.npy', C)
log('Spearman 完成')
ov = np.load(OUT + '/overlap.npy')
bad = ~np.isfinite(C) | (ov < 500)
C[bad] = np.nan
iu = np.triu_indices(F, 1); v = np.abs(C[iu])
res = {'abs_mean': float(np.nanmean(v)), 'abs_median': float(np.nanmedian(v)),
       'abs_p99': float(np.nanpercentile(v, 99)), 'abs_max': float(np.nanmax(v)),
       'share_gt095': float((v > 0.95).mean()), 'share_gt099': float((v > 0.99).mean()),
       'share_gt0999': float((v > 0.999).mean())}
log('Spearman 全局: %s' % json.dumps(res))
json.dump(res, open(OUT + '/spearman_global.json', 'w'), indent=1)
log('DONE')
