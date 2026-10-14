import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
import numpy as np
from scipy.stats import rankdata

T0 = time.time()
BASE = '/home/chenzongwei/rust_pyfunc/perf_sandbox'
OUT = BASE + '/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = np.array(meta['names']); RUNTIME = meta['seconds']
X = np.load(OUT + '/vals.npy')
N, F = X.shape
log('排名（Spearman 用），shape', X.shape)
Rk = rankdata(X, method='average', axis=0, nan_policy='omit').astype(np.float64)
del X
log('排名完成')

fin = np.isfinite(Rk)
A = np.where(fin, Rk, 0.0); M = fin.astype(np.float64)
del Rk
AT = np.ascontiguousarray(A.T); DT = np.ascontiguousarray((A * A).T); MT = np.ascontiguousarray(M.T)
Sxy = AT @ A; log('  A^T A')
Sx = AT @ MT;  log('  A^T M')
Sxx = DT @ MT; log('  D^T M')
Cn = MT @ MT;  log('  M^T M')
del AT, DT, MT, A, M
R = {}
C = np.empty((F, F), np.float32)
for s in range(0, F, 1024):
    e = min(s + 1024, F)
    nij = Cn[s:e]; sxij = Sx[s:e]; syij = Sx.T[s:e]
    sxxij = Sxx[s:e]; syyij = Sxx.T[s:e]; sxyij = Sxy[s:e]
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = sxyij - sxij * syij / nij
        vx = sxxij - sxij * sxij / nij
        vy = syyij - syij * syij / nij
        C[s:e] = (cov / np.sqrt(vx * vy)).astype(np.float32)
np.fill_diagonal(C, np.nan)
np.save(OUT + '/corr_spearman.npy', C)
log('Spearman 相关矩阵完成')
iu = np.triu_indices(F, 1); v = np.abs(C[iu])
R['spearman_global'] = {'abs_mean': float(np.nanmean(v)), 'abs_median': float(np.nanmedian(v)),
                        'abs_p99': float(np.nanpercentile(v, 99)), 'abs_max': float(np.nanmax(v)),
                        'share_gt095': float((v > 0.95).mean()), 'share_gt099': float((v > 0.99).mean())}

def greedy(corr, thr, absv=True):
    Cx = np.abs(corr) if absv else corr
    alive = np.ones(F, bool); groups = []
    for i in range(F):
        if not alive[i]: continue
        cand = np.nonzero(alive & (Cx[i] >= thr))[0]
        cand = cand[cand > i]
        if cand.size:
            alive[cand] = False; groups.append((int(i), cand))
    return alive, groups
R['spearman_greedy'] = {}
for thr in (0.995, 0.99, 0.95):
    alive, groups = greedy(C, thr, True)
    k = int((~alive).sum()); sizes = sorted([len(g[1]) + 1 for g in groups], reverse=True)
    R['spearman_greedy']['%.3f_abs' % thr] = {'cols_dropped': k, 'share_cols': k / F,
                                              'proportional_seconds': k / F * RUNTIME,
                                              'groups': len(groups), 'max_group': sizes[0] if sizes else 0}
    log('Spearman 贪心 %.3f: 丢 %d (%.1f%%)' % (thr, k, 100 * k / F))
json.dump(R, open(OUT + '/analysis_spearman.json', 'w'), ensure_ascii=False, indent=1)
log('Spearman 阶段完成')
