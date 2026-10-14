import os
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
os.environ.setdefault('MKL_NUM_THREADS', '25')
import json, re, time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

T0 = time.time()
BASE = '/home/chenzongwei/rust_pyfunc/perf_sandbox'
OUT = BASE + '/yp_out'
def log(*a):
    print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = meta['names']; codes = meta['codes']; RUNTIME = meta['seconds']
X = np.load(OUT + '/vals.npy')
N, F = X.shape
log('加载完成 N=%d F=%d 原始耗时=%.2fs' % (N, F, RUNTIME))

PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
pre_of = {}
metric_of = {}
for i, nm in enumerate(names):
    for p in PREF:
        if nm.startswith(p + '_'):
            pre_of[i] = p; metric_of[i] = nm[len(p) + 1:]; break
assert len(pre_of) == F, (len(pre_of), F)
pre_arr = np.array([pre_of[i] for i in range(F)])
met_arr = np.array([metric_of[i] for i in range(F)])
log('前缀分块 OK，每块列数 =', {p: int((pre_arr == p).sum()) for p in PREF})

# ---------- NaN 结构 ----------
fin = np.isfinite(X)
col_fin = fin.sum(0)
row_fin = fin.sum(1)
res = {'setup': {'date': 20241231, 'rayon': 25, 'N_codes': int(N), 'F_names': int(F),
                 'runtime_s': float(RUNTIME),
                 'rows_all_nan': int((row_fin == 0).sum()),
                 'rows_partial_nan': int(((row_fin > 0) & (row_fin < F)).sum()),
                 'rows_full_finite': int((row_fin == F).sum()),
                 'cols_all_nan': int((col_fin == 0).sum()),
                 'col_finite_min': int(col_fin.min()),
                 'col_finite_median': int(np.median(col_fin)),
                 'col_finite_max': int(col_fin.max())}}
log('NaN 结构:', json.dumps(res['setup'], ensure_ascii=False))

# ---------- 两两成对相关（pairwise-complete） ----------
A = np.where(fin, X, 0.0).astype(np.float64)
del X
M = fin.astype(np.float64); del fin
AT = np.ascontiguousarray(A.T); DT = np.ascontiguousarray((A * A).T); MT = np.ascontiguousarray(M.T)
log('开始 4 次矩阵乘（F=%d, N=%d）' % (F, N))
Sxy = AT @ A;               log('  A^T A 完成')
Sx = AT @ MT;               log('  A^T M 完成')
Sxx = DT @ MT;              log('  D^T M 完成')
Cn = MT @ MT;               log('  M^T M 完成')
del AT, DT, MT, A, M

corr = np.empty((F, F), np.float32)
CH = 1024
SxT = Sxy.T  # not used; keep memory sane
for s in range(0, F, CH):
    e = min(s + CH, F)
    nij = Cn[s:e]; sxij = Sx[s:e]; syij = Sx.T[s:e]
    sxxij = Sxx[s:e]; syyij = Sxx.T[s:e]; sxyij = Sxy[s:e]
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = sxyij - sxij * syij / nij
        vx = sxxij - sxij * sxij / nij
        vy = syyij - syij * syij / nij
        corr[s:e] = (cov / np.sqrt(vx * vy)).astype(np.float32)
np.fill_diagonal(corr, np.nan)
log('相关矩阵完成，形状', corr.shape, '内存 %.2f GB' % (corr.nbytes / 1e9))
np.save(OUT + '/corr.npy', corr)
np.save(OUT + '/pre_idx.npy', pre_arr.astype('U16'))

iu = np.triu_indices(F, 1)
absv = np.abs(corr[iu])
res['corr_global'] = {
    'pairs': int(len(absv)),
    'abs_mean': float(np.nanmean(absv)),
    'abs_median': float(np.nanmedian(absv)),
    'abs_p90': float(np.nanpercentile(absv, 90)),
    'abs_p99': float(np.nanpercentile(absv, 99)),
    'abs_p999': float(np.nanpercentile(absv, 99.9)),
    'abs_max': float(np.nanmax(absv)),
    'share_gt095': float((absv > 0.95).mean()),
    'share_gt099': float((absv > 0.99).mean()),
    'share_gt0999': float((absv > 0.999).mean()),
    'signed_gt099': float((corr[iu] > 0.99).mean()),
}
log('全局 |corr| 分布:', json.dumps(res['corr_global'], ensure_ascii=False))
json.dump(res, open(OUT + '/analysis_part1.json', 'w'), ensure_ascii=False, indent=1)
log('part1 已保存')
