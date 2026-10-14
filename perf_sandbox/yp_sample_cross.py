import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
import numpy as np
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = meta['names']
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
F = len(names)
pre_arr = np.zeros(F, np.int8); met_arr = np.zeros(F, np.int32)
for i, nm in enumerate(names):
    for k, p in enumerate(PREF):
        if nm.startswith(p + '_'): pre_arr[i] = k; break
X = np.load(OUT + '/vals.npy')
fin = np.isfinite(X)
Xf = np.where(fin, X, 0.0).astype(np.float64)
del X
log('ready')

rng = np.random.default_rng(20241231)
S = 600000
I = rng.integers(0, F, 2 * S); J = rng.integers(0, F, 2 * S)
keep = (pre_arr[I] != pre_arr[J]) & (I != J)
I = I[keep]; J = J[keep]
log('跨矩阵且不同指标候选 %d（用其中 %d）' % (len(I), S))
I = I[:S]; J = J[:S]

def sample_corr(I, J, chunk=6000):
    rs = []; ns = []
    for s in range(0, len(I), chunk):
        i = I[s:s+chunk]; j = J[s:s+chunk]
        b = fin[:, i] & fin[:, j]
        xi = Xf[:, i] * b; xj = Xf[:, j] * b
        n = b.sum(0)
        sx = xi.sum(0); sy = xj.sum(0)
        sxy = (xi * xj).sum(0); sxx = (xi * xi).sum(0); syy = (xj * xj).sum(0)
        with np.errstate(invalid='ignore', divide='ignore'):
            cov = sxy - sx * sy / n; vx = sxx - sx * sx / n; vy = syy - sy * sy / n
            r = cov / np.sqrt(vx * vy)
        rs.append(r); ns.append(n)
    return np.concatenate(rs), np.concatenate(ns)

r, n = sample_corr(I, J)
good = n >= 500
rg = np.abs(r[good])
R = {'sample_total': int(len(r)), 'usable_n>=500': int(good.sum()),
     'n_median': int(np.median(n)),
     'abs_mean': float(np.nanmean(rg)), 'abs_median': float(np.nanmedian(rg)),
     'abs_p99': float(np.nanpercentile(rg, 99)), 'abs_max': float(np.nanmax(rg)),
     'share_gt08': float((rg > 0.8).mean()), 'share_gt09': float((rg > 0.9).mean()),
     'share_gt095': float((rg > 0.95).mean()), 'share_gt099': float((rg > 0.99).mean()),
     'count_gt099': int((rg > 0.99).sum()), 'count_gt095': int((rg > 0.95).sum())}
log('跨矩阵不同指标采样: %s' % json.dumps(R, ensure_ascii=False))
# 同一指标跨矩阵的对照（抽样，非全量）
ii = np.arange(0, F, 13)
m2 = {}
for k, p in enumerate(PREF):
    m2[p] = np.where(pre_arr == k)[0]
pairs_i = []; pairs_j = []
rng2 = np.random.default_rng(5)
for _ in range(200000):
    a = rng2.integers(0, F); b = rng2.integers(0, F)
    if pre_arr[a] != pre_arr[b] and names[a][len(PREF[pre_arr[a]])+1:] == names[b][len(PREF[pre_arr[b]])+1:]:
        pairs_i.append(a); pairs_j.append(b)
r2, n2 = sample_corr(np.array(pairs_i), np.array(pairs_j))
R['tau_pairs_sampled'] = int(len(r2))
for t in (0.95, 0.99):
    R['tau_share_gt%s' % str(t).replace('.', '')] = float((np.abs(r2) > t).mean())
R['tau_abs_mean'] = float(np.nanmean(np.abs(r2)))
log('同指标跨矩阵抽样: n=%d abs_mean=%.4f >0.99=%.4f' % (len(r2), R['tau_abs_mean'], R['tau_share_gt099']))
json.dump(R, open(OUT + '/sample_cross.json', 'w'), ensure_ascii=False, indent=1)
log('DONE')
