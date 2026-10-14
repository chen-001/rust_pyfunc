import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

T0 = time.time()
BASE = '/home/chenzongwei/rust_pyfunc/perf_sandbox'
OUT = BASE + '/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = meta['names']; RUNTIME = meta['seconds']
corr = np.load(OUT + '/corr.npy')
pre_arr = np.load(OUT + '/pre_idx.npy').astype(str)
F = corr.shape[0]
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
met = np.array([n[len(p)+1:] for n, p in zip(names, pre_arr)])
X = np.load(OUT + '/vals.npy')

R = {'setup': json.load(open(OUT + '/analysis_part1.json'))['setup']}

# ---------- 1. 每个矩阵块内（701×701） ----------
blk = {}
for p in PREF:
    idx = np.where(pre_arr == p)[0]
    sub = np.abs(corr[np.ix_(idx, idx)])
    iu = np.triu_indices(len(idx), 1)
    v = sub[iu]
    blk[p] = {'n_cols': len(idx), 'abs_mean': float(np.nanmean(v)), 'abs_median': float(np.nanmedian(v)),
              'abs_p99': float(np.nanpercentile(v, 99)), 'abs_max': float(np.nanmax(v)),
              'share_gt095': float((v > 0.95).mean()), 'share_gt099': float((v > 0.99).mean()),
              'share_gt0995': float((v > 0.995).mean())}
    log('块内 %s %s' % (p, json.dumps(blk[p])))
R['within_matrix_block'] = blk

# ---------- 2. tau 轴：同一指标名跨 15 张矩阵 ----------
pos = {}
for i in range(F):
    pos.setdefault(met[i], {})[pre_arr[i]] = i
full_metrics = [m for m in pos if len(pos[m]) == 15]
log('跨 15 张都存在的指标名 %d / %d' % (len(full_metrics), len(pos)))
tau_mean = np.zeros((15, 15)); tau_cnt = np.zeros((15, 15))
adj_abs = []; all_abs = []; far_abs = []
per_metric_max = []
for m in full_metrics:
    idx = np.array([pos[m][p] for p in PREF])
    c = np.abs(corr[np.ix_(idx, idx)])
    tau_mean += np.nan_to_num(c); tau_cnt += 1
    iu = np.triu_indices(15, 1)
    v = c[iu]; d = np.abs(np.subtract.outer(np.arange(15), np.arange(15)))[iu]
    all_abs.append(v); adj_abs.append(v[d == 1]); far_abs.append(v[d >= 3])
    k = np.nanargmax(np.where(np.eye(15, dtype=bool), np.nan, c))
    per_metric_max.append((float(c.flat[k]), m, PREF[k // 15], PREF[k % 15]))
tau_mean /= np.maximum(tau_cnt, 1)
np.fill_diagonal(tau_mean, np.nan)
adj = np.concatenate(adj_abs); al = np.concatenate(all_abs); far = np.concatenate(far_abs)
R['tau_axis'] = {
    'metrics': len(full_metrics),
    'adjacent_abs_mean': float(np.mean(adj)), 'adjacent_gt095': float((adj > 0.95).mean()),
    'adjacent_gt099': float((adj > 0.99).mean()),
    'all_pairs_abs_mean': float(np.mean(al)), 'all_pairs_gt095': float((al > 0.95).mean()),
    'all_pairs_gt099': float((al > 0.99).mean()),
    'far_pairs_abs_mean': float(np.mean(far)), 'far_pairs_gt095': float((far > 0.95).mean()),
    'mean_abs_matrix': tau_mean.tolist(), 'prefixes': PREF}
per_metric_max.sort(reverse=True)
R['tau_axis']['top_metric_pairs'] = [{'abs_corr': a, 'metric': m, 'mat_a': pa, 'mat_b': pb}
                                     for a, m, pa, pb in per_metric_max[:20]]
log('tau 轴: 相邻平均|corr|=%.4f 全对平均=%.4f 间隔>=3 平均=%.4f' %
    (R['tau_axis']['adjacent_abs_mean'], R['tau_axis']['all_pairs_abs_mean'], R['tau_axis']['far_pairs_abs_mean']))
log('tau 邻接矩阵(平均|corr|):'); print(np.round(tau_mean, 3))

# ---------- 3. 指标族 ----------
toks = {}
for m in met:
    t = m.split('_')[0]
    toks[t] = toks.get(t, 0) + 1
open(OUT + '/metric_names.txt', 'w').write('\n'.join(sorted(set(met))))
R['metric_first_tokens'] = dict(sorted(toks.items(), key=lambda kv: -kv[1]))
log('首 token 直方图:', json.dumps(R['metric_first_tokens'], ensure_ascii=False))

json.dump(R, open(OUT + '/analysis_part2.json', 'w'), ensure_ascii=False, indent=1)
log('part2 已保存')
