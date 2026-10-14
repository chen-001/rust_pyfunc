import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)
meta = json.load(open(OUT + '/meta.json'))
names = np.array(meta['names']); RUNTIME = meta['seconds']
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
F = len(names)
pre_arr = np.zeros(F, np.int8); met_arr = np.empty(F, object)
for i, nm in enumerate(names):
    for k, p in enumerate(PREF):
        if nm.startswith(p + '_'): pre_arr[i] = k; met_arr[i] = nm[len(p)+1:]; break
corr = np.load(OUT + '/corr.npy')
ov = np.load(OUT + '/overlap.npy')
res = {'setup': {'date': 20241231, 'rayon_threads': 25, 'N_codes': int(N := 5107), 'F_names': int(F),
                 'py_yupei_pair_seconds': float(RUNTIME), 'corr_seconds': None}}
bad = ~np.isfinite(corr) | (ov < 500)
res['dropped_pairs'] = {'nonfinite_or_low_overlap': int(bad.sum())}
corr[bad] = np.nan
np.fill_diagonal(corr, np.nan)
res['overlap'] = {'min': int(ov.min()), 'median': int(np.median(ov)), 'max': int(ov.max()),
                  'share_pairs_ge500': float((ov >= 500).mean())}
iu = np.triu_indices(F, 1); v = np.abs(corr[iu]); sv = corr[iu]
res['global'] = {'abs_mean': float(np.nanmean(v)), 'abs_median': float(np.nanmedian(v)),
                 'abs_p99': float(np.nanpercentile(v, 99)), 'abs_p999': float(np.nanpercentile(v, 99.9)),
                 'abs_max': float(np.nanmax(v)),
                 'share_gt095': float((v > 0.95).mean()), 'share_gt099': float((v > 0.99).mean()),
                 'share_gt0999': float((v > 0.999).mean()), 'share_gt09999': float((v > 0.9999).mean()),
                 'signed_gt095': float((sv > 0.95).mean()), 'signed_gt099': float((sv > 0.99).mean()),
                 'n_pairs': int(len(v))}
log('global: %s' % json.dumps(res['global']))

# 每张矩阵块内
res['within_prefix'] = {}
for k, p in enumerate(PREF):
    idx = np.where(pre_arr == k)[0]
    sub = np.abs(corr[np.ix_(idx, idx)]); j = np.triu_indices(len(idx), 1); w = sub[j]
    res['within_prefix'][p] = {'n_cols': int(len(idx)), 'abs_mean': float(np.nanmean(w)),
        'abs_median': float(np.nanmedian(w)), 'abs_p99': float(np.nanpercentile(w, 99)),
        'abs_max': float(np.nanmax(w)), 'share_gt095': float((w > 0.95).mean()),
        'share_gt099': float((w > 0.99).mean())}
log('块内统计完成')

# tau 轴
pos = {}
for i in range(F): pos.setdefault(met_arr[i], {})[PREF[pre_arr[i]]] = i
full = [m for m in pos if len(pos[m]) == 15]
tm = np.zeros((15, 15)); adj = []; far = []; allo = []
for m in full:
    idx = np.array([pos[m][p] for p in PREF])
    c = np.abs(corr[np.ix_(idx, idx)]); tm += np.nan_to_num(c)
    x = np.triu_indices(15, 1); w = c[x]
    d = np.abs(np.subtract.outer(np.arange(15), np.arange(15)))[x]
    allo.append(w); adj.append(w[d == 1]); far.append(w[d >= 3])
tm /= len(full); np.fill_diagonal(tm, np.nan)
adj = np.concatenate(adj); far = np.concatenate(far); allo = np.concatenate(allo)
# 每张矩阵被同指标其它 tau 替代的程度
per_p = {}
for k, p in enumerate(PREF):
    mx = []
    for m in full:
        c = np.abs(corr[pos[m][p], [pos[m][q] for q in PREF if q != p]])
        mx.append(np.nanmax(c))
    mx = np.array(mx)
    per_p[p] = {'avg_max_abs_corr': float(np.nanmean(mx)), 'share_max_gt099': float((mx > 0.99).mean()),
                'share_max_gt0999': float((mx > 0.999).mean())}
res['tau_axis'] = {'n_metrics': len(full), 'adjacent_abs_mean': float(np.nanmean(adj)),
    'adjacent_gt099': float((adj > 0.99).mean()), 'all_abs_mean': float(np.nanmean(allo)),
    'all_gt095': float((allo > 0.95).mean()), 'all_gt099': float((allo > 0.99).mean()),
    'far_abs_mean': float(np.nanmean(far)), 'far_gt099': float((far > 0.99).mean()),
    'mean_abs_matrix': tm.tolist(), 'prefixes': PREF, 'per_prefix': per_p}
log('tau 轴: 相邻 %.4f 全对 %.4f 间隔>=3 %.4f' % (np.nanmean(adj), np.nanmean(allo), np.nanmean(far)))
print(np.round(tm, 3))

# 贪心去重
def greedy(C, thr, absv=True):
    Cx = np.abs(C) if absv else C
    alive = np.ones(F, bool); groups = []
    for i in range(F):
        if not alive[i]: continue
        cand = np.nonzero(alive & (Cx[i] >= thr))[0]
        cand = cand[cand > i]
        if cand.size: alive[cand] = False; groups.append((int(i), cand))
    return alive, groups
res['greedy'] = {}
for thr in (0.9999, 0.999, 0.99, 0.95):
    for tag, absv in (('abs', True), ('pos', False)):
        alive, groups = greedy(corr, thr, absv)
        k = int((~alive).sum()); sizes = sorted([len(g[1]) + 1 for g in groups], reverse=True)
        d = {p: int((~alive & (pre_arr == k2)).sum()) for k2, p in enumerate(PREF)}
        res['greedy']['%.4f_%s' % (thr, tag)] = {'cols_dropped': k, 'cols_left': int(alive.sum()),
            'share_cols': k / F, 'proportional_seconds': k / F * RUNTIME, 'groups': len(groups),
            'max_group': sizes[0] if sizes else 0, 'top_group_sizes': sizes[:12],
            'dropped_by_matrix': d}
        log('贪心 %.4f %s: 丢 %d (%.2f%%) 省比例口径 %.1f s 组数 %d 最大 %d' %
            (thr, tag, k, 100*k/F, k/F*RUNTIME, len(groups), sizes[0] if sizes else 0))
        np.save(OUT + '/alive_%.4f_%s.npy' % (thr, tag), alive)
        if thr in (0.99, 0.999) and tag == 'abs':
            reps = []
            for i, cand in groups:
                if len(cand) + 1 < 3: continue
                sub = np.abs(corr[i, cand])
                reps.append({'rep': names[i], 'members': list(names[cand][:15]), 'size': len(cand)+1,
                             'min_abs': float(np.nanmin(sub)), 'max_abs': float(np.nanmax(sub))})
            reps.sort(key=lambda x: -x['size'])
            res['greedy']['%.4f_%s' % (thr, tag)]['largest_groups'] = reps[:20]
        if thr == 0.99 and tag == 'abs':
            res['exact_dup_pairs'] = int((v > 0.9999).sum())
json.dump(res, open(OUT + '/analysis.json', 'w'), ensure_ascii=False, indent=1, default=str)
log('DONE')
