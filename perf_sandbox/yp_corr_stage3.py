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
names = np.array(meta['names']); RUNTIME = meta['seconds']
corr = np.load(OUT + '/corr.npy')
pre_arr = np.load(OUT + '/pre_idx.npy').astype(str)
F = corr.shape[0]
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
R = json.load(open(OUT + '/analysis_part2.json'))

def edges(thr, absv=True):
    rs, cs = [], []
    for s in range(0, F, 512):
        e = min(s + 512, F)
        sub = corr[s:e, s:]
        if absv: sub = np.abs(sub)
        ii, jj = np.nonzero(sub >= thr)
        ii = ii + s; jj = jj + s
        k = jj > ii
        if k.any(): rs.append(ii[k]); cs.append(jj[k])
    if not rs: return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(rs), np.concatenate(cs)

R['clusters'] = {}
for thr in (0.999, 0.995, 0.99, 0.95):
    r, c = edges(thr)
    g = coo_matrix((np.ones(len(r), np.int8), (r, c)), shape=(F, F))
    nc, lab = connected_components(g, directed=False)
    sizes = np.bincount(lab)
    big = sizes[sizes >= 3]
    R['clusters']['%.3f' % thr] = {
        'edges': int(len(r)), 'components_total': int(nc),
        'singletons': int((sizes == 1).sum()),
        'comp_ge3': int(len(big)), 'cols_in_comp_ge3': int(big.sum()),
        'max_comp': int(sizes.max()),
        'top_sizes': sorted([int(x) for x in sizes if x >= 3], reverse=True)[:15],
        'cols_in_comp_ge2': int(sizes[sizes >= 2].sum())}
    log('阈值 %.3f: 边=%d 连通块(>=3)=%d 涉及列=%d 最大块=%d' %
        (thr, len(r), len(big), big.sum(), sizes.max()))
    np.save(OUT + '/label_%.3f.npy' % thr, lab)

# ---------- 贪心去重（星形，无链式） ----------
def greedy(thr, absv=True):
    C = np.abs(corr) if absv else corr
    alive = np.ones(F, bool)
    groups = []
    for i in range(F):
        if not alive[i]: continue
        cand = np.nonzero(alive & (C[i] >= thr))[0]
        cand = cand[cand > i]
        if cand.size:
            alive[cand] = False
            groups.append((int(i), cand))
    return alive, groups

R['greedy'] = {}
for thr in (0.999, 0.995, 0.99, 0.95):
    for tag, absv in (('abs', True), ('pos', False)):
        alive, groups = greedy(thr, absv)
        k = int((~alive).sum())
        dropped_by_pre = {p: int((~alive & (pre_arr == p)).sum()) for p in PREF}
        sizes = sorted([len(g[1]) + 1 for g in groups], reverse=True)
        R['greedy']['%.3f_%s' % (thr, tag)] = {
            'cols_dropped': k, 'cols_left': int(alive.sum()),
            'share_cols': k / F, 'proportional_seconds': k / F * RUNTIME,
            'groups': len(groups), 'max_group': int(sizes[0]) if sizes else 0,
            'top_group_sizes': sizes[:15],
            'dropped_by_matrix': dropped_by_pre,
            'dropped_by_matrix_frac': {p: round(v / ((pre_arr == p).sum()), 4)
                                       for p, v in dropped_by_pre.items()}}
        log('贪心 %.3f %s: 丢 %d 列 (%.1f%%) => 比例口径省 %.1f s; 最大组 %d' %
            (thr, tag, k, 100 * k / F, k / F * RUNTIME, sizes[0] if sizes else 0))
        np.save(OUT + '/alive_%.3f_%s.npy' % (thr, tag), alive)
        if thr == 0.99 and tag == 'abs':
            reps = []
            for i, cand in groups:
                sub = np.abs(corr[i, cand])
                reps.append({'rep': names[i], 'members': list(names[cand][:12]),
                             'size': len(cand) + 1, 'min_abs': float(sub.min()),
                             'max_abs': float(sub.max())})
            reps.sort(key=lambda d: -d['size'])
            R['greedy']['%.3f_%s' % (thr, tag)]['largest_groups'] = reps[:15]

# ---------- 3. 单块 τ 冗余度 ----------
R['tau_per_matrix_redundancy'] = {}
met = np.array([n[len(p)+1:] for n, p in zip(names, pre_arr)])
pos = {}
for i in range(F): pos.setdefault(met[i], {})[pre_arr[i]] = i
full_metrics = [m for m in pos if len(pos[m]) == 15]
for p in PREF:
    mx = []
    for m in full_metrics:
        c = np.abs(corr[pos[m][p], [pos[m][q] for q in PREF if q != p]])
        mx.append(np.nanmax(c))
    R['tau_per_matrix_redundancy'][p] = {'avg_max_abs_corr': float(np.mean(mx)),
                                         'share_max_gt099': float((np.array(mx) > 0.99).mean())}
log('每张矩阵 τ 冗余度:');
for p in PREF: log('  ', p, R['tau_per_matrix_redundancy'][p])

json.dump(R, open(OUT + '/analysis_part3.json', 'w'), ensure_ascii=False, indent=1)
log('part3 已保存')
