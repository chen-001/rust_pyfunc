import json
import numpy as np
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta = json.load(open(OUT + '/meta.json')); names = np.array(meta['names']); RUNTIME = meta['seconds']
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
F = len(names)
pre_arr = np.zeros(F, np.int8)
for i, nm in enumerate(names):
    for k, p in enumerate(PREF):
        if nm.startswith(p + '_'): pre_arr[i] = k; break
ov = np.load(OUT + '/overlap.npy')
C = np.load(OUT + '/corr_spearman.npy')
C[~np.isfinite(C) | (ov < 500)] = np.nan
np.fill_diagonal(C, np.nan)
def greedy(Cf, thr):
    alive = np.ones(F, bool); groups = []
    for i in range(F):
        if not alive[i]: continue
        cand = np.nonzero(alive & (Cf[i] >= thr))[0]; cand = cand[cand > i]
        if cand.size: alive[cand] = False; groups.append((int(i), cand))
    return alive, groups
A = np.abs(C)
out = {}
for thr in (0.9999, 0.999, 0.99, 0.95):
    alive, groups = greedy(A, thr)
    k = int((~alive).sum()); sizes = sorted([len(g[1]) + 1 for g in groups], reverse=True)
    out['%.4f' % thr] = {'dropped': k, 'share': k / F, 'prop_s_thisrun': k / F * RUNTIME,
                         'prop_s_868': k / F * 868.29, 'groups': len(groups),
                         'max_group': sizes[0] if sizes else 0, 'top': sizes[:10]}
    print('Spearman %.4f: 丢 %d (%.2f%%) 比例口径 %.1f s(@1241.91) / %.1f s(@868.29) 组数 %d 最大 %d' %
          (thr, k, 100*k/F, k/F*RUNTIME, k/F*868.29, len(groups), sizes[0] if sizes else 0))
# Pearson 对照
P = np.load(OUT + '/corr.npy'); P[~np.isfinite(P) | (ov < 500)] = np.nan; np.fill_diagonal(P, np.nan)
print()
for thr in (0.9999, 0.999, 0.99, 0.95):
    al, gr = greedy(np.abs(P), thr); k = int((~al).sum())
    print('Pearson  %.4f: 丢 %d (%.2f%%) 比例口径 %.1f s(@1241.91) / %.1f s(@868.29)' %
          (thr, k, 100*k/F, k/F*RUNTIME, k/F*868.29))
out['pearson_ref'] = {'runtime_this_run': RUNTIME, 'F': F, 'N': 5107,
                      'proportional_note': 'k/F*1241.91 (本次实测) 或 k/F*868.29 (此前 50 并行那次)'}
json.dump(out, open(OUT + '/spearman_greedy.json', 'w'), ensure_ascii=False, indent=1)
