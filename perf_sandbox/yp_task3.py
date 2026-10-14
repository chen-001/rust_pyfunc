import json, numpy as np, collections, re
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta=json.load(open(OUT+'/meta.json')); names=meta['names']
alive=np.load(OUT+'/alive_0.9900_abs.npy')
print('alive', alive.shape, alive.dtype, '保留', int(alive.sum()), '删除', int((~alive).sum()))
prefices = sorted(set('_'.join(n.split('_')[:2]) for n in names))
print('矩阵前缀数', len(prefices))
by_pref = {}
for i, n in enumerate(names):
    p = '_'.join(n.split('_')[:2]); m = '_'.join(n.split('_')[2:])
    by_pref.setdefault(p, {})[m] = (i, bool(alive[i]))
common = set.intersection(*[set(v) for v in by_pref.values()])
print('15 档都有的指标名', len(common))
all_dropped = []
for m in sorted(common):
    if all(not by_pref[p][m][1] for p in prefices):
        all_dropped.append(m)
print('在全部 15 档都被删的指标名:', len(all_dropped))
fam = collections.Counter()
for m in all_dropped:
    fam[m.split('_')[0]] += 1
print('按族统计:', dict(fam.most_common()))
print()
print('--- 名单（前 45）---')
for m in all_dropped[:45]: print('   ', m)
json.dump(all_dropped, open(OUT+'/all15_dropped_099.json','w'), ensure_ascii=False, indent=1)
