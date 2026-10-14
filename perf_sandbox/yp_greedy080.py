import json, numpy as np, collections, time
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta=json.load(open(OUT+'/meta.json')); names=meta['names']; n=len(names)
pref=['_'.join(x.split('_')[:2]) for x in names]
print('列数', n, '前缀数', len(set(pref)), flush=True)
C=np.load(OUT+'/corr.npy', mmap_mode='r')
def greedy(thr):
    kept=[]; ka=None
    t0=time.time()
    for i in range(n):
        if not kept:
            kept.append(i); ka=np.abs(np.asarray(C[i,kept])); continue
        v=np.abs(np.asarray(C[i,kept]))
        v=np.where(np.isfinite(v), v, 0.0)
        if v.max(initial=0.0) < thr:
            kept.append(i)
        if i % 2000 == 0: print('  thr=%.2f i=%d kept=%d %.0fs' % (thr,i,len(kept),time.time()-t0), flush=True)
    return kept
for thr in (0.80,):
    kept=greedy(thr)
    dropped=[i for i in range(n) if i not in set(kept)]
    print('阈值 %.2f：保留 %d 列，删 %d 列（%.1f%%）' % (thr, len(kept), len(dropped), 100.0*len(dropped)/n), flush=True)
    print('删除的列按矩阵前缀分布：', dict(collections.Counter(pref[i] for i in dropped).most_common()), flush=True)
    print('各矩阵被删比例：', {p: round(100.0*collections.Counter(pref[i] for i in dropped)[p]/sum(1 for x in pref if x==p),1) for p in sorted(set(pref))}, flush=True)
    json.dump({'thr':thr,'kept':len(kept),'dropped':[names[i] for i in dropped]}, open(OUT+'/greedy_080.json','w'), ensure_ascii=False)
