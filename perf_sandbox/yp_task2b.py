import json, numpy as np, collections
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta=json.load(open(OUT+'/meta.json')); names=meta['names']
V=np.load(OUT+'/vals.npy', mmap_mode='r')
n, f = V.shape
print('形状', V.shape)
fin = np.isfinite(V)
nfin = fin.sum(axis=0)
nz = ((V != 0) & fin).sum(axis=0)
uniq = np.array([np.unique(np.asarray(V[:, i])[fin[:, i]]).size if nfin[i] else 0 for i in range(f)])
degen = (nfin > 0) & (uniq == 1)
print('完全退化的列（非 NaN 值只有一种）:', int(degen.sum()))
cats = collections.Counter()
idx_allnan = np.where(nfin == 0)[0]
idx_allzero = np.where(degen & (nz == 0))[0]
idx_const = np.where(degen & (nz > 0))[0]
print('  全 NaN 列:', len(idx_allnan))
print('  全 0 列:', len(idx_allzero))
print('  常数但非零列:', len(idx_const))
def metric(n):
    p = n.split('_'); return '_'.join(p[2:])
c_allnan = collections.Counter(metric(names[i]) for i in idx_allnan)
c_allzero = collections.Counter(metric(names[i]) for i in idx_allzero)
c_const = collections.Counter(metric(names[i]) for i in idx_const)
print()
print('--- 全 NaN 列 的指标名（前 12）---'); [print('   %-40s %d 档' % (k, v)) for k, v in c_allnan.most_common(12)]
print('--- 全 0 列 的指标名（前 12）---'); [print('   %-40s %d 档' % (k, v)) for k, v in c_allzero.most_common(12)]
print('--- 常数非零列 的指标名（前 12）---'); [print('   %-40s %d 档' % (k, v)) for k, v in c_const.most_common(12)]
json.dump({'allnan': [names[i] for i in idx_allnan], 'allzero': [names[i] for i in idx_allzero], 'const': [names[i] for i in idx_const]},
          open(OUT+'/degen_cats.json','w'), ensure_ascii=False)
