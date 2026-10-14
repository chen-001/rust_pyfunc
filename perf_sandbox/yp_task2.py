import json, re, numpy as np
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta=json.load(open(OUT+'/meta.json')); names=meta['names']
d=json.load(open(OUT+'/drop_by_metric.json'))
degen_all=d['degen_all']
print('degen_all 类型', type(degen_all), '长度', len(degen_all))
print('样例', json.dumps(degen_all[:3] if isinstance(degen_all,list) else list(degen_all.items())[:3], ensure_ascii=False)[:300])
prefices = sorted(set(n.split('_')[0]+'_'+n.split('_')[1] for n in names))
print('矩阵前缀', len(prefices), prefices[:3], prefices[-3:])
V=None
print()
print('--- 任务2：退化列到底是「常数」还是「全 0 / 全 NaN」 ---')
degen_names = degen_all if isinstance(degen_all, list) else list(degen_all)
V = np.load(OUT+'/vals.npy', mmap_mode='r')
idx = [i for i,n in enumerate(names) if any(n.endswith(m) for m in degen_names)]
print('退化指示名', len(degen_names), '匹配列数', len(idx))
rep = {}
for i in idx:
    col = np.asarray(V[:, i], dtype=np.float64)
    fin = np.isfinite(col)
    n_nan = int((~fin).sum()); nz = int((col[fin] != 0).sum())
    uniq = int(np.unique(col[fin]).size) if fin.any() else 0
    key = (n_nan, nz, uniq)
    rep.setdefault(key, []).append(names[i])
print('分类（全 NaN 数, 非零数, 唯一值数）→ 列数')
for k, v in sorted(rep.items(), key=lambda x: -len(x[1])):
    print('  ', k, '→', len(v), '列  例：', v[0])
np.save(OUT+'/degen_col_idx.npy', np.array(idx))
json.dump({'degen_cols': [names[i] for i in idx]}, open(OUT+'/degen_cols.json','w'))
