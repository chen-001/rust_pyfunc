import json, numpy as np, collections
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
cats=json.load(open(OUT+'/degen_cats.json'))
zero=[n for n in cats['allzero']]
print('全 0 列数', len(zero))
pref=collections.Counter('_'.join(n.split('_')[:2]) for n in zero)
print('涉及的矩阵前缀:', len(pref), sorted(pref.keys()))
met=collections.Counter('_'.join(n.split('_')[2:]) for n in zero)
print('指标名数', len(met))
print('带 degen 的:', [m for m in met if 'degen' in m])
nondeg={m:c for m,c in met.items() if 'degen' not in m}
print('其余指标名', len(nondeg), '列数', sum(nondeg.values()))
json.dump({'cols': zero, 'prefixes': sorted(pref.keys()), 'metrics': list(met)}, open(OUT+'/zero_cols.json','w'), ensure_ascii=False)
