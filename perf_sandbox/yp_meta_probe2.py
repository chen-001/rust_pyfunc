import json, numpy as np
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
d=json.load(open(OUT+'/drop_by_metric.json'))
for k,v in d.items():
    print(k, type(v).__name__, (len(v) if hasattr(v,'__len__') else v))
dbm = d['drop_by_metric']
if isinstance(dbm, dict):
    ks=list(dbm)[:5]
    for k in ks: print('  ', k, '→', json.dumps(dbm[k], ensure_ascii=False)[:160])
an=json.load(open(OUT+'/analysis.json'))
g=an['greedy']
print('greedy keys', list(g.keys()) if isinstance(g,dict) else type(g))
for k in (list(g)[:4] if isinstance(g,dict) else []): print('  ', k, json.dumps(g[k], ensure_ascii=False)[:200])
