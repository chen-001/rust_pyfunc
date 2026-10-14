import json, numpy as np
OUT='/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
meta = json.load(open(OUT+'/meta.json'))
print('meta keys:', list(meta.keys())[:10])
names = meta.get('names') or meta.get('columns')
print('names 数', len(names) if names else None, (names[:3] if names else ''))
an = json.load(open(OUT+'/analysis.json'))
print('analysis keys:', list(an.keys())[:20])
d = json.load(open(OUT+'/drop_by_metric.json'))
print('drop_by_metric 类型', type(d), (list(d)[:5] if isinstance(d, dict) else len(d)))
