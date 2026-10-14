import os, json, time
os.environ['RAYON_NUM_THREADS'] = '10'
import numpy as np
import rust_pyfunc as rp
DATE = int(os.environ.get('YP_DATE', '20241129'))
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
zero = json.load(open(OUT + '/zero_cols.json'))['cols']
print('日期', DATE, '待查列数', len(zero), flush=True)
t0 = time.time()
codes, names, vals = rp.py_yupei_pair(DATE)
V = np.asarray(vals, dtype=np.float32).reshape(len(codes), len(names))
print('跑完 %.1f 秒  形状 %s' % (time.time() - t0, (V.shape,)), flush=True)
nidx = {n: i for i, n in enumerate(names)}
idx = [nidx[c] for c in zero if c in nidx]
sub = V[:, idx]
np.save(OUT + '/zero_check_%d.npy' % DATE, sub)
json.dump({'date': DATE, 'cols': [c for c in zero if c in nidx], 'missing': [c for c in zero if c not in nidx]},
          open(OUT + '/zero_check_%d.json' % DATE, 'w'), ensure_ascii=False)
print('缺列', len([c for c in zero if c not in nidx]), flush=True)
