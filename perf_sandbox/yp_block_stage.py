import os, json, time
os.environ.setdefault('OMP_NUM_THREADS', '25')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '25')
import numpy as np
T0 = time.time()
OUT = '/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_out'
def log(*a): print('[%7.1fs]' % (time.time() - T0), *a, flush=True)

meta = json.load(open(OUT + '/meta.json'))
names = meta['names']; RUNTIME = meta['seconds']
X = np.load(OUT + '/vals.npy')
N, F = X.shape
PREF = ['cnt_t005','cnt_t01','cnt_t02','cnt_t05','cnt_t1','cnt_t3','cnt_t5','cnt_t10','cnt_t30',
        'vol_t02','vol_t05','vol_t1','vol_t3','vol_t5','vol_t30']
pre_of = {}; met_of = {}
for i, nm in enumerate(names):
    for p in PREF:
        if nm.startswith(p + '_'):
            pre_of[i] = p; met_of[i] = nm[len(p) + 1:]; break
assert len(pre_of) == F
pre_arr = np.array([pre_of[i] for i in range(F)])
met_arr = np.array([met_of[i] for i in range(F)])
log('分块 OK')

fin = np.isfinite(X)
A = np.where(fin, X, 0.0).astype(np.float64)
del X
M = fin.astype(np.float64); del fin
log('A/M 就绪')

def pcorr(Ab, Mb, want_n=False):
    AT = np.ascontiguousarray(Ab.T); DT = np.ascontiguousarray((Ab * Ab).T)
    MT = np.ascontiguousarray(Mb.T)
    Sxy = AT @ Ab; Sx = AT @ MT; Sxx = DT @ MT; Cn = MT @ MT
    SxT = Sx.T; SxxT = Sxx.T
    with np.errstate(invalid='ignore', divide='ignore'):
        cov = Sxy - Sx * SxT / Cn
        vx = Sxx - Sx * Sx / Cn
        vy = SxxT - SxT * SxT / Cn
        C = (cov / np.sqrt(vx * vy)).astype(np.float32)
    return (C, Cn) if want_n else C

# ---- 1. 块内 701x701 ----
BLK = np.full((15, 701, 701), np.nan, np.float32)
BLKN = np.zeros((15, 701, 701), np.int16)
for bi, p in enumerate(PREF):
    idx = np.where(pre_arr == p)[0]
    C, Cn = pcorr(A[:, idx], M[:, idx], True)
    np.fill_diagonal(C, np.nan)
    BLK[bi] = C; BLKN[bi] = np.rint(Cn).astype(np.int16)
    log('块内 %s 完成 (n 中位 %d)' % (p, int(np.median(Cn))))
np.save(OUT + '/block_corr.npy', BLK); np.save(OUT + '/block_n.npy', BLKN)

# ---- 2. tau 轴 701 x 15 x 15 ----
metrics = sorted(set(met_arr.tolist()))
log('指标名 %d 个' % len(metrics))
pos = {m: {} for m in metrics}
for i in range(F):
    pos[met_arr[i]][pre_arr[i]] = i
TAU = np.full((len(metrics), 15, 15), np.nan, np.float32)
TAUN = np.zeros((len(metrics), 15, 15), np.int16)
for mi, m in enumerate(metrics):
    idx = np.array([pos[m][p] for p in PREF])
    C, Cn = pcorr(A[:, idx], M[:, idx], True)
    np.fill_diagonal(C, np.nan)
    TAU[mi] = C; TAUN[mi] = np.rint(Cn).astype(np.int16)
np.save(OUT + '/tau_corr.npy', TAU); np.save(OUT + '/tau_n.npy', TAUN)
json.dump({'metrics': metrics, 'prefixes': PREF}, open(OUT + '/tau_meta.json', 'w'))
log('tau 轴完成')
log('DONE')
