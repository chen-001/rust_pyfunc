import numpy as np
rng = np.random.default_rng(7)
N, F = 60, 12
X = rng.normal(size=(N, F)).astype(np.float64)
X = X * rng.uniform(0.1, 5, F) + rng.normal(size=F) * 3
nanmask = rng.random((N, F)) < 0.25
X[nanmask] = np.nan

fin = np.isfinite(X)
A = np.where(fin, X, 0.0); M = fin.astype(np.float64)
Sxy = A.T @ A; Sx = A.T @ M; Sxx = (A*A).T @ M; Cn = M.T @ M
n = Cn; sx = Sx; sy = Sx.T; sxx = Sxx; syy = Sxx.T
with np.errstate(invalid='ignore', divide='ignore'):
    cov = Sxy - sx*sy/n; vx = sxx - sx*sx/n; vy = syy - sy*sy/n
    C = cov/np.sqrt(vx*vy)

# brute force pairwise
BF = np.full((F, F), np.nan)
for i in range(F):
    for j in range(F):
        if i == j: continue
        m = fin[:, i] & fin[:, j]
        if m.sum() < 3: continue
        a = X[m, i]; b = X[m, j]
        if a.std() == 0 or b.std() == 0: continue
        BF[i, j] = np.corrcoef(a, b)[0, 1]

err = np.nanmax(np.abs(C - BF))
print('pairwise-complete max err vs brute force =', err)
print('symmetric max err =', np.nanmax(np.abs(C - C.T)))

# scipy reference on a subset
from scipy.stats import pearsonr
print('sample check C[0,1]=%.10f  BF=%.10f' % (C[0,1], BF[0,1]))

# greedy / edges sanity
def greedy(corr, thr, absv=True):
    Fx = corr.shape[0]
    Cx = np.abs(corr) if absv else corr
    alive = np.ones(Fx, bool); groups = []
    for i in range(Fx):
        if not alive[i]: continue
        cand = np.nonzero(alive & (Cx[i] >= thr))[0]
        cand = cand[cand > i]
        if cand.size:
            alive[cand] = False; groups.append((int(i), cand))
    return alive, groups
al, gr = greedy(C, 0.99)
print('greedy dropped', int((~al).sum()), 'groups', len(gr))
print('OK')
