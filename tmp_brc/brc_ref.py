"""Independent Python reference implementation of BRC (Binned Rank Correlation).

Written from the mathematical definition in task-3, NOT from the Rust source.

Per day t, given f_i (signal) and r_i (future return), both length n:
  1. R_i  = ascending average rank of r, in [1, n] (ties -> average rank).
  2. order = stable ascending sort of f (ties keep original index order).
  3. W_k = sum_{i<k} R[order[i]];  U_k = W_k - k(k+1)/2;
     D_k = 1 - 2*U_k / (k*(n-k)),  k = 1..n-1.
  4. m = floor(n/2);  S_t = mean_{k=1..m} D_k;  L_t = mean_{k=1..m} D_{n-k}.
  5. aggregate over IC sampling grid days ((local_t+1) % gap == 0):
     BRC_S = mean S_t, BRC_L = mean L_t, BRC = min(BRC_S, BRC_L).
  6. n < 2 or any non-finite value -> day is skipped (nan).
"""

import numpy as np


def avg_rank_ascending(x):
    """Ascending average ranks in [1, n]; ties share the mean of their positions."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="stable")
    sx = x[order]
    new = np.empty(n, dtype=bool)
    new[0] = True
    np.not_equal(sx[1:], sx[:-1], out=new[1:])
    grp = np.cumsum(new) - 1
    counts = np.bincount(grp)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    avg = starts.astype(np.float64) + (counts - 1) * 0.5 + 1.0
    out = np.empty(n, dtype=np.float64)
    out[order] = avg[grp]
    return out


def brc_day_halves(f, r):
    """Return (S_t, L_t) for one day, or (nan, nan) when the day is invalid."""
    f = np.asarray(f, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = f.size
    if n < 2 or f.size != r.size:
        return (np.nan, np.nan)
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        return (np.nan, np.nan)

    R = avg_rank_ascending(r)
    order = np.argsort(f, kind="stable")
    Rk = R[order]

    k = np.arange(1, n, dtype=np.float64)
    W = np.cumsum(Rk)[: n - 1]
    U = W - k * (k + 1.0) / 2.0
    D = 1.0 - 2.0 * U / (k * (n - k))

    m = n // 2
    S = D[:m].mean()
    # D_{n-k} for k = 1..m -> zero-based indices n-2 down to n-m-1
    L = D[n - m - 1 : n - 1].mean()
    return (float(S), float(L))


def brc_aggregate(halves):
    """Mean of valid (S, L) pairs -> (BRC_S, BRC_L, BRC); nan when nothing valid."""
    arr = np.asarray(list(halves), dtype=np.float64)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    arr = arr.reshape(-1, 2)
    good = np.isfinite(arr).all(axis=1)
    if not good.any():
        return (np.nan, np.nan, np.nan)
    s = arr[good, 0].mean()
    l = arr[good, 1].mean()
    return (float(s), float(l), float(min(s, l)))


def brc_series(f_mat, r_mat, grid_mask):
    """f_mat/r_mat: (T, n) per-day cross sections; grid_mask: bool (T,).

    Returns (BRC_S, BRC_L, BRC) over the grid days that are valid.
    """
    halves = [
        brc_day_halves(f_mat[t], r_mat[t])
        for t in range(len(grid_mask))
        if grid_mask[t]
    ]
    return brc_aggregate(halves)


# ---------------------------------------------------------------------------
# O(n^2) pairwise cross-check of the rank-sum identity.  Only for small n.
# A pair (a in first k, b in rest) contributes 1 when R_a > R_b, 0.5 when the
# ranks are equal, 0 otherwise -- this is exactly what the average-rank
# formula W_k - k(k+1)/2 computes.
# ---------------------------------------------------------------------------
def brc_day_halves_pairwise(f, r):
    f = np.asarray(f, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = f.size
    if n < 2 or f.size != r.size:
        return (np.nan, np.nan)
    if not (np.isfinite(f).all() and np.isfinite(r).all()):
        return (np.nan, np.nan)
    R = avg_rank_ascending(r)
    order = np.argsort(f, kind="stable")
    Rk = R[order]
    D = np.empty(n - 1, dtype=np.float64)
    for k in range(1, n):
        left = Rk[:k]
        right = Rk[k:]
        cnt = 0.0
        for a in left:
            cnt += float((a > right).sum()) + 0.5 * float((a == right).sum())
        D[k - 1] = 1.0 - 2.0 * cnt / (k * (n - k))
    m = n // 2
    return (float(D[:m].mean()), float(D[n - m - 1 : n - 1].mean()))
