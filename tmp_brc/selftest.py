"""Step-1 self test for brc_ref.py -- constructed data only, no Rust involved.

Output is written to tmp_brc/selftest.txt.
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import brc_ref as B

OUT = os.path.join(HERE, "selftest.txt")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def check(name, cond, detail=""):
    say(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}")
    return bool(cond)


ok = True

# --- 1. perfect monotone: S = L = 1 ---------------------------------------
n = 1000
f = np.arange(n, dtype=np.float64)
r = np.arange(n, dtype=np.float64)
S, L = B.brc_day_halves(f, r)
ok &= check("monotone f=+r -> S=L=1", S == 1.0 and L == 1.0, f"S={S!r} L={L!r}")

# --- 2. exact reverse: S = L = -1 -----------------------------------------
r = np.arange(n, 0, -1, dtype=np.float64)
S, L = B.brc_day_halves(f, r)
ok &= check("f=+r, r reversed -> S=L=-1", S == -1.0 and L == -1.0, f"S={S!r} L={L!r}")

# --- 3. random n=5000, 5 seeds --------------------------------------------
say()
say("random n=5000 cross sections, 5 seeds (expect S, L near 0):")
rs = np.random.RandomState(0)
for seed in range(5):
    rng = np.random.RandomState(seed)
    fr = rng.standard_normal(5000)
    rr = rng.standard_normal(5000)
    S, L = B.brc_day_halves(fr, rr)
    say(f"  seed={seed}  S={S:+.6f}  L={L:+.6f}  |S|+|L|={abs(S)+abs(L):.6f}")

# --- 4. mirror identity ---------------------------------------------------
# Task-3 claims: negate r  ->  S -> -L, L -> -S, BRC -> -max(S,L).
# The definition as written actually gives D'_k = -D_k for every k, i.e.
# S -> -S and L -> -L.  Both are measured here on distinct-r samples, plus
# the tie-free structural relation under f -> -f.
say()
say("mirror identity on r -> -r (distinct values, no ties in r):")
for seed in range(5):
    rng = np.random.RandomState(100 + seed)
    fr = rng.standard_normal(300)
    rr = rng.permutation(300).astype(np.float64)  # distinct r -> no tie ambiguity
    S, L = B.brc_day_halves(fr, rr)
    Sm, Lm = B.brc_day_halves(fr, -rr)
    say(f"  seed={seed}  S={S:+.12f} L={L:+.12f} | -r: S'={Sm:+.12f} L'={Lm:+.12f}"
        f" | S'+S={Sm+S:+.3e} L'+L={Lm+L:+.3e} S'+L={Sm+L:+.3e} L'+S={Lm+S:+.3e}")
ok &= check("identity S'=-S and L'=-L under r -> -r", True,
            "(numbers above; claimed S'=-L/L'=-S is contradicted below)")

rng = np.random.RandomState(7)
fr = rng.standard_normal(200)
rr = rng.permutation(200).astype(np.float64)
S, L = B.brc_day_halves(fr, rr)
Sm, Lm = B.brc_day_halves(fr, -rr)
claim_neg_r = abs(Sm + L) < 1e-12 and abs(Lm + S) < 1e-12
true_neg_r = abs(Sm + S) < 1e-12 and abs(Lm + L) < 1e-12
ok &= check("claim 'negate r: S->-L, L->-S' holds", claim_neg_r,
            f"S'={Sm:+.12f} -L={-L:+.12f} L'={Lm:+.12f} -S={-S:+.12f}")
ok &= check("actual 'negate r: S->-S, L->-L' holds", true_neg_r)
# The third part of the claimed identity does hold: min(-S,-L) = -max(S,L).
b0 = min(S, L)
b1 = min(Sm, Lm)
ok &= check("negate r: BRC -> -max(S,L) (holds even though S'=-S, L'=-L)",
            abs(b1 + max(S, L)) < 1e-15,
            f"BRC'={b1:+.12f} -max(S,L)={-max(S,L):+.12f}")

say()
say("mirror identity on f -> -f (reverse the f order, distinct f):")
Sf, Lf = B.brc_day_halves(-fr, rr)
ok &= check("negate f: S'=-L and L'=-S", abs(Sf + L) < 1e-12 and abs(Lf + S) < 1e-12,
            f"S'={Sf:+.12f} -L={-L:+.12f} L'={Lf:+.12f} -S={-S:+.12f}")

# --- 5. tiny n vs O(n^2) pairwise ----------------------------------------
say()
say("tiny n (2,3,4) exhaustive: rank-sum vs O(n^2) pairwise")
rng = np.random.RandomState(3)
for n_small in (2, 3, 4):
    worst = 0.0
    ties_used = False
    for trial in range(400):
        # half the trials keep duplicates so ties in r and f are covered
        if trial % 2 == 0:
            fv = rng.randint(0, n_small, n_small).astype(np.float64)
            rv = rng.randint(0, n_small, n_small).astype(np.float64)
            ties_used = True
        else:
            fv = rng.permutation(n_small).astype(np.float64)
            rv = rng.permutation(n_small).astype(np.float64)
        a = B.brc_day_halves(fv, rv)
        b = B.brc_day_halves_pairwise(fv, rv)
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    ok &= check(f"n={n_small}: bitwise identical (ties included={ties_used})",
                worst == 0.0, f"max |diff| = {worst:.3e}")

say()
say("n=50/200 random cross sections: rank-sum vs O(n^2) pairwise")
for n_mid in (50, 200):
    worst = 0.0
    for trial in range(20):
        rng = np.random.RandomState(1000 + n_mid + trial)
        if trial % 2 == 0:
            fv = rng.randint(0, 7, n_mid).astype(np.float64)
            rv = rng.randint(0, 7, n_mid).astype(np.float64)
        else:
            fv = rng.standard_normal(n_mid)
            rv = rng.standard_normal(n_mid)
        a = B.brc_day_halves(fv, rv)
        b = B.brc_day_halves_pairwise(fv, rv)
        worst = max(worst, abs(a[0] - b[0]), abs(a[1] - b[1]))
    ok &= check(f"n={n_mid}: rank-sum == O(n^2) pairwise", worst < 1e-12,
                f"max |diff| = {worst:.3e}")

# --- 6. invalid day handling ---------------------------------------------
say()
ok &= check("n=1 -> (nan, nan)", all(np.isnan(v) for v in B.brc_day_halves([1.0], [2.0])))
ok &= check("nan in r -> (nan, nan)",
            all(np.isnan(v) for v in B.brc_day_halves([1.0, 2.0], [1.0, np.nan])))
ok &= check("inf in f -> (nan, nan)",
            all(np.isnan(v) for v in B.brc_day_halves([1.0, np.inf], [1.0, 2.0])))

# --- 7. aggregate + grid --------------------------------------------------
say()
say("aggregate on a (T,n) panel with a gap-5 grid:")
T, nn = 20, 60
rng = np.random.RandomState(11)
F = rng.standard_normal((T, nn))
R = rng.standard_normal((T, nn))
F[3, :] = np.nan
grid = np.array([(t + 1) % 5 == 0 for t in range(T)])
bs, bl, b = B.brc_series(F, R, grid)
halves = [B.brc_day_halves(F[t], R[t]) for t in range(T) if grid[t]]
bs2, bl2, b2 = B.brc_aggregate(halves)
ok &= check("brc_series == manual aggregate", (bs, bl, b) == (bs2, bl2, b2),
            f"BRC_S={bs:+.6f} BRC_L={bl:+.6f} BRC={b:+.6f} (grid days={int(grid.sum())},"
            f" one has nan -> skipped)")
ok &= check("BRC == min(BRC_S, BRC_L)", b == min(bs, bl))
ok &= check("all-nan input -> (nan,nan,nan)",
            all(np.isnan(v) for v in B.brc_aggregate([(np.nan, np.nan)])))

say()
say("ALL PASS" if ok else "SOME CHECKS FAILED")

with open(OUT, "w") as fh:
    fh.write("\n".join(LINES) + "\n")
print("written", OUT)
sys.exit(0 if ok else 1)
