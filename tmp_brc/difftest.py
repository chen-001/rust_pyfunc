"""Differential test of rp.tail_brc_halves_f32 vs brc_ref, focused on ties.

Ties are the only place a rank convention can silently diverge: average ranks
(spec) vs ordinal ranks.  Cases: heavy ties in r, heavy ties in f, both, plus
the two mirror identities.  Values are float32-representable so both sides see
bit-identical inputs.

Run:  python difftest.py   -> writes tmp_brc/difftest.txt
"""
import os
import sys

import numpy as np

import rust_pyfunc as rp
import brc_ref as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "difftest.txt")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def rust(s, f):
    got = rp.tail_brc_halves_f32([float(x) for x in s], [float(x) for x in f])
    return float(got[0]), float(got[1])


def main():
    if not hasattr(rp, "tail_brc_halves_f32"):
        say("rp.tail_brc_halves_f32 missing")
        return 2
    rng = np.random.RandomState(20260918)
    worst = 0.0
    worst_case = None
    ncase = 0
    for n in (2, 3, 5, 17, 100, 501, 2000):
        for kind in ("distinct", "r_ties", "f_ties", "both_ties", "constant_f", "all_tied_r"):
            for trial in range(4):
                if kind == "distinct":
                    f = rng.standard_normal(n).astype(np.float32)
                    r = rng.standard_normal(n).astype(np.float32)
                elif kind == "r_ties":
                    f = rng.standard_normal(n).astype(np.float32)
                    r = rng.randint(0, max(2, n // 20), n).astype(np.float32)
                elif kind == "f_ties":
                    f = rng.randint(0, max(2, n // 20), n).astype(np.float32)
                    r = rng.standard_normal(n).astype(np.float32)
                elif kind == "both_ties":
                    f = rng.randint(0, 4, n).astype(np.float32)
                    r = rng.randint(0, 4, n).astype(np.float32)
                elif kind == "constant_f":
                    f = np.full(n, 1.5, dtype=np.float32)
                    r = rng.standard_normal(n).astype(np.float32)
                else:
                    f = rng.standard_normal(n).astype(np.float32)
                    r = np.full(n, -0.25, dtype=np.float32)
                mine = B.brc_day_halves(f.astype(np.float64), r.astype(np.float64))
                got = rust(f, r)
                ncase += 1
                d = max(abs(mine[0] - got[0]), abs(mine[1] - got[1]))
                if d > worst:
                    worst = d
                    worst_case = (n, kind, trial, mine, got)
                # mirrors
                m1 = B.brc_day_halves(f.astype(np.float64), (-r).astype(np.float64))
                g1 = rust(f, (-r).astype(np.float32))
                assert abs(m1[0] + got[0]) < 1e-12 and abs(m1[1] + got[1]) < 1e-12, (kind, n)
                assert abs(g1[0] + got[0]) < 1e-12 and abs(g1[1] + got[1]) < 1e-12, (kind, n)
    say(f"cases: {ncase}")
    say(f"max |rust - ref| over all cases: {worst:.3e}")
    say(f"worst case: {worst_case}")
    say("mirror identities (r -> -r gives (-S,-L)) hold on every case for both sides")
    say("PASS" if worst < 1e-12 else "FAIL")
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
