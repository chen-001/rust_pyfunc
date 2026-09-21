"""Aggregate-level crosscheck that does not need the Lead's table.

For each factor: run every one of its 485 gap5 grid days through
rp.tail_brc_halves_f32, average them here (mean of finite days, BRC = min), and
compare with the pure-Python reference aggregate over the same days.  This
isolates the aggregation rule (grid + skip + mean + min) from the engine's own
plumbing, which only the Lead's table can test.

Run:  python e2e_rustdays.py   -> writes tmp_brc/e2e_rustdays.txt
"""
import os
import sys

import numpy as np

import rust_pyfunc as rp
import data_prep as D
import brc_ref as B
from e2e import pick_factors

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e2e_rustdays.txt")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def main():
    if not hasattr(rp, "tail_brc_halves_f32"):
        say("rp.tail_brc_halves_f32 missing")
        return 2
    allnames = rp.factor_store_v5_info(D.STORE)["factor_names"]
    dates, _ = D.load_axes()
    ret, ret_sum, restrict = D.load_returns()
    worst = 0.0
    for nm in pick_factors():
        F = D.load_factor_matrix(allnames.index(nm), dates)
        eff = D.effective_days(F, dates)
        grid = D.grid_days(eff)
        ref_halves, rust_halves = [], []
        for lt, _ in grid:
            s, f, _ = D.day_cross_section(F, ret, ret_sum, restrict, eff, lt)
            ref_halves.append(B.brc_day_halves(s, f))
            got = rp.tail_brc_halves_f32([float(x) for x in s], [float(x) for x in f])
            rust_halves.append((float(got[0]), float(got[1])))
        rs, rl, rb = B.brc_aggregate(ref_halves)
        us, ul, ub = B.brc_aggregate(rust_halves)
        d = max(abs(rs - us), abs(rl - ul), abs(rb - ub))
        worst = max(worst, d)
        say(f"{nm}")
        say(f"  days={len(grid)}  ref   BRC_S={rs:+.12f} BRC_L={rl:+.12f} BRC={rb:+.12f}")
        say(f"                rust  BRC_S={us:+.12f} BRC_L={ul:+.12f} BRC={ub:+.12f}")
        say(f"                max|diff|={d:.3e}")
    say("")
    say(f"max |aggregate diff| over all factors: {worst:.3e}")
    say("PASS" if worst < 1e-12 else "FAIL")
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("written", OUT)
    return 0 if worst < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
