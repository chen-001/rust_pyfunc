"""Step-3 end-to-end crosscheck against the Lead's hm104_brc table.

Signal convention (frozen by the Lead, matching the engine): the engine does not
feed BRC the raw store value but slot 0 (`_smooth_1`) of
tail_rank_fill_roll_block_f32 = cross-sectional average rank + missing-rank fill
at (valid_count+1)/2 for stocks with restrict == 0.

Reference values from the Lead's run (raw/rolled channel, gap5, 2016-01-01 ..
2025-12-31, plain style, full backtest):
    factor (with _smooth_1 suffix)                BRC_S                BRC_L
"""
import os
import sys

import numpy as np

import rust_pyfunc as rp
import data_prep as D
import brc_ref as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "e2e_rankfill.txt")

TABLE = {
    "x60y3_buy_cold_f18_lz_complexity": (0.004445460905, 0.004614370163),
    "x60y3_ba_cold_f26_corr_f34": (-0.004212185340, -0.000176642859),
    "x15y10_buy_cold_f20_corr_f25": (0.000415766396, 0.009280319177),
    "x15y10_ba_cold_f15_corr_f28": (0.002480557564, -0.011858734962),
    "hotpool_ext_indpool_x60y3_buy_hot_f34_entropy_1d": (0.006755916972, 0.002143325664),
    "hotpool_ext_indpool_x15y10_buy_hot_f12_curvature": (-0.002268479630, 0.009353028638),
}

LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def main():
    allnames = rp.factor_store_v5_info(D.STORE)["factor_names"]
    dates, _ = D.load_axes()
    ret, ret_sum, restrict = D.load_returns()
    worst_s = worst_l = 0.0
    worst_name = None
    for nm, (ts, tl) in TABLE.items():
        F = D.load_factor_matrix(allnames.index(nm), dates)
        eff = D.effective_days(F, dates)
        grid = D.grid_days(eff)
        halves = []
        for lt, _ in grid:
            s, f, _ = D.day_cross_section_rankfill(F, ret, ret_sum, restrict, eff, lt)
            halves.append(B.brc_day_halves(s, f))
        bs, bl, b = B.brc_aggregate(halves)
        ds, dl = abs(bs - ts), abs(bl - tl)
        if max(ds, dl) > max(worst_s, worst_l):
            worst_name = nm
        worst_s, worst_l = max(worst_s, ds), max(worst_l, dl)
        say(f"{nm}")
        say(f"  days={len(grid)}  ref BRC_S={bs:+.12f} BRC_L={bl:+.12f} BRC={b:+.12f}")
        say(f"                table BRC_S={ts:+.12f} BRC_L={tl:+.12f}")
        say(f"                |dS|={ds:.3e} |dL|={dl:.3e}")
    say("")
    say(f"max |dBRC_S| = {worst_s:.3e}")
    say(f"max |dBRC_L| = {worst_l:.3e}")
    say(f"worst factor: {worst_name}")
    say("PASS (1e-12)" if max(worst_s, worst_l) < 1e-12 else "FAIL (1e-12)")
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("written", OUT)
    return 0 if max(worst_s, worst_l) < 1e-12 else 1


if __name__ == "__main__":
    sys.exit(main())
