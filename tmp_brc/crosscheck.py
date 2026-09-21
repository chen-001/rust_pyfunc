"""Step-2 crosscheck: my reference BRC vs rp.tail_brc_halves_f32 on real days.

Run:  python crosscheck.py
Writes tmp_brc/crosscheck.txt
"""
import json
import os
import sys

import numpy as np

import rust_pyfunc as rp
import data_prep as D
import brc_ref as B

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "crosscheck.txt")
LINES = []


def say(s=""):
    LINES.append(str(s))
    print(s, flush=True)


def ref_f32(signal, future):
    """Same formulas but accumulating in float32, to separate logic from precision."""
    f = np.asarray(signal, dtype=np.float32)
    r = np.asarray(future, dtype=np.float32)
    n = f.size
    R = B.avg_rank_ascending(r.astype(np.float64)).astype(np.float32)
    order = np.argsort(f, kind="stable")
    Rk = R[order]
    k = np.arange(1, n, dtype=np.float32)
    W = np.cumsum(Rk, dtype=np.float32)[: n - 1]
    U = W - k * (k + np.float32(1.0)) / np.float32(2.0)
    D = np.float32(1.0) - np.float32(2.0) * U / (k * (np.float32(n) - k))
    m = n // 2
    return float(D[:m].mean(dtype=np.float32)), float(D[n - m - 1 : n - 1].mean(dtype=np.float32))


def relerr(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-30)


def main():
    if not hasattr(rp, "tail_brc_halves_f32"):
        say("rp.tail_brc_halves_f32 不存在 -- Rust 侧尚未构建，无法做单日对账。")
        with open(OUT, "w") as fh:
            fh.write("\n".join(LINES) + "\n")
        return 2

    cfg = json.load(open(D.META + "/tail_v4_config.json"))
    allnames = rp.factor_store_v5_info(D.STORE)["factor_names"]
    names = cfg["factor_names"]
    picks = [names[0], names[5000], names[12000], names[18000]]
    dates, _ = D.load_axes()
    ret, ret_sum, restrict = D.load_returns()

    say(f"entry   : rp.tail_brc_halves_f32")
    say(f"factors : {picks}")
    say(f"axis    : meta/dates.npy {dates.size} days, backtest_start={D.BACKTEST_START}, gap={D.GAP}")
    say("")

    worst_f64 = 0.0
    worst_f32 = 0.0
    rows = []
    for nm in picks:
        F = D.load_factor_matrix(allnames.index(nm), dates)
        eff = D.effective_days(F, dates)
        grid = D.grid_days(eff)
        sel = [grid[0], grid[len(grid) // 3], grid[len(grid) // 2],
               grid[(2 * len(grid)) // 3], grid[-1]]
        for local_t, r in sel:
            s, f, h = D.day_cross_section(F, ret, ret_sum, restrict, eff, local_t)
            S64, L64 = B.brc_day_halves(s, f)
            S32, L32 = ref_f32(s, f)
            got = rp.tail_brc_halves_f32(
                np.ascontiguousarray(s, dtype=np.float32),
                np.ascontiguousarray(f, dtype=np.float32),
            )
            gS, gL = float(got[0]), float(got[1])
            eS64, eL64 = relerr(S64, gS), relerr(L64, gL)
            eS32, eL32 = relerr(S32, gS), relerr(L32, gL)
            worst_f64 = max(worst_f64, eS64, eL64)
            worst_f32 = max(worst_f32, eS32, eL32)
            rows.append((nm, local_t, int(dates[eff[local_t]]), s.size, S64, L64, gS, gL,
                         eS64, eL64, S32, L32))
            say(f"{nm}")
            say(f"  local_t={local_t} date={dates[eff[local_t]]} n={s.size} held_row={h}")
            say(f"  ref(f64) S={S64:+.12f} L={L64:+.12f}")
            say(f"  rust     S={gS:+.12f} L={gL:+.12f}")
            say(f"  |dS|={abs(S64-gS):.3e} |dL|={abs(L64-gL):.3e}"
                f"  relS={eS64:.3e} relL={eL64:.3e}")
            say(f"  ref(f32-accum) S={S32:+.12f} L={L32:+.12f}"
                f"  relS={eS32:.3e} relL={eL32:.3e}")
            say("")

    say(f"max relative error vs ref(f64) : {worst_f64:.3e}")
    say(f"max relative error vs ref(f32) : {worst_f32:.3e}")
    say(f"threshold 1e-9 -> {'PASS' if worst_f64 < 1e-9 else 'FAIL'}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("written", OUT)
    return 0 if worst_f64 < 1e-9 else 1


if __name__ == "__main__":
    sys.exit(main())
