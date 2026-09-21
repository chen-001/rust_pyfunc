"""Step-3 end-to-end crosscheck: my reference BRC_S/BRC_L over the whole run vs the
hm104_brc table produced by the Lead.

Stage A (now): compute BRC_S / BRC_L / BRC for a few factors over every gap5 IC grid
day of the 2016-01-01..2025-12-31 axis, cache to tmp_brc/e2e_ref.npz.
Stage B (after the Lead notifies): read the table, match factors, compare.

Run:  python e2e.py            # compute + compare if the table exists
      python e2e.py --compute  # compute only
"""
import json
import os
import sys

import numpy as np

import rust_pyfunc as rp
import data_prep as D
import brc_ref as B

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "e2e_ref.npz")
OUT = os.path.join(HERE, "e2e_crosscheck.txt")
TABLE_DIR = "/nas197/user_home_unsafe/chenzongwei/hm104_brc_tail_v4"


def pick_factors(n=6):
    names = json.load(open(D.META + "/tail_v4_config.json"))["factor_names"]
    step = len(names) // (n + 1)
    return [names[step * (i + 1)] for i in range(n)]


def compute():
    allnames = rp.factor_store_v5_info(D.STORE)["factor_names"]
    dates, _ = D.load_axes()
    ret, ret_sum, restrict = D.load_returns()
    picks = pick_factors()
    # grid A: the rule in task-3, (local_t+1) % gap == 0  -> 485 days
    # grid B: the date list this run actually recorded in ic_ts/ic_neu_gap5_dates.npy
    gridB = np.load("/nas197/user_home_unsafe/chenzongwei/hm104_ssm3_tail_v4/ic_ts/"
                    "ic_neu_gap5_dates.npy")
    res = {}
    for nm in picks:
        F = D.load_factor_matrix(allnames.index(nm), dates)
        eff = D.effective_days(F, dates)
        gridA = D.grid_days(eff)
        halvesA = [
            B.brc_day_halves(*D.day_cross_section(F, ret, ret_sum, restrict, eff, t)[:2])
            for t, _ in gridA
        ]
        bs, bl, b = B.brc_aggregate(halvesA)
        # grid B: map recorded dates back to local_t
        eff_dates = dates[np.asarray(eff)]
        loc = np.searchsorted(eff_dates, gridB)
        ok = (loc < eff_dates.size) & (eff_dates[np.clip(loc, 0, eff_dates.size - 1)] == gridB)
        halvesB = [
            B.brc_day_halves(*D.day_cross_section(F, ret, ret_sum, restrict, eff, int(t))[:2])
            for t in loc[ok]
        ]
        bsB, blB, bB = B.brc_aggregate(halvesB)
        res[nm] = (bs, bl, b, len(gridA), bsB, blB, bB, int(ok.sum()))
        print(f"{nm}", flush=True)
        print(f"  gridA(spec 485): BRC_S={bs:+.12f} BRC_L={bl:+.12f} BRC={b:+.12f} days={len(gridA)}")
        print(f"  gridB(run {int(ok.sum())}): BRC_S={bsB:+.12f} BRC_L={blB:+.12f} BRC={bB:+.12f}")
    np.savez(CACHE, names=np.array(list(res.keys())),
             vals=np.array([res[k] for k in res]))
    print("cached", CACHE)


def load_table():
    """Read the Lead's table; returns (path, dataframe, {lowercase col -> col})."""
    import glob
    cands = sorted(glob.glob(TABLE_DIR + "/**/*.parquet", recursive=True))
    cands += sorted(glob.glob(TABLE_DIR + "/**/*.csv", recursive=True))
    for p in cands:
        try:
            if p.endswith(".parquet"):
                import pyarrow.parquet as pq
                t = pq.read_table(p).to_pandas()
            else:
                t = __import__("pandas").read_csv(p)
        except Exception:
            continue
        cols = {c.lower(): c for c in t.columns}
        if "brc_s" in cols and "brc_l" in cols:
            return p, t, cols
    return None, None, None


def compare():
    z = np.load(CACHE)
    names = [str(x) for x in z["names"]]
    vals = z["vals"]
    path, t, cols = load_table()
    lines = []
    if t is None:
        lines.append(f"table not found under {TABLE_DIR} (files: "
                     f"{os.listdir(TABLE_DIR) if os.path.isdir(TABLE_DIR) else 'dir missing'})")
    else:
        lines.append(f"table: {path}")
        lines.append(f"columns: {list(t.columns)}")
        lines.append(f"rows: {len(t)}")
        namecol = cols.get("factor_name") or cols.get("factor") or cols.get("name") or t.columns[0]
        src = cols.get("source_factor")
        key = t[namecol].astype(str).values
        srcv = t[src].astype(str).values if src else None
        # gridA ref = vals[:, 0:3], gridB ref = vals[:, 4:7]
        worst = {"A": 0.0, "B": 0.0}
        for nm, v in zip(names, vals):
            hit = np.flatnonzero(key == nm)
            if hit.size == 0:
                hit = np.flatnonzero([k.startswith(nm) for k in key])
            if hit.size == 0 and srcv is not None:
                hit = np.flatnonzero(srcv == nm)
            if hit.size == 0:
                lines.append(f"{nm}: NOT FOUND in table")
                continue
            row = t.iloc[hit[0]]
            lines.append(f"{nm} -> table row {row[namecol]}")
            for lbl, ia, ib in (("BRC", 0, 6), ("BRC_S", 1, 4), ("BRC_L", 2, 5)):
                c = cols.get(lbl.lower())
                if c is None:
                    continue
                got = float(row[c])
                dA = abs(got - v[ia])
                dB = abs(got - v[ib])
                worst["A"] = max(worst["A"], dA)
                worst["B"] = max(worst["B"], dB)
                lines.append(f"    {lbl}: table={got:+.12f}")
                lines.append(f"      gridA(spec, {int(v[3])}d) ref={v[ia]:+.12f} |d|={dA:.3e}")
                lines.append(f"      gridB(run,  {int(v[7])}d) ref={v[ib]:+.12f} |d|={dB:.3e}")
        lines.append(f"max abs deviation gridA = {worst['A']:.3e}")
        lines.append(f"max abs deviation gridB = {worst['B']:.3e}")
        lines.append(f"verdict: {'gridA' if worst['A'] <= worst['B'] else 'gridB'}"
                     f" (A={worst['A']:.3e}, B={worst['B']:.3e})")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    if not os.path.exists(CACHE) or "--compute" in sys.argv:
        compute()
    if "--compute" not in sys.argv:
        compare()
