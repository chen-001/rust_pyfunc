"""临时探索脚本 2：把 hm41 旧 pandas 结果与我打算写的 Rust 算法（纯 Python 复刻）对齐。

不复刻的东西：
  - 读文件：这里用 pandas 直接读 CSV，和 Rust 侧的 f64 自解析等价
  - 排序稳定性：Python 的 sorted 是稳定的，和 Rust sort_by_key 一致
这个脚本只用来在编译之前把语义错误（尤其是 price 对齐、错位截断）抓出来，用完删。
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/home/chenzongwei/pythoncode")
import design_whatever_out as dwo

MKT = "/ssd_data/stock/{d}/market_data/{c}_{d}_market_data.csv"

SESSIONS = [
    ("allday", "09:30", "14:57"),
    ("open30", "09:30", "10:00"),
    ("morning", "09:30", "11:30"),
    ("afternoon", "13:00", "14:57"),
    ("close30", "14:30", "14:57"),
]
CUTS = [(6, "num5"), (11, "num10")]

BLOCK = [
    "asks_dura_mean", "bids_dura_mean", "asks_dura_std", "bids_dura_std",
    "asks_dura_max", "bids_dura_max", "asks_dura_skew", "bids_dura_skew",
    "asks_dura_corr", "bids_dura_corr", "asks_first_sum", "asks_last_sum",
    "bids_first_sum", "bids_last_sum", "ab_first", "ab_last",
    "asks_dura_sum_abs", "bids_dura_sum_abs", "ab_first_abs", "ab_last_abs",
    "asks_bids_dura_sum_plus", "asks_bids_dura_sum_minus",
    "asks_dura_sum_rate", "bids_dura_sum_rate", "ab_first_rate", "ab_last_rate",
    "corr_asks_first", "corr_asks_last", "corr_bids_first", "corr_bids_last",
    "corr_asks_first_last", "corr_bids_first_last",
    "corr_first_asks_bids", "corr_last_asks_bids",
]


def read_book(code, date, side):
    df = pd.read_csv(MKT.format(d=date, c=code))
    df = df[df.last_prc != 0]
    prcs = [f"{side}_prc{i}" for i in range(1, 11)]
    vols = [f"{side}_vol{i}" for i in range(1, 11)]
    rows = []
    t = pd.to_timedelta(df.exchtime, unit="us") + pd.Timestamp("1970-01-01 08:00:00")
    for k in range(10):
        p = df[prcs[k]].to_numpy()
        v = df[vols[k]].to_numpy()
        for i in range(len(df)):
            if p[i] == 0.0:
                continue
            rows.append((t.iloc[i].value, k + 1, p[i], v[i]))
    rows.sort(key=lambda r: (r[0], r[1]))
    return rows


def dedup(rows, keep_first):
    seen = {}
    out = []
    it = rows if keep_first else list(reversed(rows))
    for r in it:
        if r[2] not in seen:
            seen[r[2]] = 1
            out.append(r)
    return out


def cc(a, b):
    """DataFrame.corr 的 Pearson；样本不足或方差为 0 返回 NaN。"""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def side(f, l):
    lidx = {r[2]: r for r in l}
    dura = [(lidx[r[2]][0] - r[0]) / 1e9 for r in f]
    ns_f = [r[0] for r in f]
    num_f = [float(r[1]) for r in f]
    vol_f = [r[3] for r in f]
    vol_l = [r[3] for r in l]
    jp, jdiff = [], []
    lidset = {r[2]: r[3] for r in l}
    for r in f:
        if r[2] in lidset:
            jp.append(r[2])
            jdiff.append(lidset[r[2]] - r[3])
    return dict(
        dura_mean=np.mean(dura) if dura else float("nan"),
        dura_std=np.sqrt(np.var(dura, ddof=1)) if len(dura) > 1 else float("nan"),
        dura_max=np.max(dura) if dura else float("nan"),
        dura_skew=pd.Series(dura, dtype=float).skew(),
        dura_corr=cc([r[2] for r in f], dura),
        first_sum=np.sum(vol_f), last_sum=np.sum(vol_l),
        corr_first=cc(ns_f, num_f),
        corr_last=cc([r[0] for r in l], [float(r[1]) for r in l]),
        corr_first_last=cc(jp, jdiff),
    )


def sing1(aw, bw, cut):
    if not aw or not bw:
        return [np.nan] * 34
    af_rows = [r for r in aw if r[1] < cut]
    bf_rows = [r for r in bw if r[1] < cut]
    af, al = dedup(af_rows, True), dedup(af_rows, False)
    bf, bl = dedup(bf_rows, True), dedup(bf_rows, False)
    a, b = side(af, al), side(bf, bl)
    p1, d1, p2, d2 = [], [], [], []
    bfv = {r[2]: r[3] for r in bf}
    for r in af:
        if r[2] in bfv:
            p1.append(r[2]); d1.append(r[3] - bfv[r[2]])
    blv = {r[2]: r[3] for r in bl}
    for r in al:
        if r[2] in blv:
            p2.append(r[2]); d2.append(r[3] - blv[r[2]])
    ads = a["last_sum"] - a["first_sum"]
    bds = b["last_sum"] - b["first_sum"]
    abf = a["first_sum"] - b["first_sum"]
    abl = a["last_sum"] - b["last_sum"]
    return [
        a["dura_mean"], b["dura_mean"], a["dura_std"], b["dura_std"],
        a["dura_max"], b["dura_max"], a["dura_skew"], b["dura_skew"],
        a["dura_corr"], b["dura_corr"],
        a["first_sum"], a["last_sum"], b["first_sum"], b["last_sum"],
        ads, bds, abf, abl, abs(ads), abs(bds), abs(abf), abs(abl),
        abs(ads + bds), abs(ads - bds),
        ads / (a["last_sum"] + a["first_sum"]),
        bds / (b["last_sum"] + b["first_sum"]),
        a["first_sum"] / (a["first_sum"] + b["first_sum"]),
        a["last_sum"] / (a["last_sum"] + b["last_sum"]),
        a["corr_first"], a["corr_last"], b["corr_first"], b["corr_last"],
        a["corr_first_last"], b["corr_first_last"],
        cc(p1, d1), cc(p2, d2),
    ]


def mine(code, date):
    asks = read_book(code, date, "ask")
    bids = read_book(code, date, "bid")
    res = []
    for _n, lo, hi in SESSIONS:
        # between_time 比的是「当天的钟点」，不是绝对时间
        l = (pd.Timestamp(f"1970-01-01 {lo}").value % 86_400_000_000_000)
        h = (pd.Timestamp(f"1970-01-01 {hi}").value % 86_400_000_000_000)
        aw = [r for r in asks if l <= r[0] % 86_400_000_000_000 <= h]
        bw = [r for r in bids if l <= r[0] % 86_400_000_000_000 <= h]
        for cut, _suf in CUTS:
            res += sing1(aw, bw, cut)
    return res


sys.path.insert(0, "/home/chenzongwei/rust_pyfunc/lib/okay_database")
import verify_series as vs

_OKAY = "/home/chenzongwei/pythoncode/成果hm41_野火/hm41_okay.py"


def old_real(code, date):
    global _MOD, _CAP
    try:
        _MOD
    except NameError:
        _MOD, _CAP = vs.load_old(_OKAY)
    vals, _is_cross = vs.old_values(_MOD, _CAP, code, date)
    return [float("nan") if v is None else float(v) for v in vals]


class Probe(dwo.TickDataBase):
    def get_factor(self):
        return raw_old(self)


def raw_old(rc):
    asks, bids = rc.read_market_data_single_price_vol_pair()
    asks = asks.set_index("exchtime").between_time("09:30", "14:57").reset_index()
    bids = bids.set_index("exchtime").between_time("09:30", "14:57").reset_index()
    a_o = asks.set_index("exchtime").between_time("09:30", "10:00").reset_index()
    b_o = bids.set_index("exchtime").between_time("09:30", "10:00").reset_index()
    a_m = asks.set_index("exchtime").between_time("09:30", "11:30").reset_index()
    b_m = bids.set_index("exchtime").between_time("09:30", "11:30").reset_index()
    a_a = asks.set_index("exchtime").between_time("13:00", "14:57").reset_index()
    b_a = bids.set_index("exchtime").between_time("13:00", "14:57").reset_index()
    a_c = asks.set_index("exchtime").between_time("14:30", "14:57").reset_index()
    b_c = bids.set_index("exchtime").between_time("14:30", "14:57").reset_index()

    def sing1(asks, bids, cut_num):
        if asks.shape[0] > 0 and bids.shape[0] > 0:
            asks = asks[asks.number < cut_num]
            bids = bids[bids.number < cut_num]
            af = asks.drop_duplicates(subset="price", keep="first").set_index("price")
            al = asks.drop_duplicates(subset="price", keep="last").set_index("price")
            bf = bids.drop_duplicates(subset="price", keep="first").set_index("price")
            bl = bids.drop_duplicates(subset="price", keep="last").set_index("price")
            ad = (al.exchtime - af.exchtime).dt.total_seconds()
            bd = (bl.exchtime - bf.exchtime).dt.total_seconds()
            ads = al.vol.sum() - af.vol.sum()
            bds = bl.vol.sum() - bf.vol.sum()
            abf = af.vol.sum() - bf.vol.sum()
            abl = al.vol.sum() - bl.vol.sum()
            afl = pd.merge(af.rename(columns={"vol": "fv"}), al.rename(columns={"vol": "lv"}), on=["price"])
            bfl = pd.merge(bf.rename(columns={"vol": "fv"}), bl.rename(columns={"vol": "lv"}), on=["price"])
            afl = afl.assign(vd=afl.lv - afl.fv)
            bfl = bfl.assign(vd=bfl.lv - bfl.fv)
            fab = pd.merge(af.rename(columns={"vol": "av"}), bf.rename(columns={"vol": "bv"}), on=["price"])
            lab = pd.merge(al.rename(columns={"vol": "av"}), bl.rename(columns={"vol": "bv"}), on=["price"])
            fab = fab.assign(vd=fab.av - fab.bv)
            lab = lab.assign(vd=lab.av - lab.bv)
            r = [
                ad.mean(), bd.mean(), ad.std(), bd.std(), ad.max(), bd.max(), ad.skew(), bd.skew(),
                ad.reset_index().corr().iloc[0, 1], bd.reset_index().corr().iloc[0, 1],
                af.vol.sum(), al.vol.sum(), bf.vol.sum(), bl.vol.sum(), ads, bds, abf, abl,
                abs(ads), abs(bds), abs(abf), abs(abl), abs(ads + bds), abs(ads - bds),
                ads / (al.vol.sum() + af.vol.sum()), bds / (bl.vol.sum() + bf.vol.sum()),
                af.vol.sum() / (af.vol.sum() + bf.vol.sum()), al.vol.sum() / (al.vol.sum() + bl.vol.sum()),
                af.corr().iloc[0, 1], al.corr().iloc[0, 1], bf.corr().iloc[0, 1], bl.corr().iloc[0, 1],
                afl.reset_index()[["price", "vd"]].corr().iloc[0, 1],
                bfl.reset_index()[["price", "vd"]].corr().iloc[0, 1],
                fab.reset_index()[["price", "vd"]].corr().iloc[0, 1],
                lab.reset_index()[["price", "vd"]].corr().iloc[0, 1],
            ]
            return [float(i) for i in r]
        return [np.nan] * 34

    def sing(a, b):
        return sing1(a, b, 6) + sing1(a, b, 11)

    return (sing(asks, bids) + sing(a_o, b_o) + sing(a_m, b_m)
            + sing(a_a, b_a) + sing(a_c, b_c))


if __name__ == "__main__":
    pairs = [(20250303, "000001"), (20250304, "600519"),
             (20180601, "000002"), (20231020, "300750")]
    for d, c in pairs:
        old = old_real(c, d)
        new = mine(c, d)
        m = max(len(old), len(new))
        bad = []
        for i in range(m):
            a = old[i] if i < len(old) else float("nan")
            b = new[i] if i < len(new) else float("nan")
            if np.isnan(a) and np.isnan(b):
                continue
            if np.isnan(a) or np.isnan(b):
                bad.append((i, a, b, "一边NaN")); continue
            rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
            if not (rel <= 1e-9):
                bad.append((i, a, b, f"rel={rel:.2e}"))
        print(f"{c} {d}: 旧 {len(old)} 新 {len(new)} 不一致 {len(bad)}")
        for t in bad[:8]:
            name = BLOCK[t[0] % 34] if t[0] < 340 else f"raw#{t[0]}"
            print("   ", name, t[0], t[1], t[2], t[3])
