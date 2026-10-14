"""日内窗口测试：用 [t0,t1) 的数据构造因子，预测 [t1,t1+H] 的中价收益（横截面 RankIC）"""
import numpy as np, pandas as pd, os
from multiprocessing import Pool
from scipy.stats import spearmanr
import importlib.util
spec = importlib.util.spec_from_file_location("p", "pilot_factors.py")
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)

DATES = ["20260916","20260918","20260922","20260924"]
CODES = p.pick_codes()

def t2i(h,m,s=0,ms=0): return ((h*100+m)*100+s)*1000+ms

def window_factors(t, m, t0, t1):
    mt = m["time"].to_numpy(); sess = (mt>=t0)&(mt<t1)
    _m = m[sess]
    out = {}
    if len(_m) < 10: return None
    sf = p.snapshot_factors(_m.reset_index(drop=True))
    if sf is not None: out.update(sf[0])
    tt = t[(t["time"].to_numpy()>=t0)&(t["time"].to_numpy()<t1)]
    if len(tt) > 100:
        tf = p.trade_factors(tt, None)
        if tf: out.update({f"w_{k}": v for k, v in tf.items() if k!="id_flag_agree"})
    mid = 0.5*(m["ask_prc1"].to_numpy()+m["bid_prc1"].to_numpy())
    sel = (mt>=t1)&(mid>0)
    if sel.sum() < 3: return None
    out["mid_t1"] = float(mid[sel][0])
    return out

def one(args):
    code, date, tag, t0, t1, t2 = args
    t, m = p.load_day(code, date)
    if t is None: return None
    f0 = window_factors(t, m, t0, t1)
    f1 = window_factors(t, m, t1, t2)
    if f0 is None or f1 is None: return None
    # 用窗口末尾一段的 VWAP(mid) 作为价格参照，避免单点噪声
    mt = m["time"].to_numpy(); mid = 0.5*(m["ask_prc1"].to_numpy()+m["bid_prc1"].to_numpy())
    def ref(a,b):
        s=(mt>=a)&(mt<b)&(mid>0)
        return float(np.mean(mid[s])) if s.sum()>=3 else np.nan
    r = ref(t1,t2)/f0["mid_t1"] - 1.0
    d = {k:v for k,v in f0.items() if k!="mid_t1"}
    d.update(code=code, date=date, tag=tag, fwd=r)
    return d

if __name__ == "__main__":
    T = {"morning->afternoon": (t2i(9,30), t2i(11,30), t2i(14,0)),
         "13:00-14:30->close": (t2i(13,0), t2i(14,30), t2i(15,0)),
         "whole_day_am->pm_close": (t2i(9,30), t2i(11,30), t2i(15,0))}
    tasks = [(c,d,tag,*win) for c in CODES for d in DATES for tag,win in T.items()]
    with Pool(24) as pool:
        res = pool.map(one, tasks, chunksize=4)
    df = pd.DataFrame([r for r in res if r])
    df.to_csv("intraday_raw.csv", index=False)
    fac = [c for c in df.columns if c not in ("code","date","tag","fwd")]
    rows=[]
    for tag, g0 in df.groupby("tag"):
        for f in fac:
            ics=[]
            for d, g in g0.groupby("date"):
                g = g.dropna(subset=[f,"fwd"])
                if len(g)<30: continue
                ics.append(spearmanr(g[f], g["fwd"]).statistic)
            if len(ics)<3: continue
            ics=np.array(ics)
            rows.append(dict(tag=tag, factor=f, IC=ics.mean(),
                             t=ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics)), n=len(ics)))
    r = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
    pd.set_option("display.width",200,"display.float_format",lambda x:f"{x: .4f}")
    print(r.head(30).to_string(index=False))
