"""Pilot: 逐笔成交 + 盘口快照 新因子快速可行性测试（日频横截面 RankIC）"""
import os, sys, glob, json, random
import numpy as np, pandas as pd
from multiprocessing import Pool

ROOT = "/ssd_data/stock"
DATES = ["20260914","20260915","20260916","20260917","20260918","20260921","20260922","20260923","20260924"]
NSTOCK = 150
SEED = 7

def pick_codes():
    d = DATES[0]
    files = glob.glob(f"{ROOT}/{d}/transaction/*_transaction.csv")
    codes = []
    for f in files:
        c = os.path.basename(f).split("_")[0]
        if c[:3] in ("000","001","002","300","301","600","601","603","605"):
            codes.append(c)
    random.Random(SEED).shuffle(codes)
    return sorted(codes[:NSTOCK])

def load_day(code, date):
    tf = f"{ROOT}/{date}/transaction/{code}_{date}_transaction.csv"
    mf = f"{ROOT}/{date}/market_data/{code}_{date}_market_data.csv"
    if not (os.path.exists(tf) and os.path.exists(mf)):
        return None, None
    try:
        t = pd.read_csv(tf, usecols=["time","price","volume","flag","ask_order","bid_order"],
                        dtype={"time":np.int64,"price":np.float64,"volume":np.float64,
                               "flag":np.int32,"ask_order":np.int64,"bid_order":np.int64})
        m = pd.read_csv(mf, usecols=["time","last_prc","status","turnover","volume",
                                     "weighted_ask_prc","weighted_bid_prc","total_ask_vol","total_bid_vol"]
                        + [f"ask_prc{i}" for i in range(1,11)] + [f"ask_vol{i}" for i in range(1,11)]
                        + [f"bid_prc{i}" for i in range(1,11)] + [f"bid_vol{i}" for i in range(1,11)])
    except Exception as e:
        print("ERR", code, date, e); return None, None
    return t, m

def snapshot_factors(m):
    askv = m[[f"ask_vol{i}" for i in range(1,11)]].to_numpy().sum(1)
    bidv = m[[f"bid_vol{i}" for i in range(1,11)]].to_numpy().sum(1)
    tot_a = m["total_ask_vol"].to_numpy(); tot_b = m["total_bid_vol"].to_numpy()
    a1 = m["ask_prc1"].to_numpy(); b1 = m["bid_prc1"].to_numpy()
    mid = 0.5*(a1+b1)
    t = m["time"].to_numpy()
    sess = (t>=93000000)&(t<=145700000)&(mid>0)&(tot_a>0)&(tot_b>0)&(askv>0)&(bidv>0)
    if sess.sum() < 50: return None
    A1, B1, MI = a1[sess], b1[sess], mid[sess]
    TA, TB, AV, BV = tot_a[sess], tot_b[sess], askv[sess], bidv[sess]
    hid_a = np.clip(TA-AV, 0, None); hid_b = np.clip(TB-BV, 0, None)
    hs = np.maximum(0.5*(A1-B1), 0.005)          # 半价差，下限 1 tick 的一半
    out = {}
    out["hid_imb"] = float(np.mean((hid_b-hid_a)/np.maximum(hid_b+hid_a,1e-9)))
    out["vdr_ask"] = float(np.mean(AV/TA)); out["vdr_bid"] = float(np.mean(BV/TB))
    out["vdr_logratio"] = float(np.mean(np.log(np.maximum(BV/TB,1e-9))-np.log(np.maximum(AV/TA,1e-9))))
    w = 0.5*(m["weighted_ask_prc"].to_numpy()[sess]+m["weighted_bid_prc"].to_numpy()[sess])
    out["fb_skew"] = float(np.mean((w-MI)/hs))
    out["obi_l1"] = float(np.mean((m["bid_vol1"].to_numpy()[sess]-m["ask_vol1"].to_numpy()[sess])
                                  /np.maximum(m["bid_vol1"].to_numpy()[sess]+m["ask_vol1"].to_numpy()[sess],1e-9)))
    ap = m[[f"ask_prc{i}" for i in range(1,11)]].to_numpy()[sess]; av = m[[f"ask_vol{i}" for i in range(1,11)]].to_numpy()[sess]
    bp = m[[f"bid_prc{i}" for i in range(1,11)]].to_numpy()[sess]; bv = m[[f"bid_vol{i}" for i in range(1,11)]].to_numpy()[sess]
    with np.errstate(invalid="ignore", divide="ignore"):
        cen_a = (ap*av).sum(1)/np.maximum(av.sum(1),1e-9)
        cen_b = (bp*bv).sum(1)/np.maximum(bv.sum(1),1e-9)
        out["book_centroid"] = float(np.mean(((cen_a-A1)-(B1-cen_b))/(2*hs)))
    # 撤单压力也放进盘口侧：撤单量相对盘口可见量的比例（同侧）
    return out, sess

def trade_factors(t, m):
    t = t.copy()
    tm = t["time"].to_numpy()
    sess = (tm>=93000000)&(tm<=145700000)
    tt = t[sess]
    if len(tt) < 200: return None
    fl = tt["flag"].to_numpy(); vol = tt["volume"].to_numpy()
    is_cancel = fl==32; is_trade = (fl==66)|(fl==83)
    ao = tt["ask_order"].to_numpy(); bo = tt["bid_order"].to_numpy()
    out = {}
    # 撤单
    cv = vol[is_cancel]; ca = ao[is_cancel]; cb = bo[is_cancel]
    ca_v = cv[ca>0].sum(); cb_v = cv[cb>0].sum()
    out["cancel_imb"] = (cb_v-ca_v)/max(cb_v+ca_v,1e-9)          # >0 = 买盘撤单多
    out["cancel_ratio"] = cv.sum()/max(cv.sum()+vol[is_trade].sum(),1e-9)
    if cv.size>10 and vol[is_trade].size>10:
        out["cancel_size_rel"] = np.log(max(np.median(cv),1)/max(np.median(vol[is_trade]),1))
    # 逐笔成交：订单链路
    tv = vol[is_trade]; tao = ao[is_trade]; tbo = bo[is_trade]
    df = pd.DataFrame({"v":tv,"ao":tao,"bo":tbo})
    # 被动单被啃：同一订单号出现的成交笔数 >=2 的量占比
    fills = {}
    for o in np.concatenate([tao, tbo]):
        if o>0: fills[o] = fills.get(o,0)+1
    cnt = pd.Series(fills)
    oa = pd.Series(tao).map(cnt).fillna(1).to_numpy(); ob = pd.Series(tbo).map(cnt).fillna(1).to_numpy()
    multi = ((oa>=2)|(ob>=2))
    out["fill_rhythm"] = tv[multi].sum()/max(tv.sum(),1e-9)
    out["fills_per_order"] = float(np.mean(np.concatenate([oa,ob])))
    # 事件时间挂单年龄：两次委托序号之差（谁大谁是主动方）
    gap = np.abs(tao-tbo).astype(float)
    out["age_evt"] = float(np.average(np.log1p(gap), weights=np.maximum(tv,1)))
    # 扫单：主动方订单号在 >1 笔成交中出现 -> 一次下单吃多档
    agg = np.where(tao>tbo, tao, tbo)
    acnt = pd.Series(agg).value_counts()
    asweep = pd.Series(agg).map(acnt).to_numpy()
    out["sweep_ratio"] = tv[asweep>=2].sum()/max(tv.sum(),1e-9)
    out["sweep_depth"] = float(np.average(np.clip(asweep,1,50), weights=np.maximum(tv,1)))
    # 主动买卖失衡（用序号判定主动方 + flag 交叉验证）
    buy = (tbo>tao)
    out["id_ofi"] = float((tv[buy].sum()-tv[~buy].sum())/max(tv.sum(),1e-9))
    out["id_flag_agree"] = float(np.mean(buy == (fl[is_trade]==66)))
    # 成交额加权平均价格相对收盘  （日内动量/位置）
    return out

def one(args):
    code, date = args
    t, m = load_day(code, date)
    if t is None: return (code, date, None)
    sf = snapshot_factors(m)
    if sf is None: return (code, date, None)
    s, _ = sf
    f = trade_factors(t, m)
    if f is None: return (code, date, None)
    s.update(f)
    last = m.iloc[-1]
    prev = m[m["last_prc"]>0]
    if len(prev)==0: return (code, date, None)
    s["close"] = float(prev["last_prc"].iloc[-1])
    s["turnover"] = float(m["turnover"].iloc[-1])
    s["code"]=code; s["date"]=date
    return (code, date, s)

if __name__ == "__main__":
    codes = pick_codes()
    print("codes", len(codes), codes[:10], flush=True)
    tasks = [(c,d) for c in codes for d in DATES]
    with Pool(24) as p:
        res = p.map(one, tasks, chunksize=8)
    rows = [r[2] for r in res if r[2] is not None]
    df = pd.DataFrame(rows)
    print("stock-days:", len(df), "unique codes:", df["code"].nunique(), flush=True)
    out = "/home/chenzongwei/rust_pyfunc/experiments/tick_ob_ideas/pilot_raw.csv"
    df.to_csv(out, index=False)
    print("saved", out)
