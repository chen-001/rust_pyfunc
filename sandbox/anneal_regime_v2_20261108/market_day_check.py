"""一天全市场（20241231）跑新版：耗时分布、各模块空值比例、退化列（全市场几乎同值）比例。50 进程。"""
import os, sys, time, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "package"))
DATE = 20241231

def run(code):
    import anneal_regime_v2_sbx as sb
    t = time.time()
    try:
        v = np.array(sb.py_anneal_regime_excess(code, DATE), dtype=np.float32)
    except Exception:
        return code, None, 0.0
    return code, v, time.time() - t

if __name__ == "__main__":
    import anneal_regime_v2_sbx as sb
    names = sb.py_anneal_regime_excess_names()
    files = sorted(os.listdir(f"/nvme/level2/{DATE}/transaction"))
    codes = [f[:6] for f in files if f[:6].isdigit() and f[0] in "036"]
    t0 = time.time()
    with Pool(50) as p:
        res = p.map(run, codes, chunksize=4)
    wall = time.time() - t0
    ok = [(c, v, dt) for c, v, dt in res if v is not None]
    M = np.stack([v for _, v, _ in ok])
    dts = np.array([dt for _, _, dt in ok])
    np.save("results/market_20241231.npy", M)
    def module(n):
        if n.startswith("s2_wdiff_"): return "M2时段对比"
        if n.startswith("s2_w"): return "M1时段"
        if n.startswith("s2_evt"): return "M6事件"
        if n.startswith("s2_scale"): return "M7尺度"
        seg = "尾盘30" if "tail30_" in n else ("下午" if "pm_" in n else "全天")
        res = "30秒" if "s30_" in n else "1分钟"
        kind = "M4状态" if n.startswith("s2_cond_") else ("M5四象限" if n.startswith("s2_quad_") else "M3降维")
        return f"{kind}-{seg}-{res}"
    mods = np.array([module(n) for n in names])
    nan_rate = np.isnan(M).mean(0)
    # 退化列：全市场取值最多的那个数占比 > 50%
    def top_share(col):
        col = col[np.isfinite(col)]
        if len(col) == 0: return 1.0
        _, cnt = np.unique(col, return_counts=True)
        return cnt.max() / len(col)
    tops = np.array([top_share(M[:, j]) for j in range(M.shape[1])])
    out = {"股票数": len(ok), "失败数": len(res) - len(ok), "墙钟秒(50进程)": round(wall, 1),
           "单股耗时中位/90分位/最大": [round(float(np.median(dts)), 3), round(float(np.quantile(dts, .9)), 3), round(float(dts.max()), 3)],
           "全部列空值率均值": round(float(nan_rate.mean()), 4), "空值率>4%的列数": int((nan_rate > 0.04).sum()),
           "众数占比>50%的退化列数": int((tops > 0.5).sum())}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    for m in sorted(set(mods)):
        k = mods == m
        print(f"{m:16s} 列数{k.sum():6d}  空值率均值{nan_rate[k].mean():.3f}  空值>4%列{(nan_rate[k] > .04).sum():5d}  退化列{(tops[k] > .5).sum():4d}")
    json.dump(out, open("results/market_20241231_summary.json", "w"), ensure_ascii=False, indent=1)
