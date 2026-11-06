"""去雷同：6 个样本日上算列与列的截面秩相关（取 |相关| 的日均值），
按「ext1 保留因子优先，其余按 gap5 5日平均 |ICIR| 从高到低」贪心保留，|相关|>阈值的后来者剔除。"""
import json, numpy as np, pandas as pd
from scipy.stats import rankdata
O = "/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out"
names = json.load(open(f"{O}/names_w1.json"))
dates = [int(x) for x in open(f"{O}/dates.txt").read().split()]
keep = [l.strip() for l in open("keep_ext1.txt") if l.strip()]
summ = pd.read_parquet(f"{O}/summary.parquet")
sample = [dates[i] for i in (2, 20, 30, 48, 60, 90)]
k = len(names)
acc = np.zeros((k, k), dtype=np.float32); nanr = np.zeros(k)
for d in sample:
    codes = json.load(open(f"{O}/{d}_w1.codes.json"))
    m = np.fromfile(f"{O}/{d}_w1.bin", dtype=np.float32).reshape(len(codes), k)
    nanr += np.isnan(m).mean(0) / len(sample)
    r = rankdata(m, axis=0, nan_policy="omit").astype(np.float32)
    r -= np.nanmean(r, 0); r = np.nan_to_num(r)            # 缺失按均值（0）处理
    sd = np.sqrt((r ** 2).sum(0)); sd[sd == 0] = np.inf
    r /= sd
    acc += np.abs(r.T @ r) / len(sample)
    print(d, flush=True)
np.save(f"{O}/abscorr.npy", acc); np.save(f"{O}/nanrate.npy", nanr)
