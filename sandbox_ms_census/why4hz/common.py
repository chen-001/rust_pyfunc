# -*- coding: utf-8 -*-
"""沪市一秒四个高峰的排查：共用工具。抽样探查，不进正式结果。"""
import numpy as np, pandas as pd, random, os, json, multiprocessing as mp
PY_SEED = 7
def sample(date, prefix, n, seed=PY_SEED):
    v=[l.strip() for l in open(f'/home/chenzongwei/rust_pyfunc/sandbox_ms_census/out/valid_{date}.txt')]
    cs=[c for c in v if c.startswith(prefix)]
    random.Random(seed).shuffle(cs); return cs[:n]
def session_mask(t):
    return ((t>=93000000)&(t<113000000))|((t>=130000000)&(t<145700000))
def cs_of(ex):  # 东八区日内 10 毫秒序号
    return np.round(((ex.astype(float)+8*3600e6)%86400e6)/1e4).astype(np.int64)
def trades(date,c):
    t=pd.read_csv(f'/ssd_data/stock/{date}/transaction/{c}_{date}_transaction.csv',usecols=['time','exchtime','localtime','volume','turnover','flag','ask_order','bid_order','price','index'])
    return t
def orders(date,c):
    p=f'/hdd/order_data/{date}/order/{c}_{date}_order.csv'
    return pd.read_csv(p,usecols=['time','exchtime','localtime','order','price','volume','func_code'])
