import time
import numpy as np
import pandas as pd
import rust_pyfunc as rp

df = rp.read_trade('000001', 20220819)
df = rp.adjust_afternoon(df, only_inday=1)
df = df.sort_values(['volume', 'exchtime'])
n = len(df)
print('rows after adjust+sort: %d' % n, flush=True)

vol = np.ascontiguousarray(df.volume.to_numpy(dtype=float))
ns = np.ascontiguousarray(df.exchtime.to_numpy().astype('int64').astype('float64'))
price = np.ascontiguousarray(df.price.to_numpy(dtype=float))
flag = np.ascontiguousarray(df.flag.to_numpy(dtype=float))
asks = np.ascontiguousarray(df.ask_order.to_numpy(dtype=float))
bids = np.ascontiguousarray(df.bid_order.to_numpy(dtype=float))

# ---- 复刻 orders 聚合，算 volume 分组规模 ----
t = df.exchtime.to_numpy().astype('int64') / 1e9
agg = {}
v = df.volume.to_numpy(dtype=np.int64)
pr = df.price.to_numpy(dtype=float)
a = df.ask_order.to_numpy()
b = df.bid_order.to_numpy()
for i in range(n):
    if a[i] != 0:
        oi = a[i]
        e = agg.get(oi)
        if e is None: agg[oi] = [v[i], pr[i], t[i]]
        else:
            nv = e[0] + v[i]; e[1] = (e[0] * e[1] + v[i] * pr[i]) / nv; e[0] = nv; e[2] = max(e[2], t[i])
    if b[i] != 0:
        oi = b[i]
        e = agg.get(oi)
        if e is None: agg[oi] = [v[i], pr[i], t[i]]
        else:
            nv = e[0] + v[i]; e[1] = (e[0] * e[1] + v[i] * pr[i]) / nv; e[0] = nv; e[2] = max(e[2], t[i])
ov = np.array([x[0] for x in agg.values()])
print('唯一订单数 = %d' % len(ov), flush=True)
sizes = np.sort(np.unique(ov, return_counts=True)[1])[::-1]
top = sizes[:10]
print('volume 分组数 = %d，最大的 10 组规模 = %s' % (len(sizes), top.tolist()), flush=True)
print('分组规模平方和 sum(k^2) = %.3e ；sum(k^2*log2 k) = %.3e' % ((sizes.astype(float)**2).sum(), ((sizes.astype(float)**2)*np.log2(np.maximum(sizes,2))).sum()), flush=True)
print('前 1 大组占全体比例 = %.1f%%' % (100.0*top[0]/len(ov)), flush=True)

for direction in ('same', 'diff'):
    ts = []
    for _ in range(2):
        t0 = time.time()
        flat = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(vol, ns, price, flag, asks, bids, 100, direction)
        ts.append(time.time() - t0)
    print('ultra_sorted_v2 direction=%-5s best=%.2f s shape=%s' % (direction, min(ts), np.asarray(flat).shape), flush=True)
