import numpy as np, pandas as pd, rust_pyfunc as rp
df = rp.read_trade('000001', 20220819)
df = rp.adjust_afternoon(df, only_inday=1)
df = df.sort_values(['volume', 'exchtime'])
v = df.volume.to_numpy(dtype=np.int64); pr = df.price.to_numpy(dtype=float)
t = df.exchtime.to_numpy().astype('int64') / 1e9
a = df.ask_order.to_numpy(); b = df.bid_order.to_numpy()
agg = {}
for i in range(len(df)):
    if a[i] != 0:
        oi = a[i]; e = agg.get(oi)
        if e is None: agg[oi] = [v[i], pr[i], t[i], False]
        else:
            nv = e[0] + v[i]; e[1] = (e[0]*e[1] + v[i]*pr[i])/nv; e[0] = nv; e[2] = max(e[2], t[i])
    if b[i] != 0:
        oi = b[i]; e = agg.get(oi)
        if e is None: agg[oi] = [v[i], pr[i], t[i], True]
        else:
            nv = e[0] + v[i]; e[1] = (e[0]*e[1] + v[i]*pr[i])/nv; e[0] = nv; e[2] = max(e[2], t[i])
arr = np.array([(x[0], 1.0 if x[3] else 0.0) for x in agg.values()])
ov = arr[:, 0]
print('orders=%d  is_bid=true 的占比=%.3f' % (len(ov), arr[:,1].mean()))
vals, counts = np.unique(ov, return_counts=True)
ka = np.zeros(len(vals)); kb = np.zeros(len(vals))
for i in range(len(ov)):
    j = np.searchsorted(vals, ov[i])
    if arr[i,1] > 0.5: kb[j] += 1
    else: ka[j] += 1
# 只算 min_count=100 的组（group_size>=100 且 target side>=100）
ok = (ka + kb) >= 100
same = (ka[ok]*(ka[ok]-1) + kb[ok]*(kb[ok]-1)).sum()
diff = (2*ka[ok]*kb[ok]).sum()
print('volume 分组数=%d，其中规模>=100 的 %d 组' % (len(vals), ok.sum()))
print('same 方向元素数 = %.4e   diff 方向元素数 = %.4e' % (same, diff))
print('按 15.8 ns/元素估算： same %.1f s   diff %.1f s' % (same*15.8e-9, diff*15.8e-9))
