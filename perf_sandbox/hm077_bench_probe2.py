import time
import numpy as np
import rust_pyfunc as rp

df = rp.read_trade('000001', 20220819)
df = rp.adjust_afternoon(df, only_inday=1)
df = df.sort_values(['volume', 'exchtime'])
vol = np.ascontiguousarray(df.volume.to_numpy(dtype=float))
ns = np.ascontiguousarray(df.exchtime.to_numpy().astype('int64').astype('float64'))
price = np.ascontiguousarray(df.price.to_numpy(dtype=float))
flag = np.ascontiguousarray(df.flag.to_numpy(dtype=np.int32))
asks = np.ascontiguousarray(df.ask_order.to_numpy(dtype=np.int64))
bids = np.ascontiguousarray(df.bid_order.to_numpy(dtype=np.int64))
print('rows %d' % len(df), flush=True)
for direction in ('same', 'diff'):
    ts = []
    for _ in range(2):
        t0 = time.time()
        flat = rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(vol, ns, price, flag, asks, bids, 100, direction)
        ts.append(time.time() - t0)
    print('ultra_sorted_v2 direction=%-5s best=%.2f s  rows=%d cols=%d' % (direction, min(ts), len(flat), len(flat[0])), flush=True)
