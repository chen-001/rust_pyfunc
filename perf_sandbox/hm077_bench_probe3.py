import time
import numpy as np
import rust_pyfunc as rp

t0 = time.time(); rp.okay_compute_stock('hm077', '000001', 20220819); warm = time.time() - t0
print('warmup full = %.2f s' % warm, flush=True)

t0 = time.time(); v = rp.okay_compute_stock('hm077', '000001', 20220819); full = time.time() - t0
print('FULL okay_compute_stock = %.2f s  (%d values)' % (full, len(v)), flush=True)

df = rp.read_trade('000001', 20220819)
df = rp.adjust_afternoon(df, only_inday=1)
df = df.sort_values(['volume', 'exchtime'])
vol = np.ascontiguousarray(df.volume.to_numpy(dtype=float))
ns = np.ascontiguousarray(df.exchtime.to_numpy().astype('int64').astype('float64'))
price = np.ascontiguousarray(df.price.to_numpy(dtype=float))
flag = np.ascontiguousarray(df.flag.to_numpy(dtype=np.int32))
asks = np.ascontiguousarray(df.ask_order.to_numpy(dtype=np.int64))
bids = np.ascontiguousarray(df.bid_order.to_numpy(dtype=np.int64))
for direction in ('same', 'diff'):
    t0 = time.time()
    rp.calculate_order_time_gap_and_price_percentile_ultra_sorted_v2(vol, ns, price, flag, asks, bids, 100, direction)
    print('  ultra_sorted_v2 %-5s = %.2f s' % (direction, time.time() - t0), flush=True)
