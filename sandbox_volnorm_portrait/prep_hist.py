"""想法B预计算:目标日前20个交易日的1分钟均量(volume.h5)与当天1分钟量 → CSV。

输入:rust_pyfunc.read_minute_data('volume', start, end)(每240行一天,列=6位代码)
输出(每目标日2个文件,sandbox_volnorm_portrait/data/ 下):
  hm20_{date}.csv  code,v0..v239   20天逐分钟 nanmean(停牌日自动排除)
  day_{date}.csv   code,v0..v239   当天逐分钟量
NaN 写文本 "nan"。
"""
import os
import numpy as np
import rust_pyfunc as rp

DATES = [20240104, 20240603, 20241008, 20260105, 20260717]
WIN = 20
OUT = "/home/chenzongwei/rust_pyfunc/sandbox_volnorm_portrait/data"
os.makedirs(OUT, exist_ok=True)

days = np.asarray(rp.td.trading_days)


def day_of(date):
    return int(np.where(days == date)[0][0])


def write_csv(path, codes, mat):
    with open(path, "w") as f:
        f.write("code," + ",".join(f"v{i}" for i in range(240)) + "\n")
        for c, row in zip(codes, mat):
            vals = ",".join("nan" if not np.isfinite(x) else f"{x:.6g}" for x in row)
            f.write(f"{c},{vals}\n")


for date in DATES:
    i = day_of(date)
    start = int(days[i - WIN])
    df = rp.read_minute_data("volume", start, date)
    n_days = df.shape[0] // 240
    codes = list(df.columns)
    arr = df.to_numpy(dtype=np.float64).reshape(n_days, 240, -1)
    hist = arr[:WIN]  # (20, 240, N)
    cur = arr[WIN - 1]  # 当天
    hm = np.nanmean(hist, axis=0)  # (240, N)
    write_csv(f"{OUT}/hm20_{date}.csv", codes, hm.T)
    write_csv(f"{OUT}/day_{date}.csv", codes, cur.T)
    valid = np.isfinite(hm).sum(axis=0)
    print(date, "n_days", n_days, "n_codes", len(codes),
          "hist_nan_cols", int((valid == 0).sum()), "min_ok_buckets", int(valid.min()))
print("done")
