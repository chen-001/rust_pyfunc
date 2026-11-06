"""拓展二沙箱评估准备：挑 4 段各 25 个连续交易日，导出每天的 ln 总市值给 Rust 做风格剔除。"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/chenzongwei/.agents/skills/daily-data-reader")
import daily_data_reader as ddr
O = "/hdd/user_home_unsafe/chenzongwei/sandbox_lfsr_ext2_out"
days = sorted(int(d) for d in os.listdir("/ssd_data/stock") if d.isdigit())
starts = [20190401, 20210802, 20230504, 20241101]
blocks = [[d for d in days if d >= s][:25] for s in starts]
dates = [d for b in blocks for d in b]
open(f"{O}/dates.txt", "w").write("\n".join(map(str, dates)) + "\n")
cap = ddr.total_cap(start=dates[0], end=dates[-1])
os.makedirs(f"{O}/ctl", exist_ok=True)
for d in dates:
    s = cap.loc[d].dropna()
    s = np.log(s[s > 0])
    pd.DataFrame({"code": [c[:6] for c in s.index], "logcap": s.values}).to_csv(f"{O}/ctl/{d}.csv", index=False)
print(len(dates), dates[0], dates[-1], cap.columns[:3].tolist(), len(s))
