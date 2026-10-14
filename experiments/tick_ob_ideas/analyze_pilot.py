import numpy as np, pandas as pd
from scipy.stats import spearmanr
df = pd.read_csv("pilot_raw.csv", dtype={"code":str,"date":str})
px = df.pivot(index="date", columns="code", values="close").sort_index()
fwd = (px.shift(-1)/px - 1.0)                      # 次日 close-to-close
df["fwd"] = [fwd.loc[d, c] if (d in fwd.index and c in fwd.columns) else np.nan
             for d, c in zip(df["date"], df["code"])]
df["turn_log"] = np.log(df["turnover"].clip(lower=1))
fac = [c for c in df.columns if c not in ("code","date","close","turnover","fwd","turn_log","id_flag_agree")]
rows = []
for f in fac:
    ics, n = [], 0
    for d, g in df.groupby("date"):
        g = g.dropna(subset=[f,"fwd"])
        if len(g) < 30: continue
        r, _ = spearmanr(g[f], g["fwd"]); ics.append(r); n += 1
    if len(ics) < 4: continue
    ics = np.array(ics)
    rows.append(dict(factor=f, IC=ics.mean(), IR=ics.mean()/ics.std(ddof=1),
                     t=ics.mean()/ics.std(ddof=1)*np.sqrt(len(ics)),
                     pos=f"{int((ics>0).sum())}/{len(ics)}",
                     cs_corr_turn=spearmanr(df[f], df["turn_log"], nan_policy="omit").statistic))
r = pd.DataFrame(rows).sort_values("t", key=abs, ascending=False)
pd.set_option("display.width", 200, "display.float_format", lambda x: f"{x: .4f}")
print(r.to_string(index=False))
print("\n=== 因子间相关性 (Spearman, 全样本) ===")
m = df[fac].corr(method="spearman")
print(m.round(2).to_string())
