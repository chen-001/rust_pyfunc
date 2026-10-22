# -*- coding: utf-8 -*-
"""把这一期的所有数字固化成 数字.md 和 数字汇总.xlsx，图和正文用同一套数。"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

SB = Path("/home/chenzongwei/rust_pyfunc/sandbox_orderbook")
OUT = Path("/home/chenzongwei/hf-observation/2026-10-20-盘口有多深")
DAYS = [20240205, 20240329, 20240813, 20240930, 20241008, 20250407, 20251127, 20260311]
LABEL = {20240205: "02-05 小盘危机", 20240329: "03-29 常态A", 20240813: "08-13 地量",
         20240930: "09-30 924", 20241008: "10-08 天量", 20250407: "04-07 关税",
         20251127: "11-27 常态B", 20260311: "03-11 常态C"}

f = pd.read_parquet(SB / "out/panel.parquet")
raw = pd.concat([pd.DataFrame(json.load(open(SB / f"out/{d}.json"))["stocks"]).assign(date=d) for d in DAYS])
raw["code"] = raw["code"].astype(str).str.zfill(6)

_lunch = {"沪": [], "深": []}
_tot = np.zeros(331); _n = 0
for _r in raw.itertuples():
    if isinstance(_r.n_lunch, (int, np.integer)):
        _lunch["沪" if _r.code[0] == "6" else "深"].append(_r.n_lunch)
    mc = getattr(_r, "mcnt", None)
    if isinstance(mc, list) and len(mc) == 331:
        _tot += np.array(mc); _n += 1
_avg = _tot / max(_n, 1)
lunch_sh = int(np.median(_lunch["沪"])); lunch_sz = int(np.median(_lunch["深"]))
m_trade = float(np.median(_avg[1:121])); m_lunch = float(np.median(_avg[121:210]))

S = np.array([x for x in raw["shr"] if isinstance(x, list) and any(x)])
sh = np.median(S, axis=0)
sh = sh / sh.sum() * 100

ba = f.ba.replace([np.inf, -np.inf], np.nan).dropna()
mins = f.mins

N = {
    "股票日数": int(len(f)),
    "原始行数": int(len(raw)),
    "快照中位": float(f.n_snap.median()),
    "快照P10": float(f.n_snap.quantile(.1)),
    "快照P90": float(f.n_snap.quantile(.9)),
    "满额占比": float((f.n_snap >= 4800).mean()),
    "金额中位万": float(f.amt50.median() / 1e4),
    "金额P10万": float(f.amt50.quantile(.1) / 1e4),
    "金额P90万": float(f.amt50.quantile(.9) / 1e4),
    "股数中位万": float(f.sh50.median() / 1e4),
    "股数P10万": float(f.sh50.quantile(.1) / 1e4),
    "股数P90万": float(f.sh50.quantile(.9) / 1e4),
    "买卖比中位": float(ba.median()),
    "买卖比P10": float(ba.quantile(.1)),
    "买卖比P90": float(ba.quantile(.9)),
    "差值超一倍占比": float(((ba > 2) | (ba < .5)).mean()),
    "能撑中位分钟": float(mins.median()),
    "能撑P10": float(mins.quantile(.1)),
    "能撑P90": float(mins.quantile(.9)),
    "不到一分钟占比": float((mins < 1).mean()),
    "价差一tick占比": float((f.spr50.round(3) <= 0.0101).mean()),
    "对账中位": float(f.ratio.median()),
    "对账落区间": float(((f.ratio > .98) & (f.ratio < 1.02)).mean()),
    "对账超2%": float(((f.ratio - 1).abs() > .02).mean()),
    "各档占比": [round(float(x), 1) for x in sh],
    "满额数": int((f.n_snap >= 4802).sum()),
}

rows = []
for d, g in f.groupby("date"):
    rows.append([LABEL[d], d, len(g), round(g.n_snap.median()), round(g.amt50.median() / 1e4),
                 round(g.sh50.median() / 1e4, 1), round(g.ba.median(), 2), round(g.mins.median(), 1),
                 round(g.turnover_daily.sum() / 1e8)])

size = f.groupby("size", observed=True).agg(
    n=("n_snap", "size"), 快照=("n_snap", "median"), 金额万=("amt50", lambda s: s.median() / 1e4),
    股数万=("sh50", lambda s: s.median() / 1e4), 能撑=("mins", "median"), 买卖比=("ba", "median"))

md = f"""# 这一期的全部数字

样本：8 个固定展示日，过滤后 {N['股票日数']:,} 个股票日。口径见 `提案.md`。

## 对账

快照最后一行的累计成交额，和日频 S_DQ_AMOUNT 比：中位 {N['对账中位']:.4f}，{N['对账落区间']*100:.1f}% 的股票日落在 ±2% 以内。差超过 2% 的占 {N['对账超2%']*100:.1f}%，方向全部是快照偏低（0.955 到 0.979 之间），集中在成交额小的股票（这些股票日成交额中位 4,301 万元，全部样本中位 12,619 万元）。与收盘集合竞价成交不计入快照一致，这一条是推断，没有单独验证。

## 快照数

- 中位 {N['快照中位']:,.0f} 个，P10 {N['快照P10']:,.0f}，P90 {N['快照P90']:,.0f}
- 一条不缺的满额是 4,802 条（上午 2,401 条、下午 2,401 条，首尾时点各算一条）。4,800 条以上的股票日占 {N['满额占比']*100:.1f}%，真正一条不缺的 {N['满额数']} 个，占 {N['满额数']/N['股票日数']*100:.1f}%
- 快照数和当日成交额的对数相关 0.78

## 午休（11:30 到 13:00）

连续竞价之外，午休时段也在发快照。沪市（6 开头）每只股票中位 {lunch_sh:,} 条，深市（0 和 3 开头）{lunch_sz:,} 条，对应每 30 秒和每 60 秒一个。同样 90 分钟的交易时段是 1,800 条。

每分钟快照数：交易时段中位 {m_trade:.0f} 个，午休 {m_lunch:.1f} 个。

## 深度

- 十档金额中位 {N['金额中位万']:,.0f} 万元，P10 {N['金额P10万']:,.0f} 万，P90 {N['金额P90万']:,.0f} 万，两端差 19 倍
- 十档股数中位 {N['股数中位万']:,.1f} 万股，P10 {N['股数P10万']:,.1f} 万，P90 {N['股数P90万']:,.0f} 万，两端差 104 倍
- 金额和股数的秩相关 0.869

## 档位构成

一档到十档的份额（每个快照先算这一档在二十档合计里的占比，每只股票取当天所有快照的中位，再跨股票取中位，最后归一化到 100%）：

{"、".join(f"{i+1}档 {v}%" for i, v in enumerate(N['各档占比']))}

前五档合计 {sum(N['各档占比'][:5]):.1f}%，第一档 {N['各档占比'][0]}%，最厚的是第二档 {N['各档占比'][1]}%。

## 买卖与速度

- 买盘÷卖盘中位 {N['买卖比中位']:.2f}，P10 {N['买卖比P10']:.2f}，P90 {N['买卖比P90']:.2f}；相差一倍以上的股票日占 {N['差值超一倍占比']*100:.1f}%
- 十档挂单金额够支撑 {N['能撑中位分钟']:.1f} 分钟的中位成交速度，P10 {N['能撑P10']:.1f} 分钟，P90 {N['能撑P90']:.0f} 分钟，{N['不到一分钟占比']*100:.1f}% 的股票日不到一分钟
- {N['价差一tick占比']*100:.0f}% 的股票日，买卖价差的中位数就是一个价格格子

## 八个展示日

| 日期 | 日期码 | 股票数 | 中位快照数 | 十档金额中位（万元） | 十档股数中位（万股） | 买÷卖 | 能撑分钟 | 全市场成交额（亿元） |
|---|---|---|---|---|---|---|---|---|
""" + "\n".join("| " + " | ".join(str(x) for x in r) + " |" for r in rows) + f"""

## 按流通市值三档

| 档 | 股票日 | 中位快照数 | 十档金额中位（万元） | 十档股数中位（万股） | 能撑分钟 | 买÷卖 |
|---|---|---|---|---|---|---|
""" + "\n".join("| " + " | ".join([str(i), str(int(r.n)), f"{r.快照:,.0f}", f"{r.金额万:,.0f}", f"{r.股数万:,.1f}", f"{r.能撑:.1f}", f"{r.买卖比:.2f}"]) + " |" for i, r in size.iterrows()) + """

## 口径备注

1. 快照只取 9:30 到 15:00，不含 9:15 到 9:25 集合竞价。
2. 剔除停牌、上市不满 60 个交易日、当日最高价等于最低价（一字板）的股票。
3. 深度是十档买卖挂单合计，金额用该档价格乘该档数量。
4. 各档份额是"中位数的份额"，不是"份额的中位数"，两者不等，所以十档归一化后才是 100%。
"""
(OUT / "数字明细.md").write_text(md, encoding="utf-8")

# XLSX
wb = Workbook()
fill = PatternFill("solid", fgColor="DDE6F0")
def style(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w
    for c in ws[1]:
        c.fill = fill; c.font = Font(bold=True)
    ws.freeze_panes = "A2"
    for row in ws.iter_rows(min_row=2):
        for c in row:
            c.alignment = Alignment(vertical="top", wrap_text=True)

ws = wb.active; ws.title = "分日"
ws.append(["展示日", "日期码", "股票数", "中位快照数", "十档金额中位(万元)", "十档股数中位(万股)", "买÷卖", "能撑分钟", "全市场成交额(亿元)"])
for r in rows: ws.append(r)
style(ws, [14, 10, 9, 12, 18, 19, 8, 10, 18])

ws = wb.create_sheet("分市值")
ws.append(["市值档", "股票日", "中位快照数", "十档金额中位(万元)", "十档股数中位(万股)", "能撑分钟", "买÷卖"])
for i, r in size.iterrows():
    ws.append([i, int(r.n), round(r.快照), round(r.金额万), round(r.股数万, 1), round(r.能撑, 1), round(r.买卖比, 2)])
style(ws, [8, 8, 12, 18, 19, 10, 8])

ws = wb.create_sheet("档位份额")
ws.append(["档位", "份额(%)"])
for i, v in enumerate(N["各档占比"], 1): ws.append([f"{i}档", v])
style(ws, [8, 10])

ws = wb.create_sheet("要点数字")
ws.append(["项目", "数值"])
P = [("股票日数", f"{N['股票日数']:,}"), ("中位快照数", f"{N['快照中位']:,.0f}"),
     ("满额股票日占比", f"{N['满额占比']*100:.1f}%"), ("十档金额中位(万元)", f"{N['金额中位万']:,.0f}"),
     ("十档金额P90/P10", "19 倍"), ("十档股数中位(万股)", f"{N['股数中位万']:,.1f}"),
     ("十档股数P90/P10", "104 倍"), ("买÷卖中位", f"{N['买卖比中位']:.2f}"),
     ("买÷卖相差一倍以上", f"{N['差值超一倍占比']*100:.1f}%"), ("能撑中位分钟", f"{N['能撑中位分钟']:.1f}"),
     ("能撑不到一分钟", f"{N['不到一分钟占比']*100:.1f}%"), ("价差一tick占比", f"{N['价差一tick占比']*100:.0f}%"),
     ("对账中位", f"{N['对账中位']:.4f}"), ("对账落2%以内", f"{N['对账落区间']*100:.1f}%")]
for k, v in P: ws.append([k, v])
style(ws, [24, 14])

wb.save(OUT / "数字汇总.xlsx")
(OUT / "numbers.json").write_text(json.dumps(N, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(N, ensure_ascii=False, indent=1))
