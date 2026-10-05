"""生成 colblk 清理第二期的计划书（XLSX）。

数据来源都是现成的，不做磁盘扫描：
- 上一轮实删记录：reports/okay_database/colblk_delete_list.json + colblk_cleanup_state.json
- 各 store 当前剩余文件构成：/tmp/plan_store.json（本轮 stat 出来的，只读元数据）
- 目录当前占用：/home/chenzongwei/hdd_chenzongwei_size_ranking.txt（上一轮排查产物）
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import openpyxl
from openpyxl.styles import Font, PatternFill

REPO = Path("/home/chenzongwei/rust_pyfunc")
OUT = REPO / "reports/okay_database/colblk清理第二期计划-20261005.xlsx"

T = 1024 ** 4
G = 1024 ** 3

DELETABLE = [
    "日内CAPM升级版-多因子模型", "urgency_v1", "为人画像", "蹭热点达人", "hot_stock_pool_v1_fix",
    "hm95", "切换时刻贡献", "hot_stock_pool_v2", "hm100", "柜员游戏退火版挂单失衡修复",
    "柜员游戏三层01", "柜员游戏退火版挂单失衡", "hm108", "柜员游戏01", "cross_yhyb",
    "可观测挂单协调骤降", "yupei_dist", "pair_exp_A_4x50", "pair_exp_B_6x50", "pair_exp_C_7x50",
    "pair_exp_D_8x50", "pair_exp_E_4x100", "pair_exp_F_8x25", "long_order_v1",
]
KEEP = {
    "个体挂单比例0703": "独立项目，有自己的 tail_v4 入选名单",
    "个股挂单比例_成果重做": "独立项目，不属于 hm94~hm110",
    "peaks_cs": "银河落九天项目，另有研究脚本在用",
    "peaks_regime": "银河落九天项目",
    "distill0710": "归属不在 hm94~hm110，601 个因子先留着",
    "distill_tick0711": "归属不在 hm94~hm110，480 个因子先留着",
    "hm54hhi": "hm54 规范化变体的来源",
    "hm54": "因子库里 hm54 只有少量因子有数据，靠它补算",
}
SERIES = {  # store → hm 系列（来自 20261005 审阅名单）
    "日内CAPM升级版-多因子模型": "hm110", "urgency_v1": "hm98", "为人画像": "hm96",
    "蹭热点达人": "hm106", "hot_stock_pool_v1_fix": "hm104+hm105", "hm95": "hm95",
    "切换时刻贡献": "hm95（旧库）", "hot_stock_pool_v2": "hm105", "hm100": "hm100",
    "柜员游戏退火版挂单失衡修复": "hm102", "柜员游戏三层01": "hm103",
    "柜员游戏退火版挂单失衡": "hm102", "hm108": "hm108", "柜员游戏01": "hm97",
    "cross_yhyb": "hm107", "可观测挂单协调骤降": "hm101", "yupei_dist": "hm109",
    "pair_exp_A_4x50": "hm108 配对实验", "pair_exp_B_6x50": "hm108 配对实验",
    "pair_exp_C_7x50": "hm108 配对实验", "pair_exp_D_8x50": "hm108 配对实验",
    "pair_exp_E_4x100": "hm108 配对实验", "pair_exp_F_8x25": "hm108 配对实验",
    "long_order_v1": "hm99",
}


def ranking() -> dict:
    out = {}
    for line in Path("/home/chenzongwei/hdd_chenzongwei_size_ranking.txt").read_text().splitlines():
        m = re.match(r"\s*(\d+)\s+([\d.]+)\s*(B|GB|TB)\s+(\S+)\s+(.+)", line)
        if m:
            _, val, unit, _, name = m.groups()
            out[name.strip()] = float(val) * {"B": 1, "GB": G, "TB": T}[unit]
    return out


def main() -> int:
    plan = {r["store"]: r for r in json.loads(Path("/tmp/plan_store.json").read_text())}
    deleted = json.loads((REPO / "reports/okay_database/colblk_delete_list.json").read_text())
    clean = json.loads((REPO / "reports/okay_database/colblk_cleanup_state.json").read_text())
    rank = ranking()

    wb = openpyxl.Workbook()
    head = Font(bold=True)
    warn = PatternFill("solid", fgColor="FFF2CC")

    ws = wb.active
    ws.title = "说明"
    for row in [
        ["colblk 清理第二期计划", None],
        ["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        [None, None],
        ["第一期删了什么", "217 个 factors.colblk，合计 11.42 TB，全部删净（已复核，目标 store 里只剩 9 个入选原件 colblk，0.28 GB）"],
        ["为什么看起来没删干净", "20261005 审阅名单把 factors.proj 明确写成「不属于删除对象」，第一期就严格只删了 colblk。"
                                 "而每个 colblk 旁边都有一份同量级的 proj，所以 store 目录体积只掉了一半，目录还在、还很大"],
        ["第二期删什么", "把这 24 个 store 里除入选原件（*_accepted）和 factor_groups.json 之外的东西全清掉"],
        ["第二期能腾多少", "proj 10.91 TB + proj.delta 0.51 TB + 零碎 = 约 11.42 TB（按文件大小合计）；"
                        "按目录实际占用算是 12.43 TB"],
        ["保留不动", "8 个暂不可删 store，合计 1.30 TB；另有 9 份入选原件 0.28 GB"],
        ["待确认", "隐藏归档 .hmokay4_archive_hm97_before_nan_promotion/factor_store_柜员游戏01，505.71 GB"],
        ["注意", "factor_store_hm29_okay_database 是补算任务正在写的临时库，不能动（写完会被任务自己删掉）"],
        ["删除方式", "ionice -c2 -n7 nice -n 19 rm：尽最大努力的最低 IO 优先级，尽量不干扰同时在跑的补算"],
        ["耗时估计", "第一期 11.42 TB 用了约 37 分钟（01:12:43→01:49:21），其中 178 个小文件只用 21 秒，"
                  "剩下 39 个含 30~280 GB 大文件用了 28 分钟，还碰上过一个 34 GB 文件卡了 28 分钟。"
                  "第二期文件构成类似，估计 1~3 小时，遇到卡顿会更久"],
    ]:
        ws.append(row)
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 110
    for r in ws.iter_rows(min_row=1, max_row=ws.max_row, max_col=1):
        r[0].font = head
    for r in ws.iter_rows(min_row=4, max_row=ws.max_row):
        r[1].alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="top")

    ws2 = wb.create_sheet("1-第二期计划删除")
    ws2.append(["store 目录", "归属系列", "factors.proj(GB)", "proj.delta(GB)", "idx/bin/元数据(GB)",
                "本期可释放(GB)", "目录当前占用(GB)", "保留的入选原件(GB)"])
    tot = 0.0
    for name in sorted(DELETABLE, key=lambda x: -plan.get(x, {}).get("proj", 0)):
        r = plan[name]
        free = (r["proj"] + r["delta"] + r["idx"] + r["bin"] + r["meta"] + r["other"]) / G
        tot += free
        ws2.append([f"factor_store_{name}", SERIES.get(name, "-"), round(r["proj"] / G, 2),
                    round(r["delta"] / G, 2), round((r["idx"] + r["bin"] + r["meta"]) / G, 3),
                    round(free, 2), round(rank.get("factor_store_" + name, 0) / G, 2),
                    round(r["accepted"] / G, 3)])
    ws2.append(["合计", f"{len(DELETABLE)} 个 store", "", "", "", round(tot, 2), "", ""])
    ws2[ws2.max_row][5].font = head

    ws3 = wb.create_sheet("2-保留清单")
    ws3.append(["store 目录", "当前占用(GB)", "保留理由"])
    for name, why in KEEP.items():
        ws3.append([f"factor_store_{name}", round(rank.get("factor_store_" + name, 0) / G, 2), why])
    ws3.append(["9 份入选原件（*_accepted 子库）", 0.28, "入选名单的原件，体积很小，留着零风险"])
    ws3.append([".hmokay4_archive_hm97_before_nan_promotion", round(505.71, 2),
                "hm97 晋升正式版前有意留的旧副本，colblk 252.89 GB + proj 252.82 GB，待你确认"])
    ws3.append(["factor_store_hm29_okay_database", round(rank.get("factor_store_hm29_okay_database", 0) / G, 2),
                "补算任务正在用的临时库，写完由任务自己删"])

    ws4 = wb.create_sheet("3-第一期已删记录")
    ws4.append(["序号", "store 目录", "colblk 相对路径", "大小(GB)", "归属系列", "状态"])
    state = {i["path"]: i for i in clean["items"]}
    for i, d in enumerate(deleted, 1):
        st = state.get(d["path"], {})
        ws4.append([i, f"factor_store_{d['store']}", d["rel"], round(d["size"] / G, 3),
                    d.get("series") or "-", st.get("status", "-")])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT)
    print(f"计划书已生成：{OUT}")
    print(f"第二期计划释放 {tot:.2f} GB = {tot / 1024:.2f} TB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
