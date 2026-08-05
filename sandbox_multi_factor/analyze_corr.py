# -*- coding: utf-8 -*-
"""多日相关矩阵汇总 + 基于真实数据的 y 精简清单生成。

读取 mf_sandbox 输出文件（探索数据），
1. Fisher z 变换平均多日相关矩阵
2. 输出平均相关矩阵与高相关对
3. 对 5 套市场因子模型，按"同源变体规则 + 相关阈值规则"生成 y 保留/剔除清单
"""
import re
import sys
import numpy as np

NAMES = [
    "active_buy_volume_ratio",   # 0  主买占比
    "order_gap_signed_vw",       # 1  订单编号差 signed
    "observable_ratio_level",    # 2  可观测挂单占比
    "book_imbalance10_level",    # 3  10档不平衡
    "observable_ratio_innovation",  # 4
    "book_imbalance10_innovation",  # 5
    "spread_bps",                # 6
    "near3_depth_share",         # 7
    "microprice_pressure_bps",   # 8
    "order_gap_magnitude",       # 9
    "large_trade_direction_v2",  # 10
    "price_log_return_3s",       # 11
    "log_volume_3s",             # 12
    "trade_arrival_clustering",  # 13
]

# 同源变体关系（子 -> 母）
KIN = {
    10: 0,   # large_trade_direction -> active_buy_volume_ratio
    13: 12,  # trade_arrival_clustering -> log_volume_3s
    4: 2,    # observable_innovation -> observable_level
    5: 3,    # imbalance_innovation -> imbalance_level
    9: 1,    # order_gap_magnitude -> order_gap_signed
}

# 5 套模型
MODELS = {
    "T1": [0, 12, 2],
    "T2": [1, 9, 6],
    "T3": [3, 7, 13],
    "F1": [0, 12, 13, 2, 1],
    "F2": [0, 12, 2, 6, 3],
}

def parse_corr_matrix(path):
    """从探索输出文件解析 14×14 相关矩阵。"""
    text = open(path).read()
    blocks = re.split(r"=== date (\d+) ===", text)
    out = {}
    for i in range(1, len(blocks), 2):
        date = blocks[i]
        body = blocks[i + 1]
        m = re.search(r"\[14×14 相关矩阵\](.*?)\n\n\[", body, re.S)
        if not m:
            continue
        mat_text = m.group(1)
        rows = []
        for line in mat_text.strip().split("\n")[2:]:  # 跳过表头两行
            parts = line.split()
            if len(parts) < 15:
                continue
            try:
                row = [float(x) for x in parts[1:15]]
            except ValueError:
                continue
            rows.append(row)
        if len(rows) == 14:
            out[date] = np.array(rows)
    return out

def fisher_mean(mats):
    """Fisher z 平均。"""
    zsum = np.zeros((14, 14))
    cnt = np.zeros((14, 14))
    for m in mats.values():
        valid = np.isfinite(m)
        z = np.arctanh(np.clip(m, -0.999999, 0.999999))
        zsum[valid] += z[valid]
        cnt[valid] += 1
    mean = np.tanh(zsum / np.maximum(cnt, 1))
    mean[~np.isfinite(mean)] = np.nan
    return mean, cnt

def main(paths):
    all_mats = {}
    for p in paths:
        all_mats.update(parse_corr_matrix(p))
    dates = sorted(all_mats.keys())
    print(f"汇总日期: {dates}")

    mean, cnt = fisher_mean(all_mats)
    np.set_printoptions(precision=3, suppress=True, linewidth=200)
    print("\n[平均相关矩阵 14×14]")
    print("        " + "".join(f"{j:>8}" for j in range(14)))
    for i in range(14):
        print(f"{NAMES[i][:24]:<24}" + "".join(f"{mean[i,j]:>8.3f}" for j in range(14)))

    print("\n[|r| >= 0.55 的高相关对（平均相关）]")
    pairs = []
    for i in range(14):
        for j in range(i + 1, 14):
            r = mean[i, j]
            if np.isfinite(r) and abs(r) >= 0.55:
                pairs.append((abs(r), i, j, r))
    pairs.sort(reverse=True)
    for _, i, j, r in pairs:
        tag = " [同源变体]" if j in KIN and KIN[j] == i or i in KIN and KIN[i] == j else ""
        print(f"  ({i:>2},{j:>2}) {NAMES[i]:<26} vs {NAMES[j]:<26} r={r:+.3f}{tag}")

    print("\n[各模型 y 精简清单]（阈值 |r|>=0.60）")
    TH = 0.60
    for model, factors in MODELS.items():
        keep = []
        drop = []
        for y in range(14):
            # 规则1: 与因子集任一成员市场均值 |r| >= TH → 剔除
            if any(np.isfinite(mean[y, f]) and abs(mean[y, f]) >= TH for f in factors):
                drop.append(y)
                continue
            # 规则2: y 是因子集成员的同源变体 → 剔除
            if y in KIN and KIN[y] in factors:
                drop.append(y)
                continue
            keep.append(y)
        print(f"\n[{model}] 因子={[NAMES[f] for f in factors]}")
        print(f"  保留({len(keep)}): {[f'{NAMES[k]}({k})' for k in keep]}")
        print(f"  剔除({len(drop)}): {[f'{NAMES[k]}({k})' for k in drop]}")

if __name__ == "__main__":
    main(sys.argv[1:])
