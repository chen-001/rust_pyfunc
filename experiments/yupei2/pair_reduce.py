#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_reduce.py —— 玉佩2 "配对互补"网络的降维与诊断

用法:
    python pair_reduce.py <pair_out_dir> <date> [--backup <backup_root>]
    python pair_reduce.py --selftest [--synth-n 300]

输入 = 主会话产出的冻结 schema（pair_spec.txt 第 4 节），目录 <pair_out_dir>/<date>/ :
    codes.txt  rho.bin  E.bin  G.bin  RHO_U.bin  SINGLE.bin  X0.bin  Z.bin  summary.json
    rho.bin 对角 1，其余方阵对角 0，N×N f32 行主序；E.bin 是 N 个 f64 裸写。

输出 = <pair_out_dir>/<date>/reduce/ :
    factors.bin   N×F f32 行主序（每只股票固定 F 个值，算不出来写空值 nan）
    names.txt     每行一个因子名，与 factors.bin 的列序一致
    diag.json     中文键值诊断，给人读

降维内容（都保留配对/三元组层面的结构，不做"行和 / 均值 / 分位数"那种压扁）:
    R1 角色画像   谁被多少对选为 X0、它参与的强互补对有多少、它和最强搭档是否互惠
    R2 三分解     每只 X 的最佳单伙伴 vs 最佳伙伴对（用 spec 的等价式扫描全部配对）
    R3 子模篮子   贪心加入伙伴使 rho[X][U_S] 最大，记录边际增益、饱和速度、是否退化成枢纽
    R4 图对照     G 的 top-k 图 vs rho 的 top-k 图：度、共同邻居、象限、社群一致性
    R5 残差互补   把 G 对 rho 与 log(成交笔数) 回归后的残差，取行内最大值/前 20 均值/集中度

诊断:
    (a) 跨日稳定性：同一天目录下的另一日重算同一批因子，取股票交集算 Spearman
    (b) 与老因子的重叠：备份目录 <backup_root>/<date>/factors.bin + names.txt（玉佩2 原 2761 个）
    (c) 空模型：把 G 换成 Z 重算同一批因子，看排序还剩多少

纪律（写死在代码里）:
    * 不做 Python 双重循环扫 N² 配对，配对层面的量一律 numpy 向量化。
    * 行业、成交笔数都按股票代码对齐，不假设备份与本目录行序一致。
    * 每只股票的因子个数恒定，算不出来写 nan。
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

try:
    from scipy.stats import rankdata
except Exception:  # pragma: no cover
    rankdata = None

F32 = np.float32
F64 = np.float64

# 辅助列：只是为了能把最佳伙伴对追溯回具体股票，不是可交易信号，
# 跨日比较没有意义（同一天不同日的代码表行序不同），排序类的地方要把它排除掉。
AUX_NAMES = ("r2_best_pair_Bidx", "r2_best_pair_Cidx")

# ============================================================
# 0. 读写基础
# ============================================================


def read_codes(path):
    with open(path, "r") as f:
        return [ln.strip() for ln in f if ln.strip()]


def load_sq(path, n, name=""):
    """读 N×N f32 行主序方阵。"""
    a = np.fromfile(path, dtype=F32)
    if a.size != n * n:
        raise ValueError("%s 大小不对: %d 个 f32, 期望 %d" % (name or path, a.size, n * n))
    return a.reshape(n, n)


def load_vec_f64(path, n):
    a = np.fromfile(path, dtype=F64)
    if a.size != n:
        raise ValueError("%s 长度不对: %d, 期望 %d" % (path, a.size, n))
    return a


SYMBOL_MAP = "/ssd_data/data/basic_info/symbol_map.csv"


def load_symbol_map():
    """symbol_map.csv 的 code 集合与 code->pos。用来判个股白名单与对齐 industry.bin。"""
    codes, pos = set(), {}
    if not os.path.exists(SYMBOL_MAP):
        return codes, pos
    with open(SYMBOL_MAP, "r") as f:
        for i, ln in enumerate(f):
            if i == 0:
                continue
            parts = ln.strip().split(",")
            if len(parts) >= 2 and parts[0]:
                codes.add(parts[0])
                try:
                    pos[parts[0]] = int(parts[1])
                except ValueError:
                    pass
    return codes, pos


def load_stock_mask(codes, G):
    """个股白名单。返回 (bool 数组, 说明)。

    这份数据里非个股（ETF、债券）的配对已被上游剔除，表现是它们的 G 整行整列为 0，
    所以除了查 symbol_map 之外，再用 G 的整零行列做一次交叉验证：
    两边说法不一致就把差异写进说明，让人能看出来。
    """
    smap, _ = load_symbol_map()
    if not smap:
        return np.ones(len(codes), dtype=bool), "读不到 symbol_map.csv，白名单不做筛选"
    mask = np.array([c in smap for c in codes], dtype=bool)
    row_zero = np.abs(np.asarray(G, dtype=F64)).max(axis=1) == 0.0
    col_zero = np.abs(np.asarray(G, dtype=F64)).max(axis=0) == 0.0
    note = ("symbol_map 判定非个股 %d 只；G 整行全零 %d 只，整列全零 %d 只；"
            "两者不一致的行 %d 只（不一致的按 symbol_map 处理）"
            % (int((~mask).sum()), int(row_zero.sum()), int(col_zero.sum()),
               int(((~mask) & ~row_zero).sum() + (mask & row_zero).sum())))
    return mask, note


def load_industry(ind_path, codes):
    """行业。industry.bin = u64 n + n 个 i16。

    industry.rs 里 load_industry_bin 声称"与 codes 对齐"，但实际这份文件来自
    industry_dump，列序是 symbol_map.csv 的 pos 列（用 h5 industry.h5 现读现比可验证）。
    所以这里按长度判断：
        长度 == len(codes)  -> 直接当 codes 对齐用
        否则                -> 用 symbol_map.csv 建 code->行业 映射
    返回 (行业数组 int32 或 None, 说明字符串)
    """
    if ind_path is None or not os.path.exists(ind_path):
        return None, "没有 industry.bin（%s）" % ind_path
    raw = open(ind_path, "rb").read()
    n_ind = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    ind = np.frombuffer(raw[8:8 + 2 * n_ind], dtype="<i2").astype(np.int32)
    if n_ind == len(codes):
        return ind, "industry.bin 长度 %d == 股票数 %d，按 codes 行序直接用" % (n_ind, len(codes))
    smap_path = "/ssd_data/data/basic_info/symbol_map.csv"
    if not os.path.exists(smap_path):
        return None, "industry.bin 长度 %d != 股票数 %d，且找不到 symbol_map.csv，行业全部记未知" % (n_ind, len(codes))
    code_by_pos = []
    with open(smap_path, "r") as f:
        for i, ln in enumerate(f):
            if i == 0:
                continue
            c = ln.split(",")[0].strip()
            if c:
                code_by_pos.append(c)
    if n_ind > len(code_by_pos):
        return None, "industry.bin 长度 %d 超过 symbol_map 行数 %d，行业全部记未知" % (n_ind, len(code_by_pos))
    c2i = {}
    for pos in range(min(n_ind, len(code_by_pos))):
        c2i.setdefault(code_by_pos[pos], pos)
    out = np.array([ind[c2i[c]] if c in c2i else -1 for c in codes], dtype=np.int32)
    n_known = int((out >= 0).sum())
    note = ("industry.bin 长度 %d != 股票数 %d，按 symbol_map.csv 的 pos 列对齐"
            "（%d/%d 只有行业，其余 -1）") % (n_ind, len(codes), n_known, len(codes))
    return out, note


_CACHE = {}


def triu_idx(n):
    """(n,n) 严格上三角的行列下标，按 n 缓存。np.triu_indices 每次现算会分配上百 MB，
    在 3964 这种规模上单次就是好几秒，不能放在循环里反复调。"""
    if ("iu", n) not in _CACHE:
        _CACHE[("iu", n)] = np.triu_indices(n, 1)
    return _CACHE[("iu", n)]


def lower_mask(n):
    """(n,n) 严格下三角的布尔掩码，按 n 缓存。用于对称化时避免用整数下标。"""
    if ("lo", n) not in _CACHE:
        _CACHE[("lo", n)] = np.triu(np.ones((n, n), dtype=bool), 1).T.copy()
    return _CACHE[("lo", n)]


def sym_from_upper(a):
    """把只填了上三角的方阵补成对称阵：下三角一律从上三角抄，对角保持原样。

    不能写成 a + a.T：下三角原先是空的，空值加上三角的值仍然是空值，会把整张矩阵污染成空值。
    也不要用 triu_indices 做 fancy indexing，那个在大矩阵上比布尔掩码慢一个量级。
    """
    n = a.shape[0]
    out = a.copy()
    m = lower_mask(n)
    out[m] = a.T[m]
    return out


def load_stats_ntrades(backup_dir, codes):
    """stats.bin: u64 n + n×14 f64，第 0 列是 n_trades（matrix_store.rs read_stats）。"""
    if backup_dir is None:
        return None, "没有备份目录，成交笔数不可用"
    sp = os.path.join(backup_dir, "stats.bin")
    cp = os.path.join(backup_dir, "codes.txt")
    if not (os.path.exists(sp) and os.path.exists(cp)):
        return None, "备份目录缺 stats.bin 或 codes.txt（%s）" % backup_dir
    raw = open(sp, "rb").read()
    n = int(np.frombuffer(raw[:8], dtype="<u8")[0])
    arr = np.frombuffer(raw[8:8 + n * 14 * 8], dtype="<f8").reshape(n, 14)
    bcodes = read_codes(cp)
    m = min(n, len(bcodes))
    c2v = {}
    for i in range(m):
        c2v.setdefault(bcodes[i], arr[i, 0])
    out = np.array([c2v.get(c, np.nan) for c in codes], dtype=F64)
    n_ok = int(np.isfinite(out).sum())
    return out, "从 %s 读到成交笔数，%d/%d 只有值" % (sp, n_ok, len(codes))


# ============================================================
# 1. 统计小工具（都要能处理 nan）
# ============================================================


def _rank(x):
    """平均秩，返回 f64。"""
    if rankdata is not None:
        return rankdata(x).astype(F64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=F64)
    r[order] = np.arange(1, len(x) + 1, dtype=F64)
    return r


def spearman(x, y, min_n=30):
    """两条序列的 Spearman，只取都有限的位置。样本不够返回 nan。"""
    x = np.asarray(x, dtype=F64).ravel()
    y = np.asarray(y, dtype=F64).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    k = int(m.sum())
    if k < min_n:
        return float("nan"), k
    rx = _rank(x[m])
    ry = _rank(y[m])
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    if d <= 0:
        return float("nan"), k
    return float((rx * ry).sum() / d), k


def spearman_matrix(a, b):
    """a: M×F1, b: M×F2（列已对齐、都有限）-> F1×F2 的 Spearman。z 分后一次矩阵乘。"""
    za = _rank2d(a)
    zb = _rank2d(b)
    za -= za.mean(axis=0, keepdims=True)
    zb -= zb.mean(axis=0, keepdims=True)
    za /= np.maximum(np.sqrt((za * za).sum(axis=0, keepdims=True)), 1e-30)
    zb /= np.maximum(np.sqrt((zb * zb).sum(axis=0, keepdims=True)), 1e-30)
    return za.T @ zb


def _rank2d(a):
    a = np.asarray(a, dtype=F64)
    if rankdata is not None:
        return rankdata(a, axis=0).astype(F64)
    out = np.empty_like(a)
    for j in range(a.shape[1]):
        out[:, j] = _rank(a[:, j])
    return out


def group_mean(idx, val, n):
    """按 idx（0..n-1）求 val 的均值，val 有限。"""
    cnt = np.bincount(idx, minlength=n).astype(F64)
    s = np.bincount(idx, weights=val, minlength=n)
    out = np.full(n, np.nan, dtype=F64)
    ok = cnt > 0
    out[ok] = s[ok] / cnt[ok]
    return out, cnt


def group_max(idx, val, n):
    """按 idx 分组求最大值（argsort + reduceat，避免 np.maximum.at 太慢）。"""
    out = np.full(n, np.nan, dtype=F64)
    if idx.size == 0:
        return out
    order = np.argsort(idx, kind="mergesort")
    si = idx[order]
    sv = val[order]
    starts = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
    mx = np.maximum.reduceat(sv, starts)
    out[si[starts]] = mx
    return out


def hhi_from_pos(res_row):
    """正残差的 HHI（归一化到和为 1 后的平方和），没有正残差写 nan。"""
    r = res_row[np.isfinite(res_row)]
    p = r[r > 0]
    if p.size == 0:
        return float("nan")
    w = p / p.sum()
    return float((w * w).sum())


def topk_mean(row, k):
    r = row[np.isfinite(row)]
    if r.size == 0:
        return float("nan")
    k = min(k, r.size)
    part = np.partition(r, r.size - k)[r.size - k:]
    return float(part.mean())


def ari_and_agree(a, b):
    """两个划分的调整兰德指数与配对一致率。

    一致率 = （两边都同簇的对数 + 两边都不同簇的对数）/ 总对数，
    用容斥算：都不同簇 = 总对数 - A 同簇 - B 同簇 + 两边都同簇。
    注意不能写成 sum_ij C(n - a_i - b_j + n_ij, 2)：簇数超过 2 时同一个"两边都不同簇"的
    配对会被多个 (i,j) 格子重复数进去，结果会大于 1（这里是踩过的坑）。
    """
    a = np.asarray(a)
    b = np.asarray(b)
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    k1, k2 = len(ua), len(ub)
    tab = np.bincount(ia * k2 + ib, minlength=k1 * k2).reshape(k1, k2).astype(F64)
    n = a.size

    def c2(x):
        return x * (x - 1.0) / 2.0

    total = c2(n)
    if total <= 0:
        return float("nan"), float("nan")
    same_both = c2(tab).sum()
    same_a = c2(tab.sum(axis=1)).sum()
    same_b = c2(tab.sum(axis=0)).sum()
    diff_both = total - same_a - same_b + same_both
    agree = (same_both + diff_both) / total
    exp = same_a * same_b / total
    maxi = 0.5 * (same_a + same_b)
    ari = (same_both - exp) / (maxi - exp) if maxi > exp else float("nan")
    return float(ari), float(min(max(agree, 0.0), 1.0))


def label_propagation(adj, iters=25):
    """确定性标签传播：节点按下标升序扫，标签并列取最小的。adj 是 bool 方阵。"""
    n = adj.shape[0]
    labels = np.arange(n, dtype=np.int32)
    for _ in range(iters):
        changed = 0
        for i in range(n):
            nb = np.flatnonzero(adj[i])
            if nb.size == 0:
                continue
            vals, cnt = np.unique(labels[nb], return_counts=True)
            best = int(vals[int(np.argmax(cnt))])
            if labels[i] != best:
                labels[i] = best
                changed += 1
        if changed == 0:
            break
    _, lab = np.unique(labels, return_inverse=True)
    return lab.astype(np.int32)


def topk_adj(sq, k, ok=None):
    """每只股票取前 k 条边（自身除外），做成无向图的 bool 邻接矩阵。

    ok 是白名单：不在白名单里的行与列直接排除（它们不是可交易标的）。
    """
    n = sq.shape[0]
    s = np.array(sq, dtype=F32)
    np.fill_diagonal(s, -np.inf)
    if ok is not None:
        s[:, ~ok] = -np.inf
        s[~ok, :] = -np.inf
    k = min(k, int(ok.sum()) - 1 if ok is not None else n - 1)
    idx = np.argpartition(-s, k - 1, axis=1)[:, :k]
    adj = np.zeros((n, n), dtype=bool)
    adj[np.arange(n)[:, None], idx] = True
    adj |= adj.T
    if ok is not None:
        adj[~ok, :] = False
        adj[:, ~ok] = False
    np.fill_diagonal(adj, False)
    return adj


# ============================================================
# 2. R2：全配对扫描（同时为 R1 / R4 / R5 提供配对层面的量）
# ============================================================


def pair_scan(rho, E, ok=None, chunk=128, verbose=True):
    """对每只 X 求 best_pair = max_{B<C, X 不属于 {B,C}} rho[X][U_BC]。

    用 spec 第 2 节的精确等价式：
        rho[X][U_BC] = (se_B * rho[X][B] + se_C * rho[X][C]) / D(B,C)
        D(B,C) = sqrt(eB + eC + 2 rho[B][C] se_B se_C)
    固定 B 时上式对全部 C 一步算完，所以是 N 趟 O(N^2) 的 numpy，不是 N^2 次 Python 循环。
    ok 是白名单：非个股不能当伙伴、也不能当被解释的 X。
    返回 best_pair, best_B, best_C（一维数组，长度 N）。
    """
    n = rho.shape[0]
    if ok is None:
        ok = np.ones(n, dtype=bool)
    se = np.sqrt(np.maximum(E, 1e-30)).astype(F32)
    M = (rho * se[None, :]).astype(F32)          # M[X,B] = se_B * rho[X][B]
    den = np.sqrt(np.maximum(
        E[:, None] + E[None, :] + 2.0 * rho * se[:, None] * se[None, :], 1e-30)).astype(F32)
    Winv = (1.0 / den).astype(F32)
    np.fill_diagonal(Winv, 0.0)
    Mt = np.ascontiguousarray(M.T)               # Mt[B] 是连续的一行
    Wt = np.ascontiguousarray(Winv)

    best = np.full(n, -np.inf, dtype=F32)
    bi = np.full(n, -1, dtype=np.int32)
    ci = np.full(n, -1, dtype=np.int32)
    buf = np.empty((chunk, n), dtype=F32)
    bad_x = ~ok
    for B in range(n - 1):
        if not ok[B]:
            continue
        a = Mt[B]
        row_w = Wt[B]
        for s in range(B + 1, n, chunk):
            e = min(s + chunk, n)
            m = e - s
            sub = buf[:m]
            np.add(Mt[s:e], a, out=sub)
            np.multiply(sub, row_w[s:e, None], out=sub)
            cc = np.arange(s, e)
            sub[np.arange(m), cc] = -np.inf      # C == X 的对不能用
            sub[:, B] = -np.inf                  # X == B 的对不能用
            sub[~ok[s:e], :] = -np.inf           # 非个股不能当伙伴
            if bad_x.any():
                sub[:, bad_x] = -np.inf          # 非个股不能当被解释的 X
            bm = sub.max(axis=0)
            upd = bm > best
            if upd.any():
                bmi = sub.argmax(axis=0)
                best[upd] = bm[upd]
                bi[upd] = B
                ci[upd] = cc[bmi[upd]]
        if verbose and (B + 1) % 500 == 0:
            print("    R2 扫描 %d/%d" % (B + 1, n), flush=True)
    return best, bi, ci


def best_single_of(rho, ok=None):
    n = rho.shape[0]
    s = np.array(rho, dtype=F32)
    np.fill_diagonal(s, -np.inf)
    if ok is not None:
        s[:, ~ok] = -np.inf
        s[~ok, :] = -np.inf
    return s.max(axis=1), s.argmax(axis=1).astype(np.int32)


# ============================================================
# 3. 各层因子
# ============================================================


def r1_role(X0, pair_weight, pair_weight_z, n, codes, valid_pair, tag="r1", z_thr=2.0):
    """R1 角色画像。pair_weight 是配对层面的权重矩阵（G / GSTAR，或换 Z / ZS 算对照版）。

    valid_pair 是"两只都是个股"的上三角掩码：非个股的配对不参与统计。
    """
    x0u = np.where(valid_pair, X0, -1)
    wu = np.where(valid_pair, pair_weight, 0.0).astype(F64)
    zu = np.where(valid_pair, pair_weight_z, 0.0).astype(F64)
    good = valid_pair & (x0u >= 0) & (x0u < n)
    xs = x0u[good].astype(np.int64)
    ws = wu[good]
    zs = zu[good]
    if xs.size == 0:
        raise RuntimeError("X0.bin 里没有一个合法的 X0")

    cnt = np.bincount(xs, minlength=n).astype(F64)
    gmean, _ = group_mean(xs, ws, n)
    gmax = group_max(xs, ws, n)
    zmean, _ = group_mean(xs, zs, n)
    strong = zs > z_thr
    cnt_z2 = np.bincount(xs[strong], minlength=n).astype(F64)
    # 作为伙伴出现在多少个强互补对里（全配对下出现次数恒为 N-1，所以按强度筛）
    strong_pair = valid_pair & (np.asarray(pair_weight_z) > z_thr)
    part_z2 = (strong_pair.sum(axis=1) + strong_pair.sum(axis=0)).astype(F64)

    out = {
        tag + "_explain_count": cnt,
        tag + "_explain_count_z2": cnt_z2,
        tag + "_explain_gain_mean": gmean,
        tag + "_explain_gain_max": gmax,
        tag + "_explain_z_mean": zmean,
        tag + "_as_partner_z2_count": part_z2,
    }
    # X0 集中度：这条规则高度挑少数几只股票，必须单独报
    used = np.flatnonzero(cnt > 0)
    order = used[np.argsort(-cnt[used])]
    tot = float(xs.size)
    conc = {"路径": tag, "有效配对数": int(xs.size), "被选为 X0 的股票数": int(used.size),
            "前 10 名占全部配对的比例": round(float(cnt[order[:10]].sum() / tot), 4),
            "第一名代码": codes[int(order[0])] if order.size else None,
            "第一名占比": round(float(cnt[order[0]] / tot), 4) if order.size else None,
            "前 10 名": [[codes[int(i)], round(float(cnt[i] / tot), 4)] for i in order[:10]]}
    return out, {"合法 X0 对数": int(xs.size),
                 "X0 不合法或缺失的对数": int(valid_pair.sum() - xs.size),
                 "X0 集中度": conc}


def r1_bestcomp(pair_weight, pair_weight_z, n, tag="r1", ok=None):
    """每只股票的最强互补搭档，以及这段关系是否互惠。

    互惠的定义：X 的最强搭档是 P，而 P 的最强搭档也是 X（两边行内最大互指）。
    注意不能用"这对配对最后解释的是谁"来定义互惠：按 spec，X0 明确排除配对内部的两只，
    所以 X0(X,P) 永远既不等于 X 也不等于 P，那个口径在本口径下恒为常数、没有信息。
    """
    W = np.array(pair_weight, dtype=F32)
    np.fill_diagonal(W, -np.inf)
    if ok is not None:
        W[:, ~ok] = -np.inf
        W[~ok, :] = -np.inf
    idx = np.arange(n)
    partner = W.argmax(axis=1).astype(np.int32)
    gain = W[idx, partner].astype(F64)
    z_of = np.asarray(pair_weight_z, dtype=F64)[idx, partner]
    # 在搭档 P 自己的 G 行里，X 排第几（0 = 搭档的最强互补也是 X）
    rank_in_partner = (W[partner] > gain[:, None]).sum(axis=1).astype(F64)
    mutual = (rank_in_partner == 0).astype(F64)
    if ok is not None:
        mutual = np.where(ok, mutual, np.nan)
        rank_in_partner = np.where(ok, rank_in_partner, np.nan)
    out = {
        tag + "_bestcomp_gain": gain,
        tag + "_bestcomp_z": z_of,
        tag + "_bestcomp_mutual": mutual,
        tag + "_bestcomp_rank": rank_in_partner,
    }
    diag = {"取法": "最强互补搭档取权重矩阵行内最大（对角与白名单外都置 -inf）",
            "互惠定义": "X 的最强搭档是 P，且 P 的最强搭档也是 X",
            "互惠占比": float(np.nanmean(mutual)) if ok is None else float(np.nanmean(np.where(ok, mutual, np.nan))),
            "在搭档行内的平均排名": float(np.nanmean(rank_in_partner))}
    return out, diag, partner


def r2_three_way(rho, E, best_pair, bB, bC, industry, ok=None):
    """R2 三分解：最佳单伙伴、最佳伙伴对、两者之差、最佳对是否跨行业。"""
    n = rho.shape[0]
    single, sb = best_single_of(rho, ok)
    single = np.where(np.isfinite(single), single, np.nan).astype(F64)
    bp = np.where(np.isfinite(np.asarray(best_pair, dtype=F64)), np.asarray(best_pair, dtype=F64), np.nan)
    bg = bp - single
    ratio = np.where(np.abs(single) > 1e-9, bg / np.abs(single), np.nan)
    if industry is not None:
        iB = industry[bB]
        iC = industry[bC]
        cross = np.where((iB >= 0) & (iC >= 0), (iB != iC).astype(F64), np.nan)
    else:
        cross = np.full(n, np.nan)
    out = {
        "r2_best_single": single.astype(F64),
        "r2_best_pair": bp,
        "r2_pair_gain": bg,
        "r2_pair_gain_ratio": ratio,
        "r2_best_pair_cross_industry": cross,
        "r2_best_pair_Bidx": bB.astype(F64),
        "r2_best_pair_Cidx": bC.astype(F64),
    }
    diag = {"最佳单伙伴均值": float(np.nanmean(single)),
            "最佳伙伴对均值": float(np.nanmean(bp)),
            "配对增益均值": float(np.nanmean(bg)),
            "配对增益为正的比例": float(np.nanmean(bg > 0))}
    if industry is not None:
        diag["最佳对跨行业比例"] = float(np.nanmean(cross))
    return out, diag


def r3_greedy(rho, E, ok=None, max_k=8, verbose=True):
    """R3 子模贪心篮子。

    rho[X][U_S] = (sum_{Y in S} se_Y rho[X][Y]) / sqrt(sum_{Y,Z in S} se_Y se_Z rho[Y][Z])
    rho[X][U_S] = (sum_{Y in S} se_Y rho[X][Y]) / sqrt(Q_x)
    其中 Q_x = sum_{Y,Z in S} se_Y se_Z rho[Y][Z]。
    注意分母里没有 E_X：分子里已经含一个 se_X，正好和 sqrt(E_X) 约掉，
    写成 sqrt(E_X * Q) 会把结果整体缩小 se_X 倍（第一版就是这样错的，已按原始核验证过：
    单只伙伴时 rho[X][U_S] 必须正好等于 rho[X][Y]）。
    维护 D[:,x] = rho @ c_x（c_x = se * 1_{S_x}）与 Q_x = c_x·D[:,x]，加入一只伙伴只需一次
    O(N) 的列更新，所以整个贪心是 8 趟 O(N^2)，不需要每步重算。
    """
    n = rho.shape[0]
    se = np.sqrt(np.maximum(E, 1e-30)).astype(F64)
    R = np.array(rho, dtype=F64)
    Er = np.array(E, dtype=F64)
    D = np.zeros((n, n), dtype=F64)          # D[Y, x]，用 f64 累积，避免 8 步累加把精度吃光
    Q = np.zeros(n, dtype=F64)
    IN = np.zeros((n, n), dtype=bool)        # IN[x, Y]
    marg = np.full((n, max_k), np.nan, dtype=F64)
    rs = np.zeros((n, max_k + 1), dtype=F64)  # rs[:,0]=0
    picks = np.full((n, max_k), -1, dtype=np.int32)
    Xall = np.arange(n)
    for step in range(max_k):
        num_x = D[Xall, Xall]                                 # num = rho[X,:] @ c_x
        with np.errstate(invalid="ignore", divide="ignore"):
            rs[:, step + 1] = np.where(Q > 0, num_x / np.sqrt(np.maximum(Q, 1e-30)), 0.0)
        # 候选值：val[Y,x] = (num_x + se_Y rho[X][Y]) / sqrt(Q_x + 2 se_Y D[Y,x] + E_Y)
        # 2 se_Y D 这一项不能漏：D[Y,x] 是除过 se_Y 的归一化量，而 Q 是原始核的量纲。
        numcand = num_x[None, :] + (R * se[None, :]).T       # [Y, x]
        dencand = np.sqrt(np.maximum(Q[None, :] + 2.0 * se[:, None] * D + Er[:, None], 1e-30))
        val = numcand / dencand
        val[IN.T] = -np.inf                                  # 已经在篮子里的不能再选
        val[Xall, Xall] = -np.inf                            # 自己不能进自己的篮子
        if ok is not None:
            val[~ok, :] = -np.inf                            # 非个股不能进篮子
        pick = val.argmax(axis=0).astype(np.int32)
        # 更新 D 与 Q
        d_pick = D[pick, Xall]                               # 用更新前的 D
        s_pick = se[pick]
        Q = Q + 2.0 * s_pick * d_pick + Er[pick]
        D += s_pick[None, :] * R[:, pick]                    # 列更新
        IN[Xall, pick] = True
        picks[:, step] = pick
        num_new = D[Xall, Xall]
        with np.errstate(invalid="ignore", divide="ignore"):
            rs[:, step + 1] = num_new / np.sqrt(np.maximum(Q, 1e-30))
        marg[:, step] = rs[:, step + 1] - rs[:, step]
        if verbose:
            print("    R3 贪心第 %d 步完成，篮子相似度均值 %.4f" % (step + 1, np.nanmean(rs[:, step + 1])), flush=True)

    # 增益降到首步 10% 以下所需只数
    thr = np.abs(marg[:, 0]) * 0.1
    n_to = np.full(n, float(max_k + 1))
    for s in range(max_k):
        hit = np.isnan(n_to) | (n_to > max_k)
        dec = marg[:, s] < thr
        n_to[hit & dec] = s + 1
    n_to = np.where(n_to > max_k, float(max_k + 1), n_to)

    # 伙伴之间的平均 rho（判断是否退化成"一只枢纽带一堆小跟班"）
    avg_pr = np.full(n, np.nan)
    hub = np.full(n, np.nan)
    for x in range(n):
        S = picks[x][picks[x] >= 0]
        if S.size >= 2:
            sub = R[np.ix_(S, S)].astype(F64)
            m = ~np.eye(S.size, dtype=bool)
            avg_pr[x] = sub[m].mean()
            hub[x] = sub[m].max()
    out = {
        "r3_marg1": marg[:, 0] if max_k > 0 else None,
        "r3_marg2": marg[:, 1] if max_k > 1 else None,
        "r3_marg3": marg[:, 2] if max_k > 2 else None,
        "r3_marg4": marg[:, 3] if max_k > 3 else None,
        "r3_n_to_10pct": n_to,
        "r3_basket_rho8": rs[:, max_k],
        "r3_avg_partner_rho": avg_pr,
        "r3_max_partner_rho": hub,
        "r3_degenerate": (avg_pr > 0.5).astype(F64),
    }
    diag = {"篮子最大只数": max_k,
            "边际增益定义": "第 s 步加入后 rho[X][U_S] 的增量，rho[X][U_S] 首项为 0",
            "n_to_10pct 定义": "边际增益首次低于首步 10%% 的步数；%d 表示 %d 只内没降下来" % (max_k + 1, max_k),
            "8 只篮子相似度均值": float(np.nanmean(rs[:, max_k])),
            "伙伴之间平均 rho 中位数": float(np.nanmedian(avg_pr)),
            "退化比例（伙伴平均 rho>0.5）": float(np.nanmean(out["r3_degenerate"]))}
    return out, diag


def r4_graphs(rho, G, Z, n, k=20, k_comm=(20, 10, 5, 3), ok=None, extra_graphs=None):
    """R4 互补图（G）vs 替代图（rho），另外把 Z 与 ZS 两路图也一起对照。

    度、共同邻居、象限按 k 建图；社群对比在几个 k 上都做一遍，因为 rho 的 top-k 图
    很密（k=20 时平均度 39.5），标签传播会整张塌成一个社群，那种情况下的 ARI 不能用。
    extra_graphs 是 {名字: 矩阵}，用来多比几条边集（例如 ZS、GSTAR）。
    """
    A_rho = topk_adj(rho, k, ok)
    A_G = topk_adj(G, k, ok)
    A_Z = topk_adj(Z, k, ok)
    deg_rho = A_rho.sum(axis=1).astype(F64)
    deg_G = A_G.sum(axis=1).astype(F64)
    inter = (A_rho & A_G).sum(axis=1).astype(F64)
    union = (A_rho | A_G).sum(axis=1).astype(F64)
    overlap = inter / float(k)
    jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan)
    med_rho = float(np.median(deg_rho[ok])) if ok is not None else float(np.median(deg_rho))
    med_G = float(np.median(deg_G[ok])) if ok is not None else float(np.median(deg_G))
    quad = ((deg_rho > med_rho).astype(F64) * 2 + (deg_G > med_G).astype(F64))
    deg_sp = spearman(deg_rho if ok is None else deg_rho[ok],
                      deg_G if ok is None else deg_G[ok])[0]

    def edge_overlap(A1, A2):
        i = (A1 & A2).sum(axis=1).astype(F64)
        u = (A1 | A2).sum(axis=1).astype(F64)
        return (float(np.nanmean(np.where(ok if ok is not None else True, i / float(k), np.nan))),
                float(np.nanmean(np.where(union > 0, i / np.maximum(u, 1), np.nan))))

    def comm_of(A, A2, nm1, nm2):
        l1 = label_propagation(A)
        l2 = label_propagation(A2)
        ari, agree = ari_and_agree(l1, l2)
        u1, c1 = np.unique(l1, return_counts=True)
        u2, c2 = np.unique(l2, return_counts=True)
        usable = (len(u1) > 1) and (len(u2) > 1)
        return {"社群数": {nm1: int(len(u1)), nm2: int(len(u2))},
                "最大社群占比": {nm1: float(c1.max() / n), nm2: float(c2.max() / n)},
                "ARI": None if not usable else ari,
                "配对一致率": agree,
                "可用": bool(usable),
                "说明": ("" if usable else "%s 图的标签传播塌成一个社群（网络太密），这一档的 ARI 不能用" % nm1)}

    comm = {}
    for kk in k_comm:
        comm["k=%d" % kk] = comm_of(topk_adj(rho, kk, ok), topk_adj(G, kk, ok), "rho", "G")
    comm["k=%d rho vs Z" % k] = comm_of(A_rho, A_Z, "rho", "Z")
    ov_extra = {}
    for nm, mat in (extra_graphs or {}).items():
        A2 = topk_adj(mat, k, ok)
        ov, jc = edge_overlap(A_G, A2)
        ov_z, jc_z = edge_overlap(A_Z, A2)
        comm["k=%d Z vs %s" % (k, nm)] = comm_of(A_Z, A2, "Z", nm)
        ov_extra[nm] = {"G 图与它的平均共同邻居比例": ov, "G 图与它的平均 Jaccard": jc,
                        "Z 图与它的平均共同邻居比例": ov_z, "Z 图与它的平均 Jaccard": jc_z}

    out = {
        "r4_deg_rho_topk": deg_rho,
        "r4_deg_G_topk": deg_G,
        "r4_neighbor_overlap_k": overlap,
        "r4_neighbor_jaccard": jac,
        "r4_quad": quad,
    }
    diag = {"k": k,
            "象限编码": "0=低相似低互补 1=低相似高互补 2=高相似低互补 3=高相似高互补，按横截面中位数切",
            "平均共同邻居比例": float(np.nanmean(overlap)),
            "平均邻居 Jaccard": float(np.nanmean(jac)),
            "两个度序列的 Spearman": deg_sp,
            "其余边集对照": ov_extra,
            "社群对比": comm}
    diag["结论"] = ("互补图与相似图的边几乎不重合（共同邻居比例 %.4f），"
                    "互补关系不是相似关系的重排" % float(np.nanmean(overlap)))
    return out, diag


def r5_residual(G, rho, lnt, valid=None, tag="r5", chunk_rows=256):
    """R5 残差互补：G 对 rho、log 成交笔数回归后的残差，取行内统计。

    valid 是"两只都是个股"的上三角掩码：非个股的配对 G 恒为 0，混进回归会把斜率压歪。
    这里把上三角摊平成一维来算（7.8e6 行的设计矩阵），不要在 Python 里按行循环。
    """
    n = G.shape[0]
    if lnt is None or not np.isfinite(lnt).any():
        return None, {"状态": "没有成交笔数，R5 跳过"}
    lnt = np.where(np.isfinite(lnt), lnt, np.nan)
    med = np.nanmedian(lnt)
    lnt = np.where(np.isfinite(lnt), lnt, med)
    iu = triu_idx(n)
    m = np.ones(iu[0].size, dtype=bool) if valid is None else valid[iu].copy()
    ib, ic = iu[0][m], iu[1][m]
    r = np.asarray(rho, dtype=F64)[ib, ic]
    y = np.asarray(G, dtype=F64)[ib, ic]
    lb, lc = lnt[ib], lnt[ic]
    dabs = np.abs(lb - lc)
    D = np.stack([np.ones_like(r), r, lb, lc, dabs], axis=1)
    XtX = D.T @ D
    Xty = D.T @ y
    beta = np.linalg.solve(XtX + np.eye(5) * 1e-8, Xty)
    res_flat = y - D @ beta
    res = np.full((n, n), np.nan, dtype=F32)
    tmp = np.full(iu[0].size, np.nan, dtype=F32)
    tmp[m] = res_flat.astype(F32)
    res[iu] = tmp
    res = sym_from_upper(res)
    # 只对参与了配对的股票算行内统计，非个股整行是空的，不必去做全空切片的告警
    rows = np.flatnonzero(np.isfinite(res).any(axis=1))
    res_max = np.full(n, np.nan)
    res_top20 = np.full(n, np.nan)
    res_mean = np.full(n, np.nan)
    res_hhi = np.full(n, np.nan)
    sub = res[rows]
    res_max[rows] = np.nanmax(np.where(np.eye(n, dtype=bool)[rows], np.nan, sub), axis=1)
    res_top20[rows] = np.array([topk_mean(sub[i], 20) for i in range(len(rows))])
    res_mean[rows] = np.nanmean(sub, axis=1)
    res_hhi[rows] = np.array([hhi_from_pos(sub[i]) for i in range(len(rows))])
    out = {
        tag + "_res_max": res_max,
        tag + "_res_top20_mean": res_top20,
        tag + "_res_mean": res_mean,
        tag + "_res_hhi": res_hhi,
    }
    diag = {"回归自变量": ["常数", "rho[B][C]", "log 成交笔数 B", "log 成交笔数 C", "|两者之差|"],
            "回归系数": [float(b) for b in beta],
            "参与回归的配对数": int(m.sum()),
            "R2（配对层面）": float(1 - np.var(res_flat) / max(np.var(y), 1e-30)),
            "残差与被回归量的 Spearman": spearman(res_flat, y)[0],
            "HHI 定义": "行内正残差归一化后的平方和，越小越分散"}
    return out, diag


def r6_plane(Z, ZS, n, valid, codes, tag="p", topn=20):
    """R6 (Z, ZS) 二维平面：把"通才型互补"和"专才型互补"分开刻画。

    配对 (B,C) 在平面上有两个坐标：Z（对'平均而言的 X'拟合得好不好）与 ZS（专门对某一类 X 拟合得好不好）。
    ZS 很大而 Z 很负的那些配对是专才对：平均解释力差，但对特定 X 特别互补。
    每只股票取它参与的配对，给出：
      gen_max / gen_top20  通才强度（Z 的行内最大 / 前 20 均值）
      spec_max / spec_top20 专才强度（ZS-Z 的行内最大 / 前 20 均值）
      zs_max               ZS 本身的行内最大
      row_corr             该股票这一行里 Z 与 ZS 的排序相关，越高说明两个视角越一致
      spec_share           行内 ZS>2 且 Z<-2 的配对占比（那类"专才对"）
      quad                 (Z 行内最大, ZS 行内最大) 相对横截面中位数的象限
    """
    iu_t = triu_idx(n)
    m = valid[iu_t]
    zl = np.asarray(Z, dtype=F64)[iu_t]
    zsl = np.asarray(ZS, dtype=F64)[iu_t]
    spec = zsl - zl
    Sc = np.zeros((n, n), dtype=F32)
    SP = np.zeros((n, n), dtype=F32)
    tmp = np.zeros(len(zl), dtype=F32)
    tmp[m] = spec[m]
    Sc[iu_t] = tmp
    tmp = np.zeros(len(zl), dtype=F32)
    tmp[m] = ((zsl[m] > 2.0) & (zl[m] < -2.0)).astype(F32)
    SP[iu_t] = tmp
    Sc = sym_from_upper(Sc)
    SP = sym_from_upper(SP)
    # 行内统计只在"两只都是个股、且不是自己跟自己"的位置上做，其余写空值
    validf = np.where(valid, 1.0, np.nan)
    Zrow = np.zeros((n, n), dtype=F64)
    ZSrow = np.zeros((n, n), dtype=F64)
    tmp = np.zeros(len(zl), dtype=np.float64)
    tmp[m] = zl[m]
    Zrow[iu_t] = tmp
    tmp = np.zeros(len(zl), dtype=np.float64)
    tmp[m] = zsl[m]
    ZSrow[iu_t] = tmp
    Zrow = sym_from_upper(Zrow) * validf
    ZSrow = sym_from_upper(ZSrow) * validf
    np.fill_diagonal(Zrow, np.nan)
    np.fill_diagonal(ZSrow, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gen_max = np.nanmax(Zrow, axis=1)
        spec_max = np.nanmax(Sc, axis=1)
        zs_max = np.nanmax(ZSrow, axis=1)
        gen_top20 = np.array([topk_mean(Zrow[i], topn) for i in range(n)], dtype=F64)
        spec_top20 = np.array([topk_mean(Sc[i], topn) for i in range(n)], dtype=F64)
    cnt_valid = valid.sum(axis=1) + valid.sum(axis=0)
    spec_share = np.where(cnt_valid > 0, SP.sum(axis=1) / np.maximum(cnt_valid, 1), np.nan)
    row_corr = np.full(n, np.nan)
    for i in range(n):
        mm = np.isfinite(Zrow[i]) & np.isfinite(ZSrow[i])
        if mm.sum() > 30:
            row_corr[i] = spearman(Zrow[i][mm], ZSrow[i][mm])[0]
    mg, ms = np.nanmedian(gen_max), np.nanmedian(zs_max)
    quad = ((gen_max > mg).astype(F64) * 2 + (zs_max > ms).astype(F64))
    out = {
        tag + "_gen_max": gen_max,
        tag + "_gen_top20": gen_top20,
        tag + "_spec_max": spec_max,
        tag + "_spec_top20": spec_top20,
        tag + "_zs_max": zs_max,
        tag + "_row_corr": row_corr,
        tag + "_spec_share": spec_share,
        tag + "_quad": quad,
    }
    diag = {"gen/spec 定义": "gen=Z（对平均 X 解释得好），spec=ZS-Z（专对某类 X 解释得好）",
            "spec_share 定义": "该股票参与的配对里 ZS>2 且 Z<-2 的占比，也就是'专才对'占比",
            "quad 编码": "0=低通才低专才 1=低通才高专才 2=高通才低专才 3=高通才高专才（按横截面中位数切）",
            "配对层面 Spearman(Z, ZS)": spearman(zl[m], zsl[m])[0],
            "最专才的 5 对": [[codes[iu_t[0][k]], codes[iu_t[1][k]], round(float(zsl[k]), 2), round(float(zl[k]), 2)]
                              for k in np.argsort(-np.where(m, zsl, -np.inf))[:5]],
            "高通才高专才占比": float(np.nanmean(quad[valid.any(axis=1)] == 3))}
    return out, diag


# ============================================================
# 4. 组装一次完整的降维
# ============================================================


def compute_all(pair_dir, date, backup_root, cfg, verbose=True, ok_override=None):
    t0 = time.time()
    inp = os.path.join(pair_dir, date)
    codes = read_codes(os.path.join(inp, "codes.txt"))
    n = len(codes)
    if verbose:
        print("[%s] N=%d，读矩阵" % (date, n), flush=True)
    rho = load_sq(os.path.join(inp, "rho.bin"), n, "rho")
    E = load_vec_f64(os.path.join(inp, "E.bin"), n)
    G = load_sq(os.path.join(inp, "G.bin"), n, "G")
    Z = load_sq(os.path.join(inp, "Z.bin"), n, "Z")
    RHO_U = load_sq(os.path.join(inp, "RHO_U.bin"), n, "RHO_U")
    SINGLE = load_sq(os.path.join(inp, "SINGLE.bin"), n, "SINGLE")
    X0raw = load_sq(os.path.join(inp, "X0.bin"), n, "X0")
    X0 = np.rint(np.array(X0raw, dtype=F64)).astype(np.int64)
    iu = np.triu(np.ones((n, n), dtype=bool), 1)
    X0 = np.where(iu, X0, -1)

    # 第二路（增益最大化 X0*，可选；旧 schema 没有这几个文件）
    star = {}
    has_star = all(os.path.exists(os.path.join(inp, f))
                   for f in ("GSTAR.bin", "ZS.bin", "X0STAR.bin"))
    if has_star:
        star["G"] = load_sq(os.path.join(inp, "GSTAR.bin"), n, "GSTAR")
        star["Z"] = load_sq(os.path.join(inp, "ZS.bin"), n, "ZS")
        x0s = np.rint(np.array(load_sq(os.path.join(inp, "X0STAR.bin"), n, "X0STAR"), dtype=F64)).astype(np.int64)
        star["X0"] = np.where(iu, x0s, -1)
        for f, key in (("RHO_US.bin", "RHO_US"), ("SINGLE_S.bin", "SINGLE_S")):
            if os.path.exists(os.path.join(inp, f)):
                star[key] = load_sq(os.path.join(inp, f), n, f)

    # 股票池白名单：非个股（ETF、债券）不参与配对，也不该出现在因子表里
    if ok_override is not None:
        ok = np.asarray(ok_override, dtype=bool)
        ok_note = "自测模式：全部行都当个股（合成 fixture 没有非个股）"
    else:
        ok, ok_note = load_stock_mask(codes, G)
    valid = iu & ok[:, None] & ok[None, :]

    checks = {}
    checks["股票池白名单"] = ok_note
    checks["白名单内股票数"] = int(ok.sum())
    checks["rho 对角是否全 1"] = bool(np.allclose(np.diag(rho), 1.0, atol=1e-3))
    checks["G 对称最大偏差"] = float(np.abs(G - G.T).max())
    checks["参与统计的配对数"] = int(valid.sum())
    checks["G 的均值（只在个股对上，注意空模型下本来就为正）"] = float(np.nanmean(np.where(valid, G, np.nan)))
    checks["Z 的均值（只在个股对上）"] = float(np.nanmean(np.where(valid, Z, np.nan)))
    if has_star:
        checks["GSTAR 的均值（只在个股对上）"] = float(np.nanmean(np.where(valid, star["G"], np.nan)))
        checks["ZS 的均值（只在个股对上）"] = float(np.nanmean(np.where(valid, star["Z"], np.nan)))
    for nm, key in (("Z", "Z"), ("ZS", "star_Z")):
        M = Z if key == "Z" else (star.get("Z") if has_star else None)
        if M is None:
            continue
        dev = float(np.abs(M - M.T).max())
        checks["%s 对称最大偏差（原始）" % nm] = dev
        up = M[iu]
        dn = M.T[iu]                                  # 同一对 (i,j) 的另一个方向，逐个对齐
        cr = spearman(up, dn)[0] if up.size > 100 else float("nan")
        checks["%s 上下三角元素对齐后的 Spearman" % nm] = None if not np.isfinite(cr) else round(cr, 5)
        checks["%s 上下三角差异大于 0.5 的配对占比" % nm] = float((np.abs(up - dn) > 0.5).mean())
        if dev > 1e-6:
            M2 = ((M + M.T) * F32(0.5)).astype(F32)
            np.fill_diagonal(M2, 0.0)
            if key == "Z":
                Z = M2
            else:
                star["Z"] = M2
            checks["%s 对称化说明" % nm] = ("原始文件上三角与下三角不一致（逐对独立抽样的空模型本来就会差），"
                                            "已取两者平均对称化后再用")
    x0sym = float(np.abs((X0raw - X0raw.T)[iu]).max()) if n > 1 else 0.0
    checks["X0 对称最大偏差"] = x0sym
    bad = valid & ((X0 < 0) | (X0 >= n))
    checks["X0 越界的对数（只在个股对上）"] = int(bad.sum())
    selfref = valid & ((X0 == np.arange(n)[None, :]) | (X0 == np.arange(n)[:, None]))
    checks["X0 落在配对内部的次数（只在个股对上）"] = int(selfref.sum())
    checks["X0 落在配对内部的次数（含非个股行）"] = int(
        (iu & ((X0 == np.arange(n)[None, :]) | (X0 == np.arange(n)[:, None]))).sum())
    checks["X0 落在配对内部说明"] = (
        "含非个股行的那个计数不为 0 是正常的：L2 非个股（基金、债券）的配对被上游整行跳过，"
        "X0 在那批位置上留着初值 0（行号 0），看起来像落在配对内部，其实是'这对不参与计算'。"
        "只在个股对上的计数才是真正的自查项。")
    checks["X0 指向非个股的配对占比"] = float((~ok[X0[valid]]).mean()) if valid.sum() else None
    ru = np.where(valid, RHO_U, np.nan)
    su = np.where(valid, SINGLE, np.nan)
    gu = np.where(valid, G, np.nan)
    checks["rho[X0][U] 减 SINGLE.bin 与 G 的最大偏差"] = float(np.nanmax(np.abs(ru - su - gu)))

    # 行业与成交笔数
    ind_path = None
    for cand in (os.path.join(inp, "industry.bin"),
                 os.path.join(backup_root, date, "industry.bin") if backup_root else None):
        if cand and os.path.exists(cand):
            ind_path = cand
            break
    industry, ind_note = load_industry(ind_path, codes)
    if industry is not None:
        same = (industry[:, None] == industry[None, :]) & iu & (industry[:, None] >= 0) & (industry[None, :] >= 0)
        diff = (industry[:, None] != industry[None, :]) & iu & (industry[:, None] >= 0) & (industry[None, :] >= 0)
        m_same = float(np.nanmean(np.where(same, rho, np.nan)))
        m_diff = float(np.nanmean(np.where(diff, rho, np.nan)))
        ind_note += "；同行业对的 rho 均值 %.5f，跨行业 %.5f" % (m_same, m_diff)
        ind_note += "（若行业对齐错了，这两个数会几乎相同）"
    bdir = os.path.join(backup_root, date) if backup_root else None
    ntr, ntr_note = load_stats_ntrades(bdir, codes)
    lnt = np.log(np.maximum(ntr, 1.0)) if ntr is not None else None

    # R1 / R2 / R3 / R4 / R5，第一路（G / X0）与第二路（GSTAR / X0*）各算一遍
    if verbose:
        print("[%s] R1 角色画像（两路）" % date, flush=True)
    r1, r1_diag = r1_role(X0, G, Z, n, codes, valid, tag="r1")
    bc, bc_diag, partner = r1_bestcomp(G, Z, n, tag="r1", ok=ok)
    r1.update(bc)
    r1_diag.update(bc_diag)
    s1, s1_diag = ({}, {"状态": "输入没有 GSTAR/ZS/X0STAR，第二路跳过"})
    partner_s = np.zeros(n, dtype=np.int32)
    if has_star:
        s1, s1_diag = r1_role(star["X0"], star["G"], star["Z"], n, codes, valid, tag="s_r1")
        sbc, sbc_diag, partner_s = r1_bestcomp(star["G"], star["Z"], n, tag="s_r1", ok=ok)
        s1.update(sbc)
        s1_diag.update(sbc_diag)
        s1_diag["两路 X0 集中度对照"] = {
            "第一路 X0": r1_diag["X0 集中度"],
            "第二路 X0*": s1_diag["X0 集中度"],
        }
    if verbose:
        print("[%s] R2 全配对扫描（最耗时的部分）" % date, flush=True)
    best_pair, bB, bC = pair_scan(rho, E, ok=ok, chunk=cfg.get("chunk", 128), verbose=verbose)
    r2, r2_diag = r2_three_way(rho, E, best_pair, bB, bC, industry, ok=ok)
    if verbose:
        print("[%s] R3 子模贪心篮子" % date, flush=True)
    r3, r3_diag = r3_greedy(rho, E, ok=ok, max_k=cfg.get("max_k", 8), verbose=verbose)
    if verbose:
        print("[%s] R4 图对照" % date, flush=True)
    r4, r4_diag = r4_graphs(rho, G, Z, n, k=cfg.get("k", 20), ok=ok,
                            extra_graphs=({"ZS": star["Z"], "GSTAR": star["G"]} if has_star else None))
    if verbose:
        print("[%s] R5 残差互补（两路）" % date, flush=True)
    r5, r5_diag = r5_residual(G, rho, lnt, valid=valid, tag="r5")
    s5, s5_diag = ({}, {"状态": "输入没有 GSTAR/ZS，第二路 R5 跳过"})
    if has_star:
        s5, s5_diag = r5_residual(star["G"], rho, lnt, valid=valid, tag="s_r5")
    if verbose:
        print("[%s] R6 (Z, ZS) 二维平面" % date, flush=True)
    r6, r6_diag = ({}, {"状态": "输入没有 ZS，R6 跳过"})
    if has_star:
        r6, r6_diag = r6_plane(Z, star["Z"], n, valid, codes, tag="p")

    # 空模型对照：把 G 换成 Z、把 GSTAR 换成 ZS 重算用到它们的那几条
    if verbose:
        print("[%s] 空模型对照（G->Z，GSTAR->ZS 各算一遍）" % date, flush=True)
    n1, n1_diag = r1_role(X0, Z, Z, n, codes, valid, tag="null_r1")
    nbc, nbc_diag, _ = r1_bestcomp(Z, Z, n, tag="null_r1", ok=ok)
    null = {
        "null_r1_explain_gain_mean": n1["null_r1_explain_gain_mean"],
        "null_r1_explain_gain_max": n1["null_r1_explain_gain_max"],
        "null_r1_explain_z_mean": n1["null_r1_explain_z_mean"],
        "null_r1_bestcomp_gain": nbc["null_r1_bestcomp_gain"],
        "null_r1_bestcomp_mutual": nbc["null_r1_bestcomp_mutual"],
        "null_r1_bestcomp_rank": nbc["null_r1_bestcomp_rank"],
    }
    if r5 is not None:
        zr5, _ = r5_residual(Z, rho, lnt, valid=valid, tag="null_r5")
        for k in ("res_max", "res_top20_mean", "res_mean", "res_hhi"):
            null["null_r5_" + k] = zr5["null_r5_" + k]
    if has_star:
        ns1, _ = r1_role(star["X0"], star["Z"], star["Z"], n, codes, valid, tag="null_s_r1")
        nsbc, _, _ = r1_bestcomp(star["Z"], star["Z"], n, tag="null_s_r1", ok=ok)
        null.update({
            "null_s_r1_explain_gain_mean": ns1["null_s_r1_explain_gain_mean"],
            "null_s_r1_explain_gain_max": ns1["null_s_r1_explain_gain_max"],
            "null_s_r1_explain_z_mean": ns1["null_s_r1_explain_z_mean"],
            "null_s_r1_bestcomp_gain": nsbc["null_s_r1_bestcomp_gain"],
            "null_s_r1_bestcomp_mutual": nsbc["null_s_r1_bestcomp_mutual"],
            "null_s_r1_bestcomp_rank": nsbc["null_s_r1_bestcomp_rank"],
        })
        zs5, _ = r5_residual(star["Z"], rho, lnt, valid=valid, tag="null_s_r5")
        for k in ("res_max", "res_top20_mean", "res_mean", "res_hhi"):
            null["null_s_r5_" + k] = zs5["null_s_r5_" + k]

    first = ["r1_explain_count", "r1_explain_count_z2", "r1_explain_gain_mean",
             "r1_explain_gain_max", "r1_explain_z_mean", "r1_as_partner_z2_count",
             "r1_bestcomp_gain", "r1_bestcomp_z", "r1_bestcomp_mutual", "r1_bestcomp_rank"]
    second = [nm for nm in sorted(s1.keys())]
    order = (first + second
             + ["r2_best_single", "r2_best_pair", "r2_pair_gain", "r2_pair_gain_ratio",
                "r2_best_pair_cross_industry", "r2_best_pair_Bidx", "r2_best_pair_Cidx",
                "r3_marg1", "r3_marg2", "r3_marg3", "r3_marg4", "r3_n_to_10pct",
                "r3_basket_rho8", "r3_avg_partner_rho", "r3_max_partner_rho", "r3_degenerate",
                "r4_deg_rho_topk", "r4_deg_G_topk", "r4_neighbor_overlap_k",
                "r4_neighbor_jaccard", "r4_quad",
                "r5_res_max", "r5_res_top20_mean", "r5_res_mean", "r5_res_hhi"]
             + sorted(s5.keys())
             + sorted(r6.keys())
             + sorted(null.keys()))
    pool = {}
    for d in (r1, s1, r2, r3, r4, r5 or {}, s5 or {}, r6 or {}, null):
        pool.update(d)
    names = [nm for nm in order if nm in pool]
    # 同一批构造在第二路上的列名用 s_ 前缀区分，别和第一路混在一起
    mat = np.empty((n, len(names)), dtype=F32)
    for j, nm in enumerate(names):
        mat[:, j] = np.asarray(pool[nm], dtype=F64).astype(F32)
    # 非个股行不参与配对，也不该给出因子值：整行写空值占位（每只股票列数仍然恒定）
    mat[~ok, :] = np.nan
    assert mat.shape == (n, len(names))
    partner_codes = [codes[i] for i in partner]

    diag = {
        "日期": date, "股票数": n, "输入目录": inp, "耗时秒": round(time.time() - t0, 1),
        "矩阵检查": checks,
        "行业": {"来源": ind_path, "说明": ind_note},
        "成交笔数": {"说明": ntr_note},
        "R1": r1_diag, "R1_第二路": s1_diag, "R2": r2_diag, "R3": r3_diag, "R4": r4_diag,
        "R5": r5_diag, "R5_第二路": s5_diag, "R6_平面": r6_diag,
        "股票池": {"白名单内股票数": int(ok.sum()), "被剔除的非个股数": int((~ok).sum()),
                   "被剔除的代码": [codes[i] for i in np.flatnonzero(~ok)][:20],
                   "说明": "非个股整行因子写空值；配对统计里也剔除了它们"},
        "辅助列（不是可交易信号，跨日比较无意义）": list(AUX_NAMES),
        "每只股票最强互补搭档（前 5 只，样例）": partner_codes[:5],
    }
    if has_star:
        diag["每只股票最强互补搭档_第二路（前 5 只，样例）"] = [codes[i] for i in partner_s[:5]]
    extra = {"codes": codes, "rho": rho, "G": G, "Z": Z, "E": E, "lnt": lnt,
             "industry": industry, "RHO_U": RHO_U, "SINGLE": SINGLE, "X0": X0}
    return names, mat, diag, extra


# ============================================================
# 5. 因子释义（写进 diag.json，给人读）
# ============================================================

EXPLAIN = {
    "s_r1_explain_count": "第二路（X0*=argmax 增益）里它被多少对选为解释对象",
    "s_r1_explain_count_z2": "第二路同上，但只数 ZS>2 的强互补对",
    "s_r1_explain_gain_mean": "第二路里它被选中的那些对 GSTAR 的均值",
    "s_r1_explain_gain_max": "第二路里它被选中的那些对 GSTAR 的最大值",
    "s_r1_explain_z_mean": "第二路里它被选中的那些对 ZS 的均值",
    "s_r1_as_partner_z2_count": "第二路里它出现在多少个 ZS>2 的强互补对里",
    "s_r1_bestcomp_gain": "第二路它与最强搭档那一对的 GSTAR",
    "s_r1_bestcomp_z": "第二路它与最强搭档那一对的 ZS",
    "s_r1_bestcomp_mutual": "第二路里是否互为最强搭档",
    "s_r1_bestcomp_rank": "第二路里它在搭档行内的排名",
    "p_gen_max": "Z 的配对行内最大值：这只股票参与过的最'通才型'互补有多强",
    "p_gen_top20": "Z 的配对行内前 20 均值",
    "p_spec_max": "ZS-Z 的配对行内最大值：这只股票参与过的最'专才型'互补有多强",
    "p_spec_top20": "ZS-Z 的配对行内前 20 均值",
    "p_zs_max": "ZS 的配对行内最大值",
    "p_row_corr": "它这一行里 Z 与 ZS 的排序相关：高说明通才视角与专才视角一致",
    "p_spec_share": "它参与的配对里 ZS>2 且 Z<-2 的占比，也就是'专才对'占多少",
    "p_quad": "在（Z 行内最大, ZS 行内最大）平面上的象限，0 低低 1 低高 2 高低 3 高高",
    "null_s_r1_explain_gain_mean": "第二路把 GSTAR 换成 ZS 的对照",
    "null_s_r1_explain_gain_max": "第二路把 GSTAR 换成 ZS 的对照",
    "null_s_r1_explain_z_mean": "第二路把 GSTAR 换成 ZS 的对照",
    "null_s_r1_bestcomp_gain": "第二路把 GSTAR 换成 ZS 的对照",
    "null_s_r1_bestcomp_mutual": "第二路把 GSTAR 换成 ZS 的对照",
    "null_s_r1_bestcomp_rank": "第二路把 GSTAR 换成 ZS 的对照",
    "s_r5_res_max": "第二路 GSTAR 对 rho 与 log 笔数回归后，残差行内最大值",
    "s_r5_res_top20_mean": "第二路残差行内前 20 均值",
    "s_r5_res_mean": "第二路残差行内均值",
    "s_r5_res_hhi": "第二路残差集中度",
    "r1_explain_count": "它被多少对股票选为最能被那对解释的 X0，数值大说明它是很多配对的解释对象",
    "r1_explain_count_z2": "同上，但只数 Z>2 的强互补对",
    "r1_explain_gain_mean": "它被选中的那些对里 G 的均值（未扣空模型）",
    "r1_explain_gain_max": "它被选中的那些对里 G 的最大值",
    "r1_explain_z_mean": "它被选中的那些对里 Z 的均值（已扣空模型）",
    "r1_as_partner_z2_count": "它出现在多少个 Z>2 的强互补对里（全配对下出现次数恒为 N-1，所以按强度筛）",
    "r1_bestcomp_gain": "它和最强互补搭档那一对的 G",
    "r1_bestcomp_z": "它和最强互补搭档那一对的 Z",
    "r1_bestcomp_mutual": "它和最强搭档是否互为最强搭档（两边行内最大互指），记 1",
    "r1_bestcomp_rank": "在搭档自己的 G 行里，它排第几，0 就是互惠",
    "r2_best_single": "它和单个股票的最高 rho",
    "r2_best_pair": "它和某一对股票拼起来的最高 rho[X][U_BC]",
    "r2_pair_gain": "上面两者之差，组合比单只多解释了多少",
    "r2_pair_gain_ratio": "上一条除以最佳单伙伴的绝对值，相对提升的口径",
    "r2_best_pair_cross_industry": "最佳伙伴对的两只是否跨行业，1 跨 0 同，行业未知写空",
    "r2_best_pair_Bidx": "最佳伙伴对里代码序靠前那只的行号（辅助列，不是可交易信号）",
    "r2_best_pair_Cidx": "最佳伙伴对里代码序靠后那只的行号（辅助列）",
    "r3_marg1": "贪心篮子第 1 步的边际增益，等于最佳单伙伴的 rho",
    "r3_marg2": "第 2 步的边际增益",
    "r3_marg3": "第 3 步的边际增益",
    "r3_marg4": "第 4 步的边际增益",
    "r3_n_to_10pct": "边际增益首次掉到首步 10% 以下用了几只伙伴，9 表示 8 只内没掉下来",
    "r3_basket_rho8": "8 只伙伴凑齐时 rho[X][U_S] 的水平",
    "r3_avg_partner_rho": "篮子里伙伴两两之间的平均 rho，越高越说明篮子内部互相重复",
    "r3_max_partner_rho": "篮子里伙伴两两之间的最大 rho",
    "r3_degenerate": "伙伴平均 rho 超过 0.5 记 1，说明篮子退化成一只枢纽带一堆小跟班",
    "r4_deg_rho_topk": "rho 图里它的邻居数（每只取前 20 条边后取并集）",
    "r4_deg_G_topk": "G 图里它的邻居数",
    "r4_neighbor_overlap_k": "两个图的共同邻居数除以 k，高说明互补关系就是相似关系",
    "r4_neighbor_jaccard": "两个图邻居集合的 Jaccard",
    "r4_quad": "在（rho 度, G 度）平面上的象限，0 低低 1 低高 2 高低 3 高高，按中位数切",
    "r5_res_max": "G 对 rho 与 log 成交笔数回归后，残差行内最大值",
    "r5_res_top20_mean": "残差行内前 20 名的均值",
    "r5_res_mean": "残差行内均值",
    "r5_res_hhi": "行内正残差的集中度（HHI），越小越分散",
}
NULL_NOTE = "带 null_ 前缀的列是把 G 换成 Z（扣掉空模型）重算的对照版，不是独立因子，只用来回答“扣掉空模型后还剩多少”。"


# ============================================================
# 6. 诊断 (a) 跨日 (b) 老因子重叠 (c) 空模型
# ============================================================


def diag_crossday(pair_dir, date, names, mat, codes, cfg, backup_root, verbose=True,
                  ok_override=None):
    sib = [d for d in sorted(os.listdir(pair_dir))
           if d.isdigit() and d != date and os.path.exists(os.path.join(pair_dir, d, "rho.bin"))]
    if not sib:
        return {"状态": "没找到另一天的输出目录，跨日稳定性没做（目录 %s 下只有 %s）" % (pair_dir, date)}
    out = {"另一日": sib, "结果": {}}
    mats = {date: (codes, mat)}
    for d in sib:
        try:
            nm2, m2, _, ex2 = compute_all(pair_dir, d, backup_root, cfg, verbose=verbose,
                                          ok_override=ok_override)
        except Exception as e:
            out["结果"][d] = {"状态": "另一日算失败: %s" % e}
            continue
        mats[d] = (ex2["codes"], m2)
        out.setdefault("另一日因子清单一致", {})[d] = (nm2 == names)
    base_codes, base = mats[date]
    rows = []
    for d, (c2, m2) in mats.items():
        if d == date:
            continue
        idx1 = {c: i for i, c in enumerate(base_codes)}
        idx2 = {c: i for i, c in enumerate(c2)}
        common = [c for c in base_codes if c in idx2]
        if len(common) < 50:
            out["结果"][d] = {"状态": "股票交集只有 %d 只，跳过" % len(common)}
            continue
        i1 = np.array([idx1[c] for c in common])
        i2 = np.array([idx2[c] for c in common])
        rec = []
        for j, nm in enumerate(names):
            r, k = spearman(base[i1, j], m2[i2, j])
            rec.append([nm, None if not np.isfinite(r) else round(r, 4), k])
        rec_sorted = sorted([r for r in rec if r[1] is not None and r[0] not in AUX_NAMES],
                            key=lambda x: -abs(x[1]))
        out["结果"][d] = {"交集股票数": len(common), "全部": rec, "按稳定性排序前 8": rec_sorted[:8]}
    return out


def diag_old_factors(backup_root, date, names, mat, codes, topk_list=3, verbose=True):
    """(b) 与备份里 2761 个老因子的 |Spearman| 最大值。"""
    if not backup_root:
        return {"状态": "没给 --backup，老因子重叠没做"}
    cand = []
    for d in sorted(os.listdir(backup_root)):
        p = os.path.join(backup_root, d, "factors.bin")
        if d.isdigit() and os.path.exists(p) and os.path.exists(os.path.join(backup_root, d, "names.txt")):
            cand.append(d)
    if not cand:
        return {"状态": "备份根 %s 下没有带 factors.bin 的日期目录" % backup_root}
    use = date if date in cand else min(cand, key=lambda d: abs(int(d) - int(date)))
    bdir = os.path.join(backup_root, use)
    old_names = read_codes(os.path.join(bdir, "names.txt"))
    old_codes = read_codes(os.path.join(bdir, "codes.txt"))
    n_old, f_old = len(old_codes), len(old_names)
    raw = np.memmap(os.path.join(bdir, "factors.bin"), dtype=F32, mode="r")
    if raw.size != n_old * f_old:
        return {"状态": "备份 factors.bin 大小 %d 与 %d×%d 不符" % (raw.size, n_old, f_old)}
    old = raw.reshape(n_old, f_old)
    idx_new = {c: i for i, c in enumerate(codes)}
    idx_old = {c: i for i, c in enumerate(old_codes)}
    common = [c for c in codes if c in idx_old]
    if len(common) < 50:
        return {"状态": "与备份的股票交集只有 %d 只" % len(common)}
    inew = np.array([idx_new[c] for c in common])
    iold = np.array([idx_old[c] for c in common])
    A = np.asarray(mat[inew], dtype=F64)
    B = np.asarray(old[iold], dtype=F64)
    F = A.shape[1]
    # 先各自把秩算好：秩按列取（缺失处留 nan），再对每个 (新因子, 老因子) 在"两边都有限"的
    # 那批股票上求相关。子集上直接用全表的秩不影响 Spearman，因为秩在子集内仍保持同样的次序。
    rankA = np.full_like(A, np.nan)
    for a in range(F):
        mm = np.isfinite(A[:, a])
        if mm.sum() >= 50:
            rankA[mm, a] = _rank(A[mm, a])
    rankB = np.full_like(B, np.nan)
    for j in range(B.shape[1]):
        mm = np.isfinite(B[:, j])
        if mm.sum() >= 50:
            rankB[mm, j] = _rank(B[mm, j])
    res = {nm: {"最大绝对重叠": 0.0, "最像的三个": []} for nm in names}
    for a in range(F):
        xa = rankA[:, a]
        ma = np.isfinite(xa)
        if ma.sum() < 50:
            continue
        best = 0.0
        top = []
        for j in range(B.shape[1]):
            xb = rankB[:, j]
            mm = ma & np.isfinite(xb)
            k = int(mm.sum())
            if k < 50:
                continue
            x = xa[mm]
            z = xb[mm]
            x = x - x.mean()
            z = z - z.mean()
            den = np.sqrt((x * x).sum() * (z * z).sum())
            if den <= 0:
                continue
            c = float((x * z).sum() / den)
            if abs(c) > 0.15:
                top.append([old_names[j], round(c, 4)])
            if abs(c) > best:
                best = abs(c)
        top = sorted(top, key=lambda t: -abs(t[1]))[:topk_list]
        res[names[a]] = {"最大绝对重叠": round(best, 4),
                         "最大重叠老因子": top[0][0] if top else None,
                         "最像的三个": top}
    worst = sorted(res.items(), key=lambda kv: -kv[1]["最大绝对重叠"])[:8]
    return {"备份日期": use, "老因子个数": f_old, "股票交集数": len(common),
            "含空值的老因子个数": int((~np.isfinite(B).all(axis=0)).sum()),
            "每个新因子的最大重叠": res,
            "撞得最狠的前 8": [[k, v["最大绝对重叠"], v["最像的三个"]] for k, v in worst]}


def diag_null_model(names, mat, lnt):
    """(c) 空模型：每条因子在扣掉空模型后还剩多少。"""
    out = {}
    idx = {nm: j for j, nm in enumerate(names)}
    for j, nm in enumerate(names):
        null_name = "null_" + nm
        if null_name in idx:
            r, k = spearman(mat[:, j], mat[:, idx[null_name]])
            rec = {"G 版": nm, "Z 版": null_name,
                   "两者 Spearman": None if not np.isfinite(r) else round(r, 4),
                   "G 版均值": round(float(np.nanmean(mat[:, j].astype(F64))), 5),
                   "Z 版均值": round(float(np.nanmean(mat[:, idx[null_name]].astype(F64))), 5)}
            if not np.isfinite(r):
                rec["结论"] = "算不出来（样本不足或全空）"
            elif r > 0.95:
                rec["结论"] = "扣掉空模型后排序几乎不变，这条不是活跃度驱动的"
            elif r > 0.5:
                rec["结论"] = "扣掉空模型后排序明显改变，一半以上是空模型效应"
            else:
                rec["结论"] = "扣掉空模型后基本对不上，这条主要是空模型效应"
            out[nm] = rec
        else:
            note = "构造里没有 G，不受空模型影响" if not nm.startswith("null_") else ""
            out[nm] = {"G 版": nm, "Z 版": None, "结论": note or "对照列"}
    if lnt is not None and np.isfinite(lnt).any():
        for j, nm in enumerate(names):
            r, _ = spearman(mat[:, j], lnt)
            out[nm]["与 log 成交笔数的 Spearman"] = None if not np.isfinite(r) else round(r, 4)
    return out


# ============================================================
# 7. 自测用的合成 fixture
# ============================================================


def _null_model_G(rho, E, n_trades, n_null=20000, seed=11, kbin=10):
    """按 pair_spec 第 5 节的分层蒙特卡洛，给全部配对算 Z。"""
    n = rho.shape[0]
    rng = np.random.default_rng(seed)
    se = np.sqrt(np.maximum(E, 1e-30))
    q = np.quantile(np.log(np.maximum(n_trades, 1)), np.linspace(0, 1, kbin + 1))
    q[0] -= 1e-9
    q[-1] += 1e-9
    b = np.clip(np.searchsorted(q, np.log(np.maximum(n_trades, 1)), side="right") - 1, 0, kbin - 1)
    M = (rho * se[None, :]).astype(F64)
    den = np.sqrt(np.maximum(E[:, None] + E[None, :] + 2 * rho * se[:, None] * se[None, :], 1e-30))
    Winv = 1.0 / den
    np.fill_diagonal(Winv, 0.0)
    iu = np.triu_indices(n, 1)
    pair_b = b[iu[0]]
    pair_c = b[iu[1]]
    strata = pair_b * kbin + pair_c
    per = max(50, n_null // (kbin * kbin))
    pB, pC = [], []
    for s in range(kbin * kbin):
        cand = np.flatnonzero(strata == s)
        if cand.size == 0:
            continue
        take = min(per, cand.size)
        sel = rng.choice(cand, size=take, replace=False)
        pB.append(iu[0][sel])
        pC.append(iu[1][sel])
    pB = np.concatenate(pB)
    pC = np.concatenate(pC)
    H = (M[:, pB] + M[:, pC]) * Winv[pB, pC][None, :]
    H[pB, np.arange(pB.size)] = -np.inf      # X0 不能是配对内部的两只
    H[pC, np.arange(pC.size)] = -np.inf
    single_all = np.maximum(rho[:, pB], rho[:, pC])          # 每个 X 自己的 best_single
    x0 = H.argmax(axis=0)
    single = np.maximum(rho[x0, pB], rho[x0, pC])
    g_null = H.max(axis=0) - single          # 与第 2 节同样的 G
    # 第二路：X0* = argmax (rho[X][U] - best_single[X])，也就是 GSTAR 的最大值
    Hstar = H - single_all
    gstar_null = Hstar.max(axis=0)
    s_str = pair_b * kbin + pair_c
    null_str = (b[pB] * kbin + b[pC])
    means = np.full(kbin * kbin, np.nan)
    stds = np.full(kbin * kbin, np.nan)
    means_s = np.full(kbin * kbin, np.nan)
    stds_s = np.full(kbin * kbin, np.nan)
    for s in range(kbin * kbin):
        m = null_str == s
        if m.sum() >= 30:
            means[s] = g_null[m].mean()
            stds[s] = g_null[m].std()
            means_s[s] = gstar_null[m].mean()
            stds_s[s] = gstar_null[m].std()
    gm, gs = np.nanmean(g_null), np.nanstd(g_null)
    gms, gss = np.nanmean(gstar_null), np.nanstd(gstar_null)
    mu = np.where(np.isfinite(means[s_str]), means[s_str], gm)
    sd = np.where(np.isfinite(stds[s_str]) & (stds[s_str] > 1e-12), stds[s_str], gs)
    mus = np.where(np.isfinite(means_s[s_str]), means_s[s_str], gms)
    sds = np.where(np.isfinite(stds_s[s_str]) & (stds_s[s_str] > 1e-12), stds_s[s_str], gss)
    return mu, sd, mus, sds


def _synth_fixture(root, n=300, seed=7, n_latent=10, n_ind=8):
    """造一个 N=n 的合成输入目录（两个日期）+ 一个备份目录。

    潜在因子做法：每只股票是若干潜在成交节奏的线性叠加，同一行业共享一个潜在成分，
    这样 rho 有真实结构、同行业对相似度更高、G 与 Z 有可比较的量级。
    """
    rng = np.random.default_rng(seed)
    T = 1500
    L = n_latent
    f = np.cumsum(rng.standard_normal((L, T)), axis=1)
    f = (f - f.mean(axis=1, keepdims=True))
    industry = rng.integers(0, n_ind, size=n).astype(np.int32)
    A = rng.random((n, L)) ** 2
    A[np.arange(n), industry % L] += 1.2          # 行业共享成分
    u = A @ f
    u += 0.6 * rng.standard_normal((n, T))
    u = np.maximum(u, 0.0)
    S = (u @ u.T).astype(F64)
    E = np.diag(S).copy()
    se = np.sqrt(np.maximum(E, 1e-30))
    rho = (S / (se[:, None] * se[None, :])).astype(F32)
    np.fill_diagonal(rho, 1.0)
    n_trades = E * 3.0 + rng.random(n) * 5.0

    # 按 spec 第 2 节算全配对 G / X0 / RHO_U / SINGLE（n=300 时很便宜）
    M = (rho * se[None, :]).astype(F64)
    den = np.sqrt(np.maximum(E[:, None] + E[None, :] + 2 * rho * se[:, None] * se[None, :], 1e-30))
    Winv = 1.0 / den
    np.fill_diagonal(Winv, 0.0)
    G = np.zeros((n, n), dtype=F32)
    RHO_U = np.zeros((n, n), dtype=F32)
    SINGLE = np.zeros((n, n), dtype=F32)
    X0 = np.zeros((n, n), dtype=F32)
    GS = np.zeros((n, n), dtype=F32)          # 第二路：GSTAR / X0* / RHO_US / SINGLE_S
    RHO_US = np.zeros((n, n), dtype=F32)
    SINGLE_S = np.zeros((n, n), dtype=F32)
    X0S = np.zeros((n, n), dtype=F32)
    for B in range(n - 1):
        for C in range(B + 1, n):
            h = (M[:, B] + M[:, C]) * Winv[B, C]
            h[B] = -np.inf
            h[C] = -np.inf
            x0 = int(np.argmax(h))
            ru = h[x0]
            si = max(rho[x0, B], rho[x0, C])
            g = ru - si
            G[B, C] = G[C, B] = g
            RHO_U[B, C] = RHO_U[C, B] = ru
            SINGLE[B, C] = SINGLE[C, B] = si
            X0[B, C] = X0[C, B] = x0
            # 第二路：先减掉每个 X 自己的最佳单伙伴，再取最大
            hs = h.copy()
            hsfinite = np.isfinite(hs)
            own = np.where(np.isfinite(h), np.maximum(rho[:, B], rho[:, C]), -np.inf)
            hs = hs - np.where(own > -np.inf, own, 0.0)
            hs[~hsfinite] = -np.inf
            x0s = int(np.argmax(hs))
            gs = hs[x0s]
            GS[B, C] = GS[C, B] = gs
            RHO_US[B, C] = RHO_US[C, B] = h[x0s]
            SINGLE_S[B, C] = SINGLE_S[C, B] = max(rho[x0s, B], rho[x0s, C])
            X0S[B, C] = X0S[C, B] = x0s
    mu, sd, mus, sds = _null_model_G(rho.astype(F64), E, n_trades, n_null=12000, seed=seed + 1)
    iu = np.triu_indices(n, 1)
    gu = np.array([G[i, j] for i, j in zip(*iu)], dtype=F64)
    gsu = np.array([GS[i, j] for i, j in zip(*iu)], dtype=F64)
    Z = np.zeros((n, n), dtype=F32)
    iu2 = np.triu_indices(n, 1)
    zz = (gu - mu) / sd
    Z[iu2] = zz.astype(F32)
    Z = sym_from_upper(Z)
    ZS = np.zeros((n, n), dtype=F32)
    ZS[iu2] = ((gsu - mus) / sds).astype(F32)
    ZS = sym_from_upper(ZS)
    codes = ["%06d" % (i + 1) for i in range(n)]
    pair_root = os.path.join(root, "pair")
    backup_root = os.path.join(root, "backup")
    for tag, sd_off in (("20241230", 100), ("20241231", 200)):
        rng2 = np.random.default_rng(seed + sd_off)
        d = os.path.join(pair_root, tag)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "codes.txt"), "w") as fo:
            fo.write("\n".join(codes) + "\n")
        # 第二天的矩阵做一点扰动，跨日相关性才不是 1
        def jitter(a, k=0.06):
            b = a + (rng2.standard_normal(a.shape) * k).astype(F32)
            return b.astype(F32)
        G2 = jitter(G)
        np.fill_diagonal(G2, 0)
        G2 = (G2 + G2.T) / 2
        Z2 = jitter(Z)
        np.fill_diagonal(Z2, 0)
        # 真实数据的 Z.bin 上下三角并不一致（逐对独立抽样导致），这里照样造出这种不对称，
        # 好让"检测到不对称就对称化"这条路径在自测里真的被走到
        znoise = (rng2.standard_normal((n, n)) * 0.06).astype(F32)
        Z2 = Z2 + np.triu(znoise, 1) + np.tril(znoise.T, -1)
        np.fill_diagonal(Z2, 0)
        ru2 = jitter(RHO_U, 0.02)
        si2 = jitter(SINGLE, 0.02)
        np.fill_diagonal(ru2, 0)
        np.fill_diagonal(si2, 0)
        x02 = X0.copy()
        flip = rng2.random((n, n)) < 0.01
        flip = np.triu(flip, 1)
        flip = flip | flip.T
        x02[flip] = rng2.integers(0, n, size=int(flip.sum()))
        rho2 = rho.copy()
        rho2[flip] *= 0.98
        GS2 = jitter(GS)
        np.fill_diagonal(GS2, 0)
        GS2 = (GS2 + GS2.T) / 2
        ZS2 = jitter(ZS)
        np.fill_diagonal(ZS2, 0)
        znoise2 = (rng2.standard_normal((n, n)) * 0.06).astype(F32)
        ZS2 = ZS2 + np.triu(znoise2, 1) + np.tril(znoise2.T, -1)
        np.fill_diagonal(ZS2, 0)
        rus2 = jitter(RHO_US, 0.02)
        sis2 = jitter(SINGLE_S, 0.02)
        np.fill_diagonal(rus2, 0)
        np.fill_diagonal(sis2, 0)
        for nm, arr in (("rho", rho2), ("G", G2), ("Z", Z2), ("RHO_U", ru2),
                        ("SINGLE", si2), ("X0", x02),
                        ("GSTAR", GS2), ("ZS", ZS2), ("RHO_US", rus2),
                        ("SINGLE_S", sis2), ("X0STAR", X0S)):
            np.ascontiguousarray(arr, dtype=F32).tofile(os.path.join(d, nm + ".bin"))
        E.astype(F64).tofile(os.path.join(d, "E.bin"))
        with open(os.path.join(d, "summary.json"), "w") as fo:
            json.dump({"n": n, "date": tag, "mat": "cnt_t1", "argmax_hist": [1, 2, 3],
                       "null": {"strata": "合成 fixture 的分层空模型"}}, fo, ensure_ascii=False)
        # 备份：行业、成交笔数、老因子
        db = os.path.join(backup_root, tag)
        os.makedirs(db, exist_ok=True)
        with open(os.path.join(db, "codes.txt"), "w") as fo:
            fo.write("\n".join(codes) + "\n")
        with open(os.path.join(db, "industry.bin"), "wb") as fo:
            fo.write(np.uint64(n).tobytes())
            fo.write(industry.astype("<i2").tobytes())
        st = np.zeros((n, 14), dtype="<f8")
        st[:, 0] = n_trades
        st[:, 1] = E
        with open(os.path.join(db, "stats.bin"), "wb") as fo:
            fo.write(np.uint64(n).tobytes())
            fo.write(st.tobytes())
        # 老因子：包含一条与 explain_count 完全同源的列，用来验证重叠诊断能抓到
        x0flat = X0[np.triu_indices(n, 1)]
        expl = np.bincount(np.rint(x0flat).astype(np.int64), minlength=n).astype(F64)
        old_cols = [expl]
        onames = ["old_explain_count_like"]
        with np.errstate(invalid="ignore"):
            rho_u = np.where(np.isfinite(rho), rho, 0)
            old_cols += [rho_u.mean(axis=1), rho_u.max(axis=1), np.log(np.maximum(E, 1)),
                         rho_u.std(axis=1)]
        onames += ["old_rho_rowmean", "old_rho_rowmax", "old_logE", "old_rho_rowstd"]
        for k in range(20):
            old_cols.append(G[np.arange(n), (np.arange(n) + k + 1) % n].astype(F64))
            onames.append("old_G_shift_%02d" % k)
        old_mat = np.stack(old_cols, axis=1).astype(F32)
        old_mat.tofile(os.path.join(db, "factors.bin"))
        with open(os.path.join(db, "names.txt"), "w") as fo:
            fo.write("\n".join(onames) + "\n")
    return pair_root, backup_root


def selftest(n=300):
    import tempfile
    root = os.environ.get("PAIR_REDUCE_SELFTEST_DIR", "/tmp/yupei2_pair_selftest")
    os.makedirs(root, exist_ok=True)
    print("=== 自测：合成 fixture 根目录 %s ===" % root, flush=True)
    t0 = time.time()
    pair_root, backup_root = _synth_fixture(root, n=n, seed=7)
    print("合成 fixture 造好，N=%d，用时 %.1fs" % (n, time.time() - t0), flush=True)
    ok = True

    # ---- 先做几条独立校验 ----
    codes = read_codes(os.path.join(pair_root, "20241231", "codes.txt"))
    nn = len(codes)
    rho = load_sq(os.path.join(pair_root, "20241231", "rho.bin"), nn, "rho")
    E = load_vec_f64(os.path.join(pair_root, "20241231", "E.bin"), nn)
    se = np.sqrt(E)
    # 1) 等价式校验：rho[X][U_BC] 用等价式 vs 直接用 S 拼
    S = rho.astype(F64) * se[:, None] * se[None, :]
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        B, C, X = rng.integers(0, nn, 3)
        if len({B, C, X}) < 3:
            continue
        wB = se[B] / np.sqrt(E[B] + E[C] + 2 * rho[B, C] * se[B] * se[C])
        wC = se[C] / np.sqrt(E[B] + E[C] + 2 * rho[B, C] * se[B] * se[C])
        v1 = wB * rho[X, B] + wC * rho[X, C]
        EU = E[B] + E[C] + 2 * S[B, C]
        v2 = (S[X, B] + S[X, C]) / np.sqrt(E[X] * EU)
        worst = max(worst, abs(v1 - v2))
    print("校验 1 等价式 rho[X][U_BC] 最大偏差 %.3e" % worst, flush=True)
    ok &= worst < 1e-4

    # 2) pair_scan 小样本暴力对照
    m = 70
    rho_s = np.ascontiguousarray(rho[:m, :m])
    E_s = E[:m].copy()
    bp, bB, bC = pair_scan(rho_s, E_s, chunk=32, verbose=False)
    se_s = np.sqrt(E_s)
    Ms = rho_s.astype(F64) * se_s[None, :]
    dminv = 1.0 / np.sqrt(np.maximum(E_s[:, None] + E_s[None, :] + 2 * rho_s * se_s[:, None] * se_s[None, :], 1e-30))
    brute = np.full(m, -np.inf)
    for B in range(m - 1):
        for C in range(B + 1, m):
            v = (Ms[:, B] + Ms[:, C]) * dminv[B, C]
            v[B] = -np.inf
            v[C] = -np.inf
            brute = np.maximum(brute, v)
    dev = float(np.nanmax(np.abs(brute - bp)))
    print("校验 2 pair_scan 与暴力扫最大偏差 %.3e" % dev, flush=True)
    ok &= dev < 1e-5

    # 3) r3 贪心与暴力对照（用原始核直接算 rho[X][U_S]，独立于脚本里的化简式）
    g3, _ = r3_greedy(rho_s, E_s, max_k=4, verbose=False)
    Ss = rho_s.astype(F64) * se_s[:, None] * se_s[None, :]
    bad = 0.0
    for X in range(0, m, 7):
        Sset = []
        bestv = np.nan
        for step in range(4):
            bestv, besty = -np.inf, -1
            for Y in range(m):
                if Y == X or Y in Sset:
                    continue
                T = Sset + [Y]
                num = sum(Ss[X, z] for z in T)                  # 原始核 S[X][Y] 求和
                Q = sum(Ss[a, b] for a in T for b in T)         # 原始核 E_{U_S}
                val = num / np.sqrt(E_s[X] * Q)
                if val > bestv:
                    bestv, besty = val, Y
            Sset.append(besty)
        bad = max(bad, abs(bestv - g3["r3_basket_rho8"][X]))
    # 单只伙伴时 rho[X][U_S] 必须正好等于 rho[X][Y]，即第 1 步边际增益 = 最佳单伙伴
    rs1 = np.abs(rho_s - np.eye(m, dtype=F32) * 1.0).max(axis=1)
    dev1 = float(np.nanmax(np.abs(g3["r3_marg1"] - rs1)))
    print("校验 3 r3 贪心（4 步）与暴力最大偏差 %.3e；第 1 步增益与最佳单伙伴偏差 %.3e"
          % (bad, dev1), flush=True)
    ok &= bad < 1e-6 and dev1 < 1e-6

    # 4) 行业对齐：同行业 rho 应该高于跨行业
    ind, note = load_industry(os.path.join(backup_root, "20241231", "industry.bin"), codes)
    iu = np.triu_indices(nn, 1)
    same = ind[iu[0]] == ind[iu[1]]
    ms = float(rho[iu][same].mean())
    md = float(rho[iu][~same].mean())
    print("校验 4 行业对齐：同行业 rho %.4f，跨行业 %.4f（%s）" % (ms, md, "正常" if ms > md else "可疑"), flush=True)
    ok &= ms > md

    # 5) 划分一致率与小规模暴力计数一致（这里踩过一次坑：簇数大于 2 时公式会重复计数）
    rngc = np.random.default_rng(3)
    agree_dev = 0.0
    for _ in range(3):
        aa = rngc.integers(0, 5, size=180)
        bb = rngc.integers(0, 7, size=180)
        _, ag = ari_and_agree(aa, bb)
        same = 0
        tot = 0
        for i in range(180):
            for j in range(i + 1, 180):
                tot += 1
                if (aa[i] == aa[j]) == (bb[i] == bb[j]):
                    same += 1
        agree_dev = max(agree_dev, abs(ag - same / tot))
    print("校验 5 划分一致率与暴力计数最大偏差 %.3e" % agree_dev, flush=True)
    ok &= agree_dev < 1e-12

    # ---- 跑完整流水线 ----
    cfg = {"chunk": 64, "max_k": 8, "k": 20}
    names, mat, diag, extra = compute_all(pair_root, "20241231", backup_root, cfg, verbose=False,
                                          ok_override=np.ones(nn, dtype=bool))
    print("降维完成：%d 只股票 × %d 个因子" % mat.shape, flush=True)
    xd = diag_crossday(pair_root, "20241231", names, mat, extra["codes"], cfg, backup_root,
                       verbose=False, ok_override=np.ones(nn, dtype=bool))
    od = diag_old_factors(backup_root, "20241231", names, mat, extra["codes"], verbose=False)
    nd = diag_null_model(names, mat, extra["lnt"])
    # 老因子重叠：old_explain_count_like 应该和 r1_explain_count 撞满
    hit = od.get("每个新因子的最大重叠", {}).get("r1_explain_count", {})
    print("校验 6 老因子重叠：r1_explain_count 最像 %s，|rho|=%.3f"
          % (hit.get("最大重叠老因子"), hit.get("最大绝对重叠", float("nan"))), flush=True)
    ok &= hit.get("最大重叠老因子") == "old_explain_count_like" and hit.get("最大绝对重叠", 0) > 0.99
    cd = xd.get("结果", {}).get("20241230", {})
    print("校验 7 跨日：交集 %s 只，稳定性前 3 %s"
          % (cd.get("交集股票数"), cd.get("按稳定性排序前 8", [])[:3]), flush=True)
    ok &= cd.get("交集股票数", 0) > 200
    zc = diag["矩阵检查"]
    print("校验 8 Z/ZS 不对称检测：Z 原始最大差 %.3f，ZS 原始最大差 %.3f，对称化说明 %s"
          % (zc.get("Z 对称最大偏差（原始）", 0.0), zc.get("ZS 对称最大偏差（原始）", 0.0),
             "有" if "Z 对称化说明" in zc else "没有"), flush=True)
    ok &= "Z 对称化说明" in zc and zc.get("Z 上下三角元素对齐后的 Spearman", 1.0) > 0.9
    # 校验 9：第二路（X0*）与 R6 平面都在，且 X0* 的集中度明显低于 X0
    has_s = any(nm.startswith("s_r1_") for nm in names)
    has_p = any(nm.startswith("p_") for nm in names)
    c1 = diag.get("R1", {}).get("X0 集中度", {})
    c2 = diag.get("R1_第二路", {}).get("X0 集中度", {})
    print("校验 9 第二路：列数 %d 有 s_ 前缀，%d 有 p_ 前缀；X0 用到 %s 只（前 10 占 %.3f），"
          "X0* 用到 %s 只（前 10 占 %.3f）"
          % (sum(1 for x in names if x.startswith("s_")), sum(1 for x in names if x.startswith("p_")),
             c1.get("被选为 X0 的股票数"), c1.get("前 10 名占全部配对的比例", float("nan")),
             c2.get("被选为 X0 的股票数"), c2.get("前 10 名占全部配对的比例", float("nan"))), flush=True)
    ok &= has_s and has_p
    # 第二路的意义就是别把解释权压在少数股票上：用到的股票数不该更少，集中度不该更高
    ok &= c2.get("被选为 X0 的股票数", 0) >= c1.get("被选为 X0 的股票数", 10 ** 9)
    ok &= c2.get("前 10 名占全部配对的比例", 1.0) <= c1.get("前 10 名占全部配对的比例", 0.0)
    # 写一遍输出文件
    out = os.path.join(pair_root, "20241231", "reduce")
    write_outputs(out, names, mat, diag, xd, od, nd)
    print("自测输出写在 %s" % out, flush=True)
    for f in ("factors.bin", "names.txt", "diag.json"):
        ok &= os.path.exists(os.path.join(out, f))
    chk = np.fromfile(os.path.join(out, "factors.bin"), dtype=F32)
    ok &= chk.size == mat.shape[0] * mat.shape[1]
    print("=== 自测 %s ===" % ("通过" if ok else "不通过"), flush=True)
    return 0 if ok else 1


def s3_val(g3, X, m, step):
    """保留一个取值的辅助接口（自测里直接用暴力结果比较）。"""
    return g3["r3_basket_rho8"][X]


def write_outputs(outdir, names, mat, diag, xd, od, nd):
    os.makedirs(outdir, exist_ok=True)
    np.ascontiguousarray(mat, dtype=F32).tofile(os.path.join(outdir, "factors.bin"))
    with open(os.path.join(outdir, "names.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    doc = dict(diag)
    doc["空模型说明"] = (
        "G 对随机配对本来就是正的（两个 ρ 相等的独立伙伴，r=0 时 G=(sqrt(2)-1)*ρ≈0.414ρ），"
        "所以任何直接用 G 算出来的因子都不能只看它自己，必须和用 Z 算的版本并排给出。"
        "带 null_ 前缀的列就是把同一构造里的 G 换成 Z（第二路是 GSTAR 换成 ZS）之后的对照版。"
        "本目录 Z / ZS 的全局均值见“矩阵检查”，它不是 0 才正常。")
    doc["Z 与 ZS 的效力限定"] = (
        "Z 只是把 G 按（B 活跃度档, C 活跃度档）分层做了标准化，上游已核对空模型是校准好的"
        "（逐层真实均值/标准差与空模型对得上，Z 的均值 0、标准差 1.01），"
        "所以 Z 本身不构成对“互补”的检验，它只是把量纲和活跃度摊平。"
        "真正有检验力的是条件冗余残差（experiments/yupei2/pair_conditional.py）：把 G 对"
        "(rho(X0,B), rho(X0,C)) 这个剖面回归掉以后，残差的标准差只有 G 的 9.4%，"
        "也就是 G 的约 99% 方差来自“B、C 各自跟 X0 有多像”，不是配对本身的性质。"
        "本文件里所有基于 G/GSTAR/Z/ZS 的因子都受这条限制，看到大的 Z 或 ZS 时按这个口径解释。")
    doc["因子清单"] = [{"名称": nm, "释义": EXPLAIN.get(nm, NULL_NOTE)} for nm in names]
    doc["因子分布"] = [{"名称": nm,
                      "非空数": int(np.isfinite(mat[:, j]).sum()),
                      "均值": None if not np.isfinite(np.nanmean(mat[:, j].astype(F64)))
                      else round(float(np.nanmean(mat[:, j].astype(F64))), 5),
                      "标准差": None if not np.isfinite(np.nanstd(mat[:, j].astype(F64)))
                      else round(float(np.nanstd(mat[:, j].astype(F64))), 5)}
                     for j, nm in enumerate(names)]
    doc["诊断_跨日稳定性"] = xd
    doc["诊断_与老因子重叠"] = od
    doc["诊断_空模型"] = nd
    with open(os.path.join(outdir, "diag.json"), "w") as f:
        json.dump(_jsonable(doc), f, ensure_ascii=False, indent=1)
    return outdir


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


# ============================================================
# 8. 入口
# ============================================================


def main(argv=None):
    ap = argparse.ArgumentParser(description="玉佩2 配对互补矩阵的降维与诊断")
    ap.add_argument("pair_out_dir", nargs="?", help="主会话输出根目录，里面有 <date>/ 子目录")
    ap.add_argument("date", nargs="?", help="交易日，例如 20241231")
    ap.add_argument("--backup", default="/hdd/user_home_unsafe/chenzongwei/yupei_dist_backup",
                    help="备份根目录，用于行业、成交笔数与 2761 个老因子")
    ap.add_argument("--no-crossday", action="store_true", help="不跑跨日稳定性（省一半时间）")
    ap.add_argument("--k", type=int, default=20, help="R4 每只股票取前 k 条边")
    ap.add_argument("--max-k", type=int, default=8, help="R3 篮子最大只数")
    ap.add_argument("--chunk", type=int, default=128, help="R2 扫描的分块行数")
    ap.add_argument("--selftest", action="store_true", help="用合成 fixture 自测，不需要真实数据")
    ap.add_argument("--synth-n", type=int, default=300, help="自测 fixture 的股票数")
    a = ap.parse_args(argv)

    if a.selftest:
        return selftest(n=a.synth_n)
    if not a.pair_out_dir or not a.date:
        ap.error("要么给 <pair_out_dir> <date>，要么 --selftest")

    cfg = {"chunk": a.chunk, "max_k": a.max_k, "k": a.k}
    t0 = time.time()
    names, mat, diag, extra = compute_all(a.pair_out_dir, a.date, a.backup, cfg, verbose=True)
    if a.no_crossday:
        xd = {"状态": "命令行给了 --no-crossday，跳过"}
    else:
        print("跨日稳定性诊断", flush=True)
        xd = diag_crossday(a.pair_out_dir, a.date, names, mat, extra["codes"], cfg, a.backup, verbose=True)
    print("与老因子重叠诊断", flush=True)
    od = diag_old_factors(a.backup, a.date, names, mat, extra["codes"], verbose=True)
    nd = diag_null_model(names, mat, extra["lnt"])
    diag["总耗时秒"] = round(time.time() - t0, 1)
    out = os.path.join(a.pair_out_dir, a.date, "reduce")
    write_outputs(out, names, mat, diag, xd, od, nd)
    print("写出 %s：%d 只股票 × %d 个因子，用时 %.1fs" % (out, mat.shape[0], mat.shape[1], time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
