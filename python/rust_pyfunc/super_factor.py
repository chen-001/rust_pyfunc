"""黑旋风四人团牌阵（Super-Factor）全历史一体筛选、组装与极速物化核心模块。

核心机制（定稿版）：
1. 全历史一体考察（零时段切分）：全程在完整交易日序列上统一评估，不切分季度或样本内外时段。
2. 零漏斗全量展开：每个有效底层原始因子直接展开为 50 个候选衍生列
   （原值/折叠 × {smooth_1, 固定日历窗 [5,10,20] × 4 统计量, 累积自由换手等权窗 [15%,30%,60%] × 4 统计量}）。
3. 双池分轨择优：固定日历窗轨（FixedCal）与换手率自适应窗轨（TurnEq）独立成团，禁止跨轨混编，收尾按 45 席同台择优（单轨上限 27）。
4. 团内两档离散权重 `{1.0, 0.5}`（最多 4 名成员：1 名主力 + 至多 3 名副手），默认优先 `1 : 1` 等权。
5. 默认采用定稿「融合版 B (`hybrid_adapt_icir`)」：超额 IC 自适应阈值锁定头部高 IC 尖刀（上限 8 把）+ 在线正交生长纯 RankICIR 稳健主力团。
"""
from __future__ import annotations

from collections import OrderedDict
import ctypes
import gc
import glob
import os
import re
import time
from typing import Iterable

import numpy as np


BARRA10_NAMES = [
    "size",
    "non_linear_size",
    "beta",
    "momentum",
    "residual_volatility",
    "liquidity",
    "book_to_price",
    "earnings_yield",
    "growth",
    "leverage",
]

SLOT_NAMES_25 = [
    "smooth_1",
    "mean_smooth_5", "max_smooth_5", "min_smooth_5", "std_smooth_5",
    "mean_smooth_10", "max_smooth_10", "min_smooth_10", "std_smooth_10",
    "mean_smooth_20", "max_smooth_20", "min_smooth_20", "std_smooth_20",
    "tceq_mean_15pct", "tceq_max_15pct", "tceq_min_15pct", "tceq_std_15pct",
    "tceq_mean_30pct", "tceq_max_30pct", "tceq_min_30pct", "tceq_std_30pct",
    "tceq_mean_60pct", "tceq_max_60pct", "tceq_min_60pct", "tceq_std_60pct",
]

SPAN_MAX_COLS = 1600
K_FINAL_SQUADS = 45
MAX_TRACK_QUOTA = 27
MAX_SQUAD_LEN = 4
MAX_RAW_REUSE_TRACK = 2
MAX_RAW_REUSE_FINAL = 3
INTER_MAX_CORR = 0.36
INTER_MEAN_CORR = 0.18
INTER_IC_CORR_HARD = 0.590
INTRA_MAX_CORR = 0.65
WEIGHT_TIERS = [1.0, 0.5]

MIN_REL_GAIN_SCAN = 0.003
MIN_REL_GAIN_ADD = 0.008
MIN_REL_GAIN_SWAP = 0.006
PREFER_EQ_REL = 0.012

K_TRACK_CANDS_CORE = 54
K_TRACK_CANDS_BAL = 54
K_TRACK_CANDS_PURE_IC = 24

_CACHED_CDLL: ctypes.CDLL | None = None


def _trim_malloc() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def ptr_f32(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_float)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def ptr_f64(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_double)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def ptr_u8(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_uint8)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def ptr_i8(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_int8)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def ptr_u16(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_uint16)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def ptr_i32(a: np.ndarray | None):
    if a is None or a.size == 0:
        return ctypes.POINTER(ctypes.c_int32)()
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _resolve_rust_pyfunc_shared_lib_path() -> str:
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    candidates: list[str] = []
    for pat in ("rust_pyfunc*.so", "rust_pyfunc*.pyd", "rust_pyfunc*.dylib"):
        candidates.extend(glob.glob(os.path.join(pkg_dir, pat)))
    if not candidates:
        raise FileNotFoundError(f"未在 {pkg_dir} 中找到 rust_pyfunc 动态链接库")
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def load_super_factor_cdll(n_threads: int | None = None) -> ctypes.CDLL:
    """加载并绑定 `rust_pyfunc` 内置的 14 个 `sf_*` 零拷贝 C-ABI 核函数。"""
    global _CACHED_CDLL
    if _CACHED_CDLL is None:
        lib_path = _resolve_rust_pyfunc_shared_lib_path()
        lib = ctypes.CDLL(lib_path)

        f32_p = ctypes.POINTER(ctypes.c_float)
        f32_pp = ctypes.POINTER(f32_p)
        f64_p = ctypes.POINTER(ctypes.c_double)
        u8_p = ctypes.POINTER(ctypes.c_uint8)
        i8_p = ctypes.POINTER(ctypes.c_int8)
        u16_p = ctypes.POINTER(ctypes.c_uint16)
        i32_p = ctypes.POINTER(ctypes.c_int32)
        sz = ctypes.c_size_t
        i32_v = ctypes.c_int32
        f64_v = ctypes.c_double

        lib.sf_init_threads.argtypes = [sz]
        lib.sf_init_threads.restype = None

        lib.sf_build_turnover_win_lens.argtypes = [f32_p, sz, sz, u8_p]
        lib.sf_build_turnover_win_lens.restype = None

        lib.sf_open_direct_store.argtypes = [ctypes.c_char_p, i32_p, sz, u8_p, sz, sz]
        lib.sf_open_direct_store.restype = ctypes.c_void_p

        lib.sf_close_direct_store.argtypes = [ctypes.c_void_p]
        lib.sf_close_direct_store.restype = None

        lib.sf_load_and_rank_store_cols_u16.argtypes = [ctypes.c_void_p, i32_p, sz, u16_p, u16_p]
        lib.sf_load_and_rank_store_cols_u16.restype = i32_v

        lib.sf_scan_store_double_buffered_top_u16.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, i32_p, sz, sz,
            u8_p, sz, sz, i32_v,
            i32_p, sz, u8_p, f32_p, f32_p,
            i32_p, sz, u8_p, f32_p, f32_p,
            i32_p, i32_p, sz, sz,
            f32_p, f32_p, f32_p,
            i8_p, f32_p, f32_p,
            i8_p, f32_p, f32_p,
            u16_p, u16_p,
            u16_p, u16_p, sz,
            u8_p, i32_p, f64_p,
        ]
        lib.sf_scan_store_double_buffered_top_u16.restype = i32_v

        lib.sf_materialize_faces_u16.argtypes = [
            u16_p, u16_p, sz, sz, sz,
            u8_p, i32_v, i32_p, sz,
            i32_p, sz, u8_p, f32_p, sz, f32_p,
            i32_p, sz, f32_p,
        ]
        lib.sf_materialize_faces_u16.restype = None

        lib.sf_eval_squad_candidates_dual_ptrs.argtypes = [
            f32_pp, f32_p, sz,
            f32_pp, f32_p, sz,
            sz, sz, sz,
            i32_p, sz, u8_p, f32_p, sz, sz, i32_v, f32_p,
            f32_p, f32_p,
        ]
        lib.sf_eval_squad_candidates_dual_ptrs.restype = None

        lib.sf_exhaustive_helper_scan_pure.argtypes = [
            f32_p, sz, sz, sz,
            i8_p, f32_p, f32_p, sz,
            i8_p, u8_p, i32_p, sz,
            f32_p, f32_p, f64_v,
            f32_p, sz,
            f64_v, f64_v, f64_v,
            i32_v, f64_v, f64_v,
            sz, i32_p, f32_p,
        ]
        lib.sf_exhaustive_helper_scan_pure.restype = sz

        lib.sf_filter_helper_cands.argtypes = [
            i32_p, sz,
            i8_p, f32_p, f32_p, i8_p, sz,
            f32_p, sz,
            f32_p, f64_v, f32_p,
            f32_p, f32_p, sz,
            f64_v, f64_v, f64_v,
            u8_p, u8_p,
        ]
        lib.sf_filter_helper_cands.restype = None

        lib.sf_combine_squads_from_uniq.argtypes = [
            f32_p, sz, sz, sz,
            i32_p, i32_p, f32_p, sz,
            f32_p,
        ]
        lib.sf_combine_squads_from_uniq.restype = None

        lib.sf_neutralize_block.argtypes = [
            f32_p, sz, sz, sz,
            u8_p, f32_p, sz,
            f32_p,
        ]
        lib.sf_neutralize_block.restype = None

        lib.sf_fill_post_composite.argtypes = [f32_p, sz, sz, sz]
        lib.sf_fill_post_composite.restype = None

        lib.sf_eval_tail_v4_exact_backtest.argtypes = [
            f32_p, sz, sz, sz,
            f32_p, f32_p, f32_p,
            i32_p, sz, sz, sz,
            f64_p, f64_p, f64_p, f64_p, f32_p, f32_p,
        ]
        lib.sf_eval_tail_v4_exact_backtest.restype = None

        _CACHED_CDLL = lib

    if n_threads is not None and n_threads > 0:
        _CACHED_CDLL.sf_init_threads(int(n_threads))
    return _CACHED_CDLL


def sanitize_lead_name(raw_name: str, max_ascii_len: int = 56) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", raw_name).strip("_")
    while s.endswith("_fold") or s.endswith("_smooth"):
        s = s[:-5].rstrip("_")
    s = s[:max_ascii_len].rstrip("_")
    while s.endswith("_fold") or s.endswith("_smooth"):
        s = s[:-5].rstrip("_")
    return s or "fac"


def decode_col_id(m: int, raw_names: list[str], sgn: int, weight: float = 1.0) -> dict:
    raw_idx = m // 50
    rem = m % 50
    var_id = rem // 25
    slot_id = rem % 25
    base_nm = raw_names[raw_idx]
    var_str = "_fold" if var_id == 1 else ""
    slot_nm = SLOT_NAMES_25[slot_id]
    win_family = "smooth_1" if slot_id == 0 else ("FixedCal" if slot_id < 13 else "TurnEq")
    return {
        "col_id": int(m),
        "raw_idx": int(raw_idx),
        "raw_name": base_nm,
        "var_id": int(var_id),
        "slot_id": int(slot_id),
        "sign": int(sgn),
        "weight": float(weight),
        "win_family": win_family,
        "derived_name": f"{base_nm}{var_str}_{slot_nm}",
    }


def _unit_fp(fp_i8: np.ndarray, fp_sc: np.ndarray, m: int, sgn: int) -> np.ndarray:
    return fp_i8[m].astype(np.float32) * (float(sgn) * float(fp_sc[m]))


def _make_pool_mask(m_total: int, pool_name: str) -> np.ndarray:
    slot_ids = (np.arange(m_total) % 50) % 25
    if pool_name == "FixedCal":
        return slot_ids < 13
    if pool_name == "TurnEq":
        return (slot_ids == 0) | (slot_ids >= 13)
    return np.ones(m_total, dtype=bool)


def _to_unit_ic_rows(mat_kt: np.ndarray) -> np.ndarray:
    if mat_kt.shape[0] == 0:
        return np.zeros_like(mat_kt, dtype=np.float32)
    fin = np.isfinite(mat_kt)
    cnt = np.maximum(fin.sum(axis=1, keepdims=True), 1)
    z = np.where(fin, mat_kt.astype(np.float32, copy=False), np.float32(0.0))
    mu = z.sum(axis=1, keepdims=True) / cnt
    c = np.where(fin, z - mu, np.float32(0.0))
    nrm = np.sqrt(np.maximum(np.sum(c * c, axis=1, keepdims=True), np.float32(1e-12)))
    return (c / nrm).astype(np.float32, copy=False)


def _to_unit_ic_1d(vec_t: np.ndarray) -> np.ndarray:
    fin = np.isfinite(vec_t)
    cnt = int(fin.sum())
    if cnt < 5:
        return np.zeros(len(vec_t), dtype=np.float32)
    z = np.where(fin, vec_t.astype(np.float32, copy=False), np.float32(0.0))
    mu = float(z.sum()) / cnt
    c = np.where(fin, z - np.float32(mu), np.float32(0.0))
    nrm = float(np.sqrt(max(float(np.dot(c, c)), 1e-12)))
    return (c / np.float32(nrm)).astype(np.float32, copy=False)


def _max_abs_ic_corr_vec_vs_mat(
    vec_1d: np.ndarray,
    mat_2d: np.ndarray,
    min_overlap_ratio: float = 0.5,
) -> float:
    """对齐交集与非零方差硬门禁检查的 Pearson IC 序列最大绝对相关系数。"""
    if mat_2d.shape[0] == 0:
        return 0.0
    t_tot = len(vec_1d)
    min_need = max(10, int(min_overlap_ratio * t_tot))
    fin1 = np.isfinite(vec_1d)
    if int(fin1.sum()) < min_need:
        return 1.0
    finm = np.isfinite(mat_2d)
    both = finm & fin1[None, :]
    n_common = both.sum(axis=1)
    if int(n_common.min()) < min_need:
        return 1.0
    x = np.where(both, mat_2d.astype(np.float64), 0.0)
    y = np.where(both, vec_1d.astype(np.float64)[None, :], 0.0)
    mx = x.sum(axis=1, keepdims=True) / n_common[:, None]
    my = y.sum(axis=1, keepdims=True) / n_common[:, None]
    xm = np.where(both, x - mx, 0.0)
    ym = np.where(both, y - my, 0.0)
    vx = np.sum(xm * xm, axis=1)
    vy = np.sum(ym * ym, axis=1)
    if np.any(vx <= 1e-12):
        raise RuntimeError("已选序列在交集上方差为0，无法计算相关系数")
    if np.any(vy <= 1e-12):
        return 1.0
    r = np.abs(np.sum(xm * ym, axis=1) / np.sqrt(vx * vy))
    return float(np.max(r))


def pairwise_ic_corr_matrix_strict(
    ic_mat_kt: np.ndarray,
    min_overlap_ratio: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """严格对齐有效交易日交集并检查两侧方差非零后计算 KxK IC 序列相关矩阵。"""
    k, t_tot = ic_mat_kt.shape
    min_need = max(10, int(min_overlap_ratio * t_tot))
    c_mat = np.eye(k, dtype=np.float64)
    arr = ic_mat_kt.astype(np.float64, copy=False)
    fin = np.isfinite(arr)
    for i in range(k):
        xi = arr[i]
        fi = fin[i]
        if i + 1 >= k:
            break
        both = fin[i + 1 :] & fi[None, :]
        n_c = both.sum(axis=1)
        if int(n_c.min()) < min_need:
            raise RuntimeError(f"IC矩阵配对交集不足: min_common={int(n_c.min())} < {min_need}")
        x_sub = np.where(both, arr[i + 1 :], 0.0)
        y_sub = np.where(both, xi[None, :], 0.0)
        mx = x_sub.sum(axis=1, keepdims=True) / n_c[:, None]
        my = y_sub.sum(axis=1, keepdims=True) / n_c[:, None]
        xm = np.where(both, x_sub - mx, 0.0)
        ym = np.where(both, y_sub - my, 0.0)
        vx = np.sum(xm * xm, axis=1)
        vy = np.sum(ym * ym, axis=1)
        if np.any(vx <= 1e-12) or np.any(vy <= 1e-12):
            raise RuntimeError("IC矩阵配对存在方差为0序列")
        r_vec = np.sum(xm * ym, axis=1) / np.sqrt(vx * vy)
        c_mat[i, i + 1 :] = r_vec
        c_mat[i + 1 :, i] = r_vec

    abs_c = np.abs(c_mat).copy()
    np.fill_diagonal(abs_c, 0.0)
    iu = np.triu_indices(k, k=1)
    abs_off = abs_c[iu]
    row_mx = abs_c.max(axis=1) if k >= 2 else np.zeros(k)
    info = {
        "mean_abs_ic_corr": float(np.mean(abs_off)) if len(abs_off) else 0.0,
        "p90_abs_ic_corr": float(np.percentile(abs_off, 90)) if len(abs_off) else 0.0,
        "median_max_abs_ic_corr": float(np.median(row_mx)) if len(row_mx) else 0.0,
        "max_abs_ic_corr": float(np.max(row_mx)) if len(row_mx) else 0.0,
        "count_max_gt_0p55": int(np.sum(row_mx > 0.55)),
        "count_max_gt_0p60": int(np.sum(row_mx > 0.6005)),
    }
    return c_mat, info


def compute_cs_corr_matrix_exact(neu_kt_n: np.ndarray, tradable: np.ndarray, n_k: int) -> np.ndarray:
    d_idxs = np.linspace(30, max(31, neu_kt_n.shape[1] - 10), 60, dtype=np.int64)
    corr_accum = np.zeros((n_k, n_k), dtype=np.float64)
    n_valid_d = 0
    for d in d_idxs:
        tr_ok = tradable[d] != 0
        if int(tr_ok.sum()) < 300:
            continue
        mat = neu_kt_n[:n_k, d, :][:, tr_ok].astype(np.float64)
        fin = np.isfinite(mat)
        cnt = fin.sum(axis=1)
        row_ok = cnt >= 200
        if int(row_ok.sum()) < max(2, n_k // 2):
            continue
        z = np.where(fin, mat, 0.0)
        means = z.sum(axis=1) / np.maximum(cnt, 1)
        centered = np.where(fin & row_ok[:, None], mat - means[:, None], 0.0)
        norms = np.sqrt(np.sum(centered * centered, axis=1))
        vn = norms > 1e-9
        unit = np.zeros_like(centered)
        unit[vn] = centered[vn] / norms[vn, None]
        c_day = unit @ unit.T
        np.fill_diagonal(c_day, 1.0)
        corr_accum += c_day
        n_valid_d += 1
    if n_valid_d == 0:
        return np.eye(n_k, dtype=np.float64)
    return corr_accum / n_valid_d


def _compute_orthogonality_45(neu_k47: np.ndarray, tradable: np.ndarray, n_sq: int) -> dict:
    d_idxs = np.arange(30, neu_k47.shape[1], 5)
    corr_accum = np.zeros((n_sq, n_sq), dtype=np.float64)
    n_valid_d = 0
    for d in d_idxs:
        tr_ok = tradable[d] != 0
        if tr_ok.sum() < 300:
            continue
        mat = neu_k47[:n_sq, d, :][:, tr_ok].astype(np.float64)
        fin = np.isfinite(mat)
        row_ok = fin.sum(axis=1) >= 200
        if row_ok.sum() < max(1, n_sq // 2):
            continue
        means = np.where(row_ok, np.nanmean(mat, axis=1), 0.0)
        centered = np.where(fin, mat - means[:, None], 0.0)
        norms = np.linalg.norm(centered, axis=1)
        vn = norms > 1e-9
        unit = np.zeros_like(centered)
        unit[vn] = centered[vn] / norms[vn, None]
        c_day = unit @ unit.T
        np.fill_diagonal(c_day, 1.0)
        corr_accum += c_day
        n_valid_d += 1
    if n_valid_d == 0 or n_sq < 2:
        return {
            "mean_abs_corr": 0.0,
            "p90_abs_corr": 0.0,
            "max_abs_corr": 0.0,
            "effective_independent_n": float(max(n_sq, 1)),
        }
    c_avg = corr_accum / n_valid_d
    iu = np.triu_indices(n_sq, k=1)
    abs_off = np.abs(c_avg[iu])
    eigvals = np.maximum(np.linalg.eigvalsh(c_avg), 0.0)
    s1 = float(np.sum(eigvals))
    s2 = float(np.sum(eigvals * eigvals))
    n_eff = (s1 * s1 / s2) if s2 > 1e-12 else 1.0
    return {
        "mean_abs_corr": float(np.mean(abs_off)),
        "p90_abs_corr": float(np.percentile(abs_off, 90)),
        "max_abs_corr": float(np.max(abs_off)),
        "effective_independent_n": float(n_eff),
    }


def _effective_n_from_corr(c_mat: np.ndarray) -> dict:
    if c_mat.shape[0] == 0:
        return {"neff_pr": 0.0, "neff_shannon": 0.0, "top1_share": 0.0, "pca_k90": 0}
    c = 0.5 * (c_mat + c_mat.T)
    np.fill_diagonal(c, 1.0)
    ev = np.maximum(np.linalg.eigvalsh(c)[::-1], 0.0)
    s1 = float(np.sum(ev))
    s2 = float(np.sum(ev * ev))
    neff = (s1 * s1 / s2) if s2 > 1e-12 else 1.0
    p = ev / max(s1, 1e-12)
    p_pos = p[p > 1e-15]
    shannon = float(np.exp(-np.sum(p_pos * np.log(p_pos))))
    k90 = int(np.searchsorted(np.cumsum(p), 0.90) + 1)
    return {
        "neff_pr": neff,
        "neff_shannon": shannon,
        "top1_share": float(p[0]),
        "pca_k90": k90,
    }


def _pack_bt_metrics(
    top_ret: np.ndarray,
    bot_ret: np.ndarray,
    mkt_ret: np.ndarray,
    idx_ret: np.ndarray,
    ic_nonoverlap: np.ndarray,
    gap: int,
    n_sq: int,
) -> dict:
    ir_sqrt = float(np.sqrt(50.0 if gap == 5 else 250.0))

    def _ret_m(r_1d: np.ndarray) -> tuple[float, float, float]:
        fin = r_1d[np.isfinite(r_1d)]
        if len(fin) < 5:
            return 0.0, 0.0, 0.0
        ann = float(np.mean(fin) * 250.0)
        sd = float(np.std(fin, ddof=1)) if len(fin) >= 2 else 0.0
        sh = float(np.mean(fin) / sd * np.sqrt(250.0)) if sd > 1e-12 else 0.0
        cum = np.cumsum(fin)
        peak = np.maximum.accumulate(np.insert(cum, 0, 0.0))[1:]
        mdd = float(np.max(peak - cum)) if len(cum) > 0 else 0.0
        return ann, sh, mdd

    def _ic_m(ic_1d: np.ndarray) -> tuple[float, float, float]:
        fin = ic_1d[np.isfinite(ic_1d)].astype(np.float64)
        if len(fin) < 3:
            return 0.0, 0.0, 0.0
        mu = float(np.mean(fin))
        sd = float(np.std(fin, ddof=0))
        ir = float((mu / sd) * ir_sqrt) if sd > 1e-12 else 0.0
        pos = float(np.mean(fin > 0.0))
        return mu, ir, pos

    ics, irs, poss = [], [], []
    exa, exs, exm = [], [], []
    hda, hds, hdm = [], [], []
    lsa, lss, lsm = [], [], []

    for i in range(n_sq):
        mi, ri, pi = _ic_m(ic_nonoverlap[i])
        ics.append(mi)
        irs.append(ri)
        poss.append(pi)
        ea, es, em = _ret_m(top_ret[i] - mkt_ret[i])
        ha, hs, hm = _ret_m(top_ret[i] - idx_ret)
        la, ls_, lm = _ret_m(top_ret[i] - bot_ret[i])
        exa.append(ea)
        exs.append(es)
        exm.append(em)
        hda.append(ha)
        hds.append(hs)
        hdm.append(hm)
        lsa.append(la)
        lss.append(ls_)
        lsm.append(lm)

    def _single(col_i: int) -> dict:
        mi, ri, pi = _ic_m(ic_nonoverlap[col_i])
        ea, es, em = _ret_m(top_ret[col_i] - mkt_ret[col_i])
        ha, hs, hm = _ret_m(top_ret[col_i] - idx_ret)
        la, ls_, lm = _ret_m(top_ret[col_i] - bot_ret[col_i])
        return {
            "ic_mean": mi,
            "icir": ri,
            "ic_pos_rate": pi,
            "top10_excess_mkt_ann": ea,
            "top10_excess_mkt_sharpe": es,
            "top10_excess_mkt_mdd": em,
            "hedge_csi1000_ann": ha,
            "hedge_csi1000_sharpe": hs,
            "hedge_csi1000_mdd": hm,
            "ls_ann": la,
            "ls_sharpe": ls_,
            "ls_mdd": lm,
        }

    return {
        "standalone45_mean": {
            "ic_mean": float(np.mean(ics)) if ics else 0.0,
            "median_ic": float(np.median(ics)) if ics else 0.0,
            "icir": float(np.mean(irs)) if irs else 0.0,
            "median_icir": float(np.median(irs)) if irs else 0.0,
            "ic_pos_rate": float(np.mean(poss)) if poss else 0.0,
            "top10_excess_mkt_ann": float(np.mean(exa)) if exa else 0.0,
            "top10_excess_mkt_sharpe": float(np.mean(exs)) if exs else 0.0,
            "top10_excess_mkt_mdd": float(np.mean(exm)) if exm else 0.0,
            "hedge_csi1000_ann": float(np.mean(hda)) if hda else 0.0,
            "hedge_csi1000_sharpe": float(np.mean(hds)) if hds else 0.0,
            "hedge_csi1000_mdd": float(np.mean(hdm)) if hdm else 0.0,
            "ls_ann": float(np.mean(lsa)) if lsa else 0.0,
            "ls_sharpe": float(np.mean(lss)) if lss else 0.0,
            "ls_mdd": float(np.mean(lsm)) if lsm else 0.0,
        },
        "composite45_post_neu": _single(n_sq + 1),
        "composite45_pre_neu": _single(n_sq),
        "per_factor_ic": [round(float(x), 6) for x in ics],
        "per_factor_icir": [round(float(x), 3) for x in irs],
        "per_factor_hedge_ann": [round(float(x), 5) for x in hda],
    }


class InRamRawU16Bank:
    """零落盘内存预分配 `(raw_u16, fold_u16)` 双秩矩阵缓存池。"""

    def __init__(
        self,
        lib: ctypes.CDLL,
        store_handle: int,
        valid_store_cols: np.ndarray,
        t_len: int,
        n_stocks: int,
        max_raw: int = 1200,
    ):
        self.lib = lib
        self.store_handle = store_handle
        self.valid_store_cols = valid_store_cols
        self.t_len = t_len
        self.n_stocks = n_stocks
        self.max_raw = min(max_raw, max(16, len(valid_store_cols)))
        self.raw_slab = np.empty((self.max_raw, t_len, n_stocks), dtype=np.uint16)
        self.fold_slab = np.empty((self.max_raw, t_len, n_stocks), dtype=np.uint16)
        self.free_slots: list[int] = list(range(self.max_raw - 1, -1, -1))
        self.cache: OrderedDict[int, int] = OrderedDict()

    def clear(self) -> None:
        self.cache.clear()
        self.free_slots.clear()
        self.raw_slab = None
        self.fold_slab = None

    def _materialize_chunk(
        self, specs: np.ndarray, sub_raw_rows: np.ndarray, win_len_turn: np.ndarray
    ) -> np.ndarray:
        l_cnt = len(specs)
        n_sub = len(sub_raw_rows)
        needed_set: set[int] = set()
        missing_raws: list[int] = []

        for r_idx_v in specs[:, 0]:
            r_idx = int(r_idx_v)
            if r_idx not in needed_set:
                needed_set.add(r_idx)
                if r_idx in self.cache:
                    self.cache.move_to_end(r_idx)
                else:
                    missing_raws.append(r_idx)

        if missing_raws:
            m_cnt = len(missing_raws)
            st_cols = np.ascontiguousarray(
                self.valid_store_cols[np.asarray(missing_raws, dtype=np.int32)],
                dtype=np.int32,
            )
            ru_batch = np.empty((m_cnt, self.t_len, self.n_stocks), dtype=np.uint16)
            fu_batch = np.empty((m_cnt, self.t_len, self.n_stocks), dtype=np.uint16)
            rc = self.lib.sf_load_and_rank_store_cols_u16(
                ctypes.c_void_p(self.store_handle),
                ptr_i32(st_cols),
                m_cnt,
                ptr_u16(ru_batch),
                ptr_u16(fu_batch),
            )
            if rc != 0:
                raise RuntimeError(f"sf_load_and_rank_store_cols_u16 失败 rc={rc}")
            for i_m, r_idx in enumerate(missing_raws):
                if self.free_slots:
                    slot = self.free_slots.pop()
                else:
                    slot = -1
                    for k_old in list(self.cache.keys()):
                        if k_old not in needed_set:
                            slot = self.cache.pop(k_old)
                            break
                    if slot < 0:
                        _, slot = self.cache.popitem(last=False)
                self.raw_slab[slot] = ru_batch[i_m]
                self.fold_slab[slot] = fu_batch[i_m]
                self.cache[r_idx] = slot
            del ru_batch, fu_batch

        local_specs = specs.copy()
        for i_s in range(l_cnt):
            local_specs[i_s, 0] = self.cache[int(specs[i_s, 0])]

        out_u = np.empty((l_cnt, n_sub, self.n_stocks), dtype=np.float32)
        self.lib.sf_materialize_faces_u16(
            ptr_u16(self.raw_slab),
            ptr_u16(self.fold_slab),
            self.max_raw,
            self.t_len,
            self.n_stocks,
            ptr_u8(win_len_turn),
            0,
            ptr_i32(sub_raw_rows),
            n_sub,
            None,
            0,
            None,
            None,
            0,
            None,
            ptr_i32(local_specs),
            l_cnt,
            ptr_f32(out_u),
        )
        return out_u

    def materialize_specs(
        self, specs: np.ndarray, sub_raw_rows: np.ndarray, win_len_turn: np.ndarray
    ) -> np.ndarray:
        uniq_r = np.unique(specs[:, 0])
        if len(uniq_r) <= self.max_raw:
            return self._materialize_chunk(specs, sub_raw_rows, win_len_turn)
        out_u = np.empty((len(specs), len(sub_raw_rows), self.n_stocks), dtype=np.float32)
        step = max(1, self.max_raw)
        for st in range(0, len(uniq_r), step):
            grp_set = set(int(x) for x in uniq_r[st : st + step])
            idx = np.array([i for i, r in enumerate(specs[:, 0]) if int(r) in grp_set], dtype=np.int32)
            out_u[idx] = self._materialize_chunk(specs[idx], sub_raw_rows, win_len_turn)
        return out_u


class VerifFaceCache:
    """成团精选阶段 LRU 截面标准化衍生面缓存。"""

    def __init__(
        self,
        lib: ctypes.CDLL,
        scan: dict,
        ctx: dict,
        sub_eval_idx: np.ndarray,
        sig_rows_all: np.ndarray,
        max_items: int = 5000,
    ):
        self.lib = lib
        self.scan = scan
        self.ctx = ctx
        self.sub_eval_idx = np.ascontiguousarray(sub_eval_idx, dtype=np.int32)
        self.sub_raw_rows = np.ascontiguousarray(sig_rows_all[sub_eval_idx], dtype=np.int32)
        self.n_sub = len(sub_eval_idx)
        self.n_stocks = ctx["n_stocks"]
        self.max_items = max_items
        self.cache: OrderedDict[tuple[int, int], np.ndarray] = OrderedDict()

    def get_many(self, items: list[tuple[int, int]]) -> list[np.ndarray]:
        missing: list[tuple[int, int]] = []
        seen_miss: set[tuple[int, int]] = set()
        for key in items:
            if key in self.cache:
                self.cache.move_to_end(key)
            elif key not in seen_miss:
                missing.append(key)
                seen_miss.add(key)

        if missing:
            l_cnt = len(missing)
            specs = np.empty((l_cnt, 4), dtype=np.int32)
            for i, (m, sgn) in enumerate(missing):
                raw_idx = m // 50
                rem = m % 50
                specs[i, 0] = raw_idx
                specs[i, 1] = rem // 25
                specs[i, 2] = rem % 25
                specs[i, 3] = sgn
            out_u = self.scan["u16_bank"].materialize_specs(specs, self.sub_raw_rows, self.ctx["win_len_turn"])
            for i, key in enumerate(missing):
                while len(self.cache) >= self.max_items:
                    self.cache.popitem(last=False)
                self.cache[key] = out_u[i].copy()

        return [self.cache[key] for key in items]


def _eval_obj_scalar(mu: float, ir: float, objective_mode: int) -> float:
    if mu <= 0.0 or ir <= 0.0:
        return -1e9
    if objective_mode == 0:
        return float(ir)
    if objective_mode == 1:
        return float(mu)
    return float(np.sqrt(mu * ir))


def compute_pure_score_full_1d(
    ic_1d: np.ndarray,
    ir_ann_sqrt: float,
    objective_mode: int,
) -> tuple[float, float, float, bool]:
    fin = np.isfinite(ic_1d)
    cnt = int(fin.sum())
    if cnt < max(20, len(ic_1d) // 2):
        return -1e9, 0.0, 0.0, False
    vals = ic_1d[fin].astype(np.float64)
    mu = float(np.mean(vals))
    if mu <= 0.0:
        return -1e9, mu, 0.0, False
    sd = max(float(np.std(vals, ddof=0)), 1e-6)
    ir = float((mu / sd) * ir_ann_sqrt)
    sc = _eval_obj_scalar(mu, ir, objective_mode)
    return sc, mu, ir, (sc > 0.0)


def build_fullhist_objective_view(scan: dict, gap: int, objective_mode: int) -> dict:
    """构建全历史一体 10-Barra 评价视图（`0`=纯RankICIR, `1`=纯RankIC, `2`=几何折中）。"""
    if gap == 5:
        ic_b10 = scan["ic5_b10"]
        vdays = scan["vdays_g5"]
        ir_sqrt = float(np.sqrt(50.0))
        mu_min_col = 0.005
        mu_floor_scan = 0.005
    else:
        ic_b10 = scan["ic1_b10"]
        vdays = scan["vdays_g1"]
        if ic_b10 is None or vdays is None:
            raise ValueError("当前 scan 未包含 gap=1 扫描结果，请开启 gaps=(5, 1)")
        ir_sqrt = float(np.sqrt(250.0))
        mu_min_col = 0.0025
        mu_floor_scan = 0.0025

    n_T = ic_b10.shape[1]
    fin = np.isfinite(ic_b10)
    cnt = np.maximum(fin.sum(axis=1), 1)
    z = np.where(fin, ic_b10, 0.0)
    mu10 = np.sum(z, axis=1, dtype=np.float64) / cnt
    sgn10 = np.where(mu10 >= 0.0, 1, -1).astype(np.int8)
    mua10 = np.abs(mu10)
    sq = np.sum(z.astype(np.float32) ** 2, axis=1, dtype=np.float64)
    var_raw = np.maximum(sq / cnt - mu10 * mu10, 0.0)
    sd10 = np.maximum(np.sqrt(var_raw), 1e-6)
    ir10 = (mua10 / sd10) * ir_sqrt

    fp_i8 = scan["fp_b10_i8"]
    fp_sc = scan["fp_b10_sc"]
    res_sc = scan["res_b10_sc"]

    valid_col = (vdays >= int(0.60 * n_T)) & (fp_sc > 0.0) & (var_raw > 1e-12) & (mua10 >= mu_min_col)

    if objective_mode == 0:
        raw_obj = ir10
        obj_name = "Pure_RankICIR"
    elif objective_mode == 1:
        raw_obj = mua10
        obj_name = "Pure_RankIC"
    else:
        raw_obj = np.sqrt(np.maximum(mua10 * ir10, 0.0))
        obj_name = "Balanced_GeomIC_ICIR"

    col_obj_clean = np.where(valid_col, raw_obj, -1e9).astype(np.float64)
    return {
        "gap": gap,
        "mode": "10barra",
        "objective_mode": objective_mode,
        "objective_name": obj_name,
        "n_T": n_T,
        "ir_sqrt": ir_sqrt,
        "mu_floor_scan": mu_floor_scan,
        "ic_target": ic_b10,
        "fp_i8": fp_i8,
        "fp_sc": fp_sc,
        "res_sc": res_sc,
        "vdays": vdays,
        "col_sgn": np.ascontiguousarray(sgn10),
        "valid_col": valid_col,
        "col_obj_clean": col_obj_clean,
        "sgn_10": sgn10,
    }


def _build_raw_dedup_order(scores: np.ndarray, max_per_raw: int = 2) -> np.ndarray:
    n_raw = len(scores) // 50
    s2d = scores.reshape(n_raw, 50)
    top_local = np.argsort(-s2d, axis=1)[:, :max_per_raw]
    base_off = (np.arange(n_raw, dtype=np.int64) * 50)[:, None]
    cand_cols = (base_off + top_local).reshape(-1)
    cand_sc = scores[cand_cols]
    valid = cand_sc > -1e8
    cand_cols = cand_cols[valid]
    cand_sc = cand_sc[valid]
    return cand_cols[np.argsort(-cand_sc)]


def deduplicate_tagged_cands(
    cands_list: list[tuple[str, list[tuple[int, int, float]]]],
) -> list[tuple[str, list[tuple[int, int, float]]]]:
    out: list[tuple[str, list[tuple[int, int, float]]]] = []
    seen: set[frozenset[tuple[int, int, int]]] = set()
    for trk, sq in cands_list:
        sig = frozenset((m, sgn, int(round(w * 100))) for m, sgn, w in sq)
        if sig not in seen:
            seen.add(sig)
            out.append((trk, sq))
    return out


def stream_scan_super_store(
    lib: ctypes.CDLL,
    ctx: dict,
    disp: str,
    store_dir: str,
    raw_names: list[str],
    valid_store_cols: np.ndarray,
    need_gap1: bool = False,
    slab_cap: int = 4500,
    must_keep_mask: np.ndarray | None = None,
) -> dict:
    """零落盘双缓冲扫描整个 `factor_store`，同时把最优 `slab_cap` 个原始因子 `u16` 秩矩阵驻留内存。"""
    n_raw = len(raw_names)
    m_total = n_raw * 50
    t_len = ctx["t_len"]
    n_stocks = ctx["n_stocks"]
    n_g5 = ctx["n_g5"]
    n_g1 = ctx["n_g1"]
    d_fp = ctx["d_fp"]

    must_keep = (
        np.ascontiguousarray(must_keep_mask, dtype=np.uint8)
        if must_keep_mask is not None
        else np.zeros(n_raw, dtype=np.uint8)
    )
    eff_slab_cap = min(int(slab_cap), max(16, n_raw))

    store_handle = lib.sf_open_direct_store(
        store_dir.encode("utf-8"),
        ptr_i32(ctx["dates"]),
        t_len,
        ptr_u8(ctx["stocks_flat"]),
        ctx["stock_stride"],
        n_stocks,
    )
    if not store_handle:
        raise RuntimeError(f"sf_open_direct_store 打开失败: {store_dir}")

    ic5_b10 = np.empty((m_total, n_g5), dtype=np.float32)
    ic1_b10 = np.empty((m_total, n_g1), dtype=np.float32) if need_gap1 else None
    fp_b10_i8 = np.empty((m_total, d_fp), dtype=np.int8)
    fp_b10_sc = np.empty(m_total, dtype=np.float32)
    res_b10_sc = np.empty(m_total, dtype=np.float32)
    vdays_g5 = np.empty(m_total, dtype=np.uint16)
    vdays_g1 = np.empty(m_total, dtype=np.uint16) if need_gap1 else None

    u16_bank = InRamRawU16Bank(lib, store_handle, valid_store_cols, t_len, n_stocks, max_raw=eff_slab_cap)
    slab_raw_ids = np.full(eff_slab_cap, -1, dtype=np.int32)
    slab_prios = np.full(eff_slab_cap, -1e18, dtype=np.float64)

    used_slots = lib.sf_scan_store_double_buffered_top_u16(
        ctypes.c_void_p(store_handle),
        disp.encode("utf-8"),
        ptr_i32(valid_store_cols),
        n_raw,
        SPAN_MAX_COLS,
        ptr_u8(ctx["win_len_turn"]),
        ctx["n_b10"],
        ctx["n_c2"],
        0,
        ptr_i32(ctx["g1_rows"]),
        n_g1,
        ptr_u8(ctx["ev_g1"]),
        ptr_f32(ctx["q14_g1"]),
        ptr_f32(ctx["ret1_ord"]),
        ptr_i32(ctx["g5_from_g1"]),
        n_g5,
        ptr_u8(ctx["ev_g5"]),
        ptr_f32(ctx["q14_g5"]),
        ptr_f32(ctx["ret5_ord"]),
        ptr_i32(ctx["fp_g1_map"]),
        ptr_i32(ctx["fp_stock_idx"]),
        ctx["n_fp_dates"],
        ctx["fp_k_per_day"],
        ptr_f32(ic5_b10),
        None,
        ptr_f32(ic1_b10) if ic1_b10 is not None else None,
        ptr_i8(fp_b10_i8),
        ptr_f32(fp_b10_sc),
        ptr_f32(res_b10_sc),
        None,
        None,
        None,
        ptr_u16(vdays_g5),
        ptr_u16(vdays_g1) if vdays_g1 is not None else None,
        ptr_u16(u16_bank.raw_slab),
        ptr_u16(u16_bank.fold_slab),
        eff_slab_cap,
        ptr_u8(must_keep),
        ptr_i32(slab_raw_ids),
        ptr_f64(slab_prios),
    )
    if used_slots < 0:
        lib.sf_close_direct_store(ctypes.c_void_p(store_handle))
        raise RuntimeError(f"sf_scan_store_double_buffered_top_u16 异常退出 rc={used_slots}")

    valid_sl = [s for s in range(used_slots) if int(slab_raw_ids[s]) >= 0]
    valid_sl.sort(key=lambda s: float(slab_prios[s]))
    u16_bank.cache = OrderedDict((int(slab_raw_ids[s]), s) for s in valid_sl)
    used_set = set(valid_sl)
    u16_bank.free_slots = [s for s in range(eff_slab_cap - 1, -1, -1) if s not in used_set]

    return {
        "store_handle": store_handle,
        "u16_bank": u16_bank,
        "raw_names": raw_names,
        "valid_store_cols": valid_store_cols,
        "n_raw": n_raw,
        "m_total": m_total,
        "ic5_b10": ic5_b10,
        "ic1_b10": ic1_b10,
        "fp_b10_i8": fp_b10_i8,
        "fp_b10_sc": fp_b10_sc,
        "res_b10_sc": res_b10_sc,
        "vdays_g5": vdays_g5,
        "vdays_g1": vdays_g1,
    }


def assemble_track_with_optional_prelocked(
    lib: ctypes.CDLL,
    scan: dict,
    ctx: dict,
    view: dict,
    face_cache: VerifFaceCache,
    ev_mat: np.ndarray,
    q14_mat: np.ndarray,
    ret_ord_mat: np.ndarray,
    n_eval_full: int,
    track_name: str,
    k_target: int,
    prelocked_squads: list[list[tuple[int, int, float]]] | None = None,
) -> list[tuple[str, list[tuple[int, int, float]]]]:
    """逐团生长锁定（Construct-and-Lock）单轨超因子组装核，支持预先占位的尖刀团正交避让。"""
    m_total = scan["m_total"]
    d_fp = ctx["d_fp"]
    n_T = view["n_T"]
    ir_sqrt = view["ir_sqrt"]
    obj_mode = view["objective_mode"]
    fp_i8 = view["fp_i8"]
    fp_sc = view["fp_sc"]
    res_sc = view["res_sc"]
    col_sgn = view["col_sgn"]
    weights_arr = np.asarray(WEIGHT_TIERS, dtype=np.float32)
    verif_ir_sqrt = (
        float(np.sqrt(250.0 / max(1, round(n_eval_full / max(1, face_cache.n_sub)))))
        if view["gap"] == 1
        else ir_sqrt
    )

    pool_mask = _make_pool_mask(m_total, track_name)
    seed_scores = np.where(pool_mask & view["valid_col"], view["col_obj_clean"], -1e9)
    order = _build_raw_dedup_order(seed_scores, max_per_raw=2)
    seed_cand_uics = _to_unit_ic_rows(view["ic_target"][order] * col_sgn[order, None].astype(np.float32))

    base_valid_u8 = np.ascontiguousarray((pool_mask & view["valid_col"]).astype(np.uint8))
    top_k_scan = min(1500, scan["n_raw"])
    out_top_idx = np.empty(top_k_scan, dtype=np.int32)
    out_top_score = np.empty(top_k_scan, dtype=np.float32)
    f32_p_t = ctypes.POINTER(ctypes.c_float)

    def _make_ptr_array(arrs: list[np.ndarray]):
        n_a = len(arrs)
        if n_a == 0:
            return ctypes.POINTER(f32_p_t)()
        c_arr = (f32_p_t * n_a)()
        for idx_a, a in enumerate(arrs):
            c_arr[idx_a] = a.ctypes.data_as(f32_p_t)
        return c_arr

    def _eval_verif_1d(ic10_row: np.ndarray) -> tuple[float, bool, np.ndarray]:
        sc, mu_t, _, ok_t = compute_pure_score_full_1d(ic10_row, verif_ir_sqrt, obj_mode)
        ok = ok_t and (mu_t > 0.0) and (sc > 0.0)
        return (sc if ok else -1e9), ok, ic10_row

    def _build_squad_ic_and_fp(members: list[tuple[int, int, float]]):
        sq_ic = np.zeros(n_T, dtype=np.float32)
        sq_fp = np.zeros(d_fp, dtype=np.float32)
        any_fin = np.zeros(n_T, dtype=bool)
        for m_i, s_i, w_i in members:
            w_eff = float(w_i) * float(res_sc[m_i]) * float(s_i)
            ic_row = view["ic_target"][m_i]
            fin_r = np.isfinite(ic_row)
            any_fin |= fin_r
            sq_ic += np.where(fin_r, ic_row * w_eff, 0.0).astype(np.float32)
            sq_fp += float(w_i) * float(res_sc[m_i]) * _unit_fp(fp_i8, fp_sc, m_i, s_i)
        nm = float(np.linalg.norm(sq_fp))
        ufp = (sq_fp / nm) if nm > 1e-8 else np.zeros(d_fp, dtype=np.float32)
        sq_ic_clean = np.where(any_fin, sq_ic, np.float32(np.nan))
        return sq_ic, sq_fp, float(np.dot(sq_fp, sq_fp)), sq_ic_clean, ufp

    init_list = list(prelocked_squads) if prelocked_squads else []
    n_init = len(init_list)
    total_cap = n_init + k_target

    locked_squads: list[list[tuple[int, int, float]]] = []
    locked_fps = np.zeros((total_cap, d_fp), dtype=np.float32)
    locked_proxy_ics = np.zeros((total_cap, n_T), dtype=np.float32)
    locked_proxy_uics = np.zeros((total_cap, n_T), dtype=np.float32)
    locked_verif_ics = np.zeros((total_cap, face_cache.n_sub), dtype=np.float32)
    raw_counts: dict[int, int] = {}
    used_cols: set[int] = set()

    for idx_pre, sq_pre in enumerate(init_list):
        _, _, _, sq_ic_clean, ufp_pre = _build_squad_ic_and_fp(sq_pre)
        pre_faces = face_cache.get_many([(m_b, s_b) for m_b, s_b, _ in sq_pre])
        base_ptrs = _make_ptr_array(pre_faces)
        base_w_arr = np.asarray([w_b for _, _, w_b in sq_pre], dtype=np.float32)
        zero_face = [np.zeros_like(pre_faces[0])]
        zero_ptrs = _make_ptr_array(zero_face)
        zero_w = np.asarray([0.0], dtype=np.float32)
        vic10_pre = np.empty((1, face_cache.n_sub), dtype=np.float32)
        vicc2_pre = np.empty((1, face_cache.n_sub), dtype=np.float32)
        lib.sf_eval_squad_candidates_dual_ptrs(
            base_ptrs, ptr_f32(base_w_arr), len(sq_pre),
            zero_ptrs, ptr_f32(zero_w), max(1, (len(sq_pre) + 1) // 2),
            1, face_cache.n_sub, ctx["n_stocks"],
            ptr_i32(face_cache.sub_eval_idx), n_eval_full,
            ptr_u8(ev_mat), ptr_f32(q14_mat), ctx["n_b10"], ctx["n_c2"], 0,
            ptr_f32(ret_ord_mat), ptr_f32(vic10_pre), ptr_f32(vicc2_pre),
        )
        locked_squads.append(sq_pre)
        locked_fps[idx_pre] = ufp_pre
        locked_proxy_ics[idx_pre] = sq_ic_clean
        locked_proxy_uics[idx_pre] = _to_unit_ic_1d(sq_ic_clean)
        locked_verif_ics[idx_pre] = vic10_pre[0]
        for m_p, _, _ in sq_pre:
            used_cols.add(m_p)
            raw_counts[m_p // 50] = raw_counts.get(m_p // 50, 0) + 1

    def _search_best_addition_fullhist(
        n_locked: int,
        base_members: list[tuple[int, int, float]],
        cur_score: float,
        top_exact_m: int,
        min_rel_gain: float,
    ):
        sq_ic, sq_fp, sq_norm_sq, sq_ic_clean, _ = _build_squad_ic_and_fp(base_members)
        in_squad_raws = {m_i // 50 for m_i, _, _ in base_members}
        blocked_raws = [r for r, c in raw_counts.items() if c >= MAX_RAW_REUSE_TRACK]
        blocked_raws.extend(in_squad_raws)
        blocked_arr = np.ascontiguousarray(np.unique(np.asarray(blocked_raws, dtype=np.int32))) if blocked_raws else None

        n_found = int(
            lib.sf_exhaustive_helper_scan_pure(
                ptr_f32(view["ic_target"]),
                m_total,
                n_T,
                n_T,
                ptr_i8(fp_i8),
                ptr_f32(fp_sc),
                ptr_f32(res_sc),
                d_fp,
                ptr_i8(col_sgn),
                ptr_u8(base_valid_u8),
                ptr_i32(blocked_arr),
                0 if blocked_arr is None else len(blocked_arr),
                ptr_f32(sq_ic),
                ptr_f32(sq_fp),
                sq_norm_sq,
                ptr_f32(weights_arr),
                len(WEIGHT_TIERS),
                INTRA_MAX_CORR,
                ir_sqrt,
                view["mu_floor_scan"],
                obj_mode,
                MIN_REL_GAIN_SCAN,
                PREFER_EQ_REL,
                top_k_scan,
                ptr_i32(out_top_idx),
                ptr_f32(out_top_score),
            )
        )
        if n_found == 0:
            return None
        part_idx = np.ascontiguousarray(out_top_idx[:n_found], dtype=np.int32)
        other_fps = np.ascontiguousarray(locked_fps[:n_locked], dtype=np.float32) if n_locked > 0 else None
        other_proxy_uics = np.ascontiguousarray(locked_proxy_uics[:n_locked], dtype=np.float32) if n_locked > 0 else None
        other_verif_ics = locked_verif_ics[:n_locked]

        out_ok1 = np.zeros(n_found, dtype=np.uint8)
        out_ok05 = np.zeros(n_found, dtype=np.uint8)
        lib.sf_filter_helper_cands(
            ptr_i32(part_idx),
            n_found,
            ptr_i8(fp_i8),
            ptr_f32(fp_sc),
            ptr_f32(res_sc),
            ptr_i8(col_sgn),
            d_fp,
            ptr_f32(view["ic_target"]),
            n_T,
            ptr_f32(sq_fp),
            float(sq_norm_sq),
            ptr_f32(sq_ic_clean),
            ptr_f32(other_fps),
            ptr_f32(other_proxy_uics),
            n_locked,
            float(INTER_MAX_CORR),
            float(INTER_MEAN_CORR),
            0.59,
            ptr_u8(out_ok1),
            ptr_u8(out_ok05),
        )

        acc_pos = np.flatnonzero((out_ok1 | out_ok05) != 0)
        cand_cols: list[tuple[int, int, np.ndarray, np.ndarray]] = []
        for idx_p in acc_pos:
            m_c = int(part_idx[idx_p])
            if m_c in used_cols:
                continue
            sgn_c = int(col_sgn[m_c])
            u_c = _unit_fp(fp_i8, fp_sc, m_c, sgn_c)
            sig_c = float(res_sc[m_c])
            fp1 = sq_fp + 1.0 * sig_c * u_c
            fp05 = sq_fp + 0.5 * sig_c * u_c
            nm1 = float(np.linalg.norm(fp1))
            nm05 = float(np.linalg.norm(fp05))
            if nm1 <= 1e-8 or nm05 <= 1e-8:
                continue
            ok1 = bool(out_ok1[idx_p] != 0)
            ok05 = bool(out_ok05[idx_p] != 0)
            cand_cols.append((m_c, sgn_c, (fp1 / nm1) if ok1 else np.zeros(0), (fp05 / nm05) if ok05 else np.zeros(0)))
            if len(cand_cols) >= top_exact_m:
                break

        if not cand_cols:
            return None

        base_faces = face_cache.get_many([(m_b, s_b) for m_b, s_b, _ in base_members])
        col_faces = face_cache.get_many([(m_c, sgn_c) for m_c, sgn_c, _, _ in cand_cols])
        base_ptrs = _make_ptr_array(base_faces)
        base_w_arr = np.asarray([w_b for _, _, w_b in base_members], dtype=np.float32)

        eval_faces = []
        eval_w = []
        for fi in col_faces:
            eval_faces.append(fi)
            eval_w.append(1.0)
            eval_faces.append(fi)
            eval_w.append(0.5)

        k_fin = len(eval_faces)
        cand_ptrs = _make_ptr_array(eval_faces)
        cand_w_arr = np.asarray(eval_w, dtype=np.float32)
        min_fin_need = (len(base_members) + 1 + 1) // 2

        verif_ic10 = np.empty((k_fin, face_cache.n_sub), dtype=np.float32)
        verif_icc2 = np.empty((k_fin, face_cache.n_sub), dtype=np.float32)
        lib.sf_eval_squad_candidates_dual_ptrs(
            base_ptrs, ptr_f32(base_w_arr), len(base_members),
            cand_ptrs, ptr_f32(cand_w_arr), min_fin_need,
            k_fin, face_cache.n_sub, ctx["n_stocks"],
            ptr_i32(face_cache.sub_eval_idx), n_eval_full,
            ptr_u8(ev_mat), ptr_f32(q14_mat), ctx["n_b10"], ctx["n_c2"], 0,
            ptr_f32(ret_ord_mat), ptr_f32(verif_ic10), ptr_f32(verif_icc2),
        )

        target_min_sc = cur_score * (1.0 + min_rel_gain)
        best_choice = None
        best_eff_sc = cur_score * (1.0 + min_rel_gain * 0.5)

        for ci_c, (m_c, sgn_c, ufp1, ufp05) in enumerate(cand_cols):
            sc1, ok1_sc, vic1 = _eval_verif_1d(verif_ic10[2 * ci_c])
            sc05, ok05_sc, vic05 = _eval_verif_1d(verif_ic10[2 * ci_c + 1])

            pass1 = (len(ufp1) > 0) and ok1_sc and (sc1 > target_min_sc)
            if pass1 and n_locked > 0:
                if _max_abs_ic_corr_vec_vs_mat(vic1, other_verif_ics) > 0.575:
                    pass1 = False

            pass05 = (len(ufp05) > 0) and ok05_sc and (sc05 > target_min_sc)
            if pass05 and n_locked > 0:
                if _max_abs_ic_corr_vec_vs_mat(vic05, other_verif_ics) > 0.575:
                    pass05 = False

            chosen_c = None
            if pass1 and pass05:
                if sc05 > sc1 * (1.0 + PREFER_EQ_REL):
                    chosen_c = (0.5, sc05, ufp05, vic05)
                else:
                    chosen_c = (1.0, sc1, ufp1, vic1)
            elif pass1:
                chosen_c = (1.0, sc1, ufp1, vic1)
            elif pass05:
                chosen_c = (0.5, sc05, ufp05, vic05)

            if chosen_c is not None:
                w_pick, sc_pick, ufp_pick, vic_pick = chosen_c
                eff_sc = sc_pick / (1.008 if w_pick < 0.99 else 1.0)
                if eff_sc > best_eff_sc:
                    best_eff_sc = eff_sc
                    best_choice = (m_c, sgn_c, w_pick, sc_pick, ufp_pick, vic_pick)

        return best_choice

    seed_pass_specs = (
        (0.32, 0.50, 1),
        (0.35, 0.54, MAX_RAW_REUSE_TRACK),
    )
    pass_cursors = [0, 0]

    for g in range(n_init, total_cap):
        seed_picked = None
        for p_idx, (seed_cs_cap, seed_ic_cap, raw_cap_seed) in enumerate(seed_pass_specs):
            oi = pass_cursors[p_idx]
            while oi < len(order) and seed_picked is None:
                b_end = min(len(order), oi + 256)
                cand_ois = []
                for k_oi in range(oi, b_end):
                    m_k = int(order[k_oi])
                    if m_k in used_cols:
                        continue
                    if raw_counts.get(m_k // 50, 0) >= raw_cap_seed:
                        continue
                    cand_ois.append(k_oi)
                if not cand_ois:
                    oi = b_end
                    continue
                if g > 0:
                    uic_batch = seed_cand_uics[cand_ois]
                    max_uic_c = np.max(np.abs(uic_batch @ locked_proxy_uics[:g].T), axis=1)
                    surv_ois = [cand_ois[i] for i in range(len(cand_ois)) if float(max_uic_c[i]) <= seed_ic_cap + 0.02]
                else:
                    surv_ois = cand_ois
                if not surv_ois:
                    oi = b_end
                    continue

                surv_ms = [int(order[k_oi]) for k_oi in surv_ois]
                surv_sgns = col_sgn[surv_ms].astype(np.float32)
                ufps_surv = fp_i8[surv_ms].astype(np.float32) * (surv_sgns * fp_sc[surv_ms])[:, None]
                if g > 0:
                    cs_mat_s = np.abs(ufps_surv @ locked_fps[:g].T)
                    cs_ok = np.max(cs_mat_s, axis=1) <= seed_cs_cap
                    if g >= 5:
                        cs_ok &= np.mean(cs_mat_s, axis=1) <= 0.17
                else:
                    cs_ok = np.ones(len(surv_ois), dtype=bool)

                chosen_next_oi = b_end
                for si_loc, k_oi in enumerate(surv_ois):
                    if not cs_ok[si_loc]:
                        continue
                    m = surv_ms[si_loc]
                    sgn = int(col_sgn[m])
                    ufp = ufps_surv[si_loc]
                    ic_v = view["ic_target"][m] * np.float32(sgn)
                    if g > 0:
                        if _max_abs_ic_corr_vec_vs_mat(ic_v, locked_proxy_ics[:g]) > seed_ic_cap:
                            continue
                        if _max_abs_ic_corr_vec_vs_mat(ic_v[face_cache.sub_eval_idx], locked_verif_ics[:g]) > seed_ic_cap + 0.015:
                            continue
                    vic0 = np.ascontiguousarray(ic_v[face_cache.sub_eval_idx], dtype=np.float32)
                    sc0, _, _, ok0 = compute_pure_score_full_1d(vic0, verif_ir_sqrt, obj_mode)
                    if not ok0 or sc0 <= 0.0:
                        used_cols.add(m)
                        continue
                    seed_picked = (m, sgn, sc0, ufp, vic0)
                    chosen_next_oi = k_oi + 1
                    break
                oi = chosen_next_oi
            pass_cursors[p_idx] = oi
            if seed_picked is not None:
                break

        if seed_picked is None:
            break

        m0, sgn0, cur_sc, cur_ufp, cur_vic = seed_picked
        sq_cur: list[tuple[int, int, float]] = [(m0, sgn0, 1.0)]
        used_cols.add(m0)
        raw_counts[m0 // 50] = raw_counts.get(m0 // 50, 0) + 1

        for _step in range(MAX_SQUAD_LEN - 1):
            res = _search_best_addition_fullhist(
                n_locked=g,
                base_members=sq_cur,
                cur_score=cur_sc,
                top_exact_m=20,
                min_rel_gain=MIN_REL_GAIN_ADD,
            )
            if res is None:
                break
            m_w, sgn_w, w_w, sc_w, ufp_w, vic_w = res
            sq_cur.append((m_w, sgn_w, w_w))
            used_cols.add(m_w)
            raw_counts[m_w // 50] = raw_counts.get(m_w // 50, 0) + 1
            cur_sc = sc_w
            cur_ufp = ufp_w
            cur_vic = vic_w

        if len(sq_cur) >= 2:
            for pos in range(len(sq_cur)):
                old_m, _, _ = sq_cur[pos]
                old_r = old_m // 50
                rem_members = sq_cur[:pos] + sq_cur[pos + 1 :]
                raw_counts[old_r] = max(0, raw_counts.get(old_r, 1) - 1)
                used_cols.discard(old_m)
                res = _search_best_addition_fullhist(
                    n_locked=g,
                    base_members=rem_members,
                    cur_score=cur_sc,
                    top_exact_m=14,
                    min_rel_gain=MIN_REL_GAIN_SWAP,
                )
                if res is not None:
                    m_w, sgn_w, w_w, sc_w, ufp_w, vic_w = res
                    sq_cur[pos] = (m_w, sgn_w, w_w)
                    used_cols.add(m_w)
                    raw_counts[m_w // 50] = raw_counts.get(m_w // 50, 0) + 1
                    cur_sc = sc_w
                    cur_ufp = ufp_w
                    cur_vic = vic_w
                else:
                    used_cols.add(old_m)
                    raw_counts[old_r] = raw_counts.get(old_r, 0) + 1

        _, _, _, sq_ic_clean, ufp_final = _build_squad_ic_and_fp(sq_cur)
        locked_squads.append(sq_cur)
        locked_fps[g] = ufp_final if np.linalg.norm(ufp_final) > 1e-8 else cur_ufp
        locked_proxy_ics[g] = sq_ic_clean
        locked_proxy_uics[g] = _to_unit_ic_1d(sq_ic_clean)
        locked_verif_ics[g] = cur_vic

    new_squads = locked_squads[n_init:]
    return [(track_name, sq) for sq in new_squads]


def eval_cand_pool_exact_mu_ir_corrs(
    lib: ctypes.CDLL,
    scan: dict,
    ctx: dict,
    cand_squads: list[list[tuple[int, int, float]]],
    gap: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """精确评估候选超因子池在全历史上的 `10-Barra` IC / RankICIR、截面残差相关矩阵与 `10-Barra`/`ColdV2` 双口径日频 IC 相关矩阵。"""
    t_len = ctx["t_len"]
    n_stocks = ctx["n_stocks"]
    n_k = len(cand_squads)
    if n_k == 0:
        z0 = np.zeros((0, 0), dtype=np.float64)
        return np.zeros(0), np.zeros(0), z0, z0, z0
    ir_sqrt = float(np.sqrt(50.0 if gap == 5 else 250.0))

    uniq_keys: list[tuple[int, int]] = []
    key_to_idx: dict[tuple[int, int], int] = {}
    for sq in cand_squads:
        for m, sgn, _ in sq:
            k = (m, sgn)
            if k not in key_to_idx:
                key_to_idx[k] = len(uniq_keys)
                uniq_keys.append(k)

    l_uniq = len(uniq_keys)
    specs = np.empty((l_uniq, 4), dtype=np.int32)
    for ui, (m, sgn) in enumerate(uniq_keys):
        rem = m % 50
        specs[ui, 0] = m // 50
        specs[ui, 1] = rem // 25
        specs[ui, 2] = rem % 25
        specs[ui, 3] = sgn

    all_t_idx = np.arange(t_len, dtype=np.int32)
    u_uniq = scan["u16_bank"].materialize_specs(specs, all_t_idx, ctx["win_len_turn"])

    sq_offsets = np.zeros(n_k + 1, dtype=np.int32)
    u_idx_list, w_list = [], []
    for g, sq in enumerate(cand_squads):
        sq_offsets[g] = len(u_idx_list)
        for m, sgn, w in sq:
            u_idx_list.append(key_to_idx[(m, sgn)])
            w_list.append(float(w))
    sq_offsets[n_k] = len(u_idx_list)

    raw_k_plus1 = np.empty((n_k + 1, t_len, n_stocks), dtype=np.float32)
    lib.sf_combine_squads_from_uniq(
        ptr_f32(u_uniq), l_uniq, t_len, n_stocks,
        ptr_i32(sq_offsets), ptr_i32(np.asarray(u_idx_list, dtype=np.int32)),
        ptr_f32(np.asarray(w_list, dtype=np.float32)), n_k,
        ptr_f32(raw_k_plus1),
    )
    del u_uniq

    eff_indices = ctx["eff_indices"]
    d_bt = len(eff_indices)
    ret_g = ctx["ret_gap5"] if gap == 5 else ctx["ret_gap1"]
    rsum_g = ctx["ret_sum_gap5"] if gap == 5 else ctx["ret_sum_gap1"]

    def _run_style(style_q: np.ndarray, n_style: int, need_cs: bool):
        neu_k = np.empty((n_k, t_len, n_stocks), dtype=np.float32)
        lib.sf_neutralize_block(
            ptr_f32(raw_k_plus1[:n_k]), n_k, t_len, n_stocks,
            ptr_u8(ctx["tradable"]), ptr_f32(style_q), n_style,
            ptr_f32(neu_k),
        )
        cs_m = compute_cs_corr_matrix_exact(neu_k, ctx["tradable"], n_k) if need_cs else None
        top_r = np.empty((n_k, d_bt), dtype=np.float64)
        bot_r = np.empty((n_k, d_bt), dtype=np.float64)
        mkt_r = np.empty((n_k, d_bt), dtype=np.float64)
        rat_r = np.empty((n_k, d_bt), dtype=np.float64)
        ic_no = np.empty((n_k, d_bt), dtype=np.float32)
        ic_dy = np.empty((n_k, d_bt), dtype=np.float32)
        lib.sf_eval_tail_v4_exact_backtest(
            ptr_f32(neu_k), n_k, t_len, n_stocks,
            ptr_f32(ret_g), ptr_f32(rsum_g), ptr_f32(ctx["restrict"]),
            ptr_i32(eff_indices), d_bt, gap, 10,
            ptr_f64(top_r), ptr_f64(bot_r), ptr_f64(mkt_r), ptr_f64(rat_r),
            ptr_f32(ic_no), ptr_f32(ic_dy),
        )
        del neu_k, top_r, bot_r, mkt_r, rat_r
        ic_cmat, _ = pairwise_ic_corr_matrix_strict(ic_dy)
        return cs_m, np.abs(ic_cmat), ic_no

    cs10, abs_ic10, ic10_no = _run_style(ctx["style_q_10_full"], ctx["n_b10"], True)
    _, abs_icc2, _ = _run_style(ctx["style_q_c2_full"], ctx["n_c2"], False)
    del raw_k_plus1

    mu_arr = np.full(n_k, -1e9, dtype=np.float64)
    ir_arr = np.full(n_k, -1e9, dtype=np.float64)
    for i in range(n_k):
        fin = ic10_no[i][np.isfinite(ic10_no[i])].astype(np.float64)
        if len(fin) >= 20:
            mu = float(np.mean(fin))
            sd = max(float(np.std(fin, ddof=0)), 1e-6)
            mu_arr[i] = mu
            ir_arr[i] = float((mu / sd) * ir_sqrt)

    return mu_arr, ir_arr, np.abs(cs10), abs_ic10, abs_icc2


def arbitrate_final_45_pure(
    cands_meta: list[tuple[str, list[tuple[int, int, float]]]],
    exact_scores: np.ndarray,
    abs_cs_mat: np.ndarray,
    abs_ic10_mat: np.ndarray,
    abs_icc2_guard: np.ndarray | None = None,
) -> tuple[list[tuple[str, list[tuple[int, int, float]]]], dict[str, int]]:
    order_idx = np.argsort(-exact_scores)
    selected: list[tuple[str, list[tuple[int, int, float]]]] = []
    sel_indices: list[int] = []
    raw_counts: dict[int, int] = {}
    track_cnt = {"FixedCal": 0, "TurnEq": 0}
    used_idx: set[int] = set()
    seen_sigs: set[frozenset[tuple[int, int]]] = set()

    pass_specs = (
        (0.35, 0.17, 0.550, MAX_TRACK_QUOTA),
        (INTER_MAX_CORR, INTER_MEAN_CORR, 0.575, 30),
        (0.38, 0.19, INTER_IC_CORR_HARD, 45),
    )
    for cs_cap, mean_cs_cap, ic_cap, quota_cap in pass_specs:
        for ci_np in order_idx:
            if len(selected) >= K_FINAL_SQUADS:
                break
            ci = int(ci_np)
            sc_val = float(exact_scores[ci])
            trk, sq = cands_meta[ci]
            if ci in used_idx or sc_val <= 0.0 or track_cnt[trk] >= quota_cap:
                continue
            sq_sig = frozenset((m_i, s_i) for m_i, s_i, _ in sq)
            if sq_sig in seen_sigs:
                continue
            sq_raws = [m_i // 50 for m_i, _, _ in sq]
            if any(raw_counts.get(r, 0) + 1 > MAX_RAW_REUSE_FINAL for r in sq_raws):
                continue
            if sel_indices:
                cs_vec = abs_cs_mat[ci, sel_indices]
                if float(np.max(cs_vec)) > cs_cap:
                    continue
                if len(sel_indices) >= 5 and float(np.mean(cs_vec)) > mean_cs_cap:
                    continue
                if float(np.max(abs_ic10_mat[ci, sel_indices])) > ic_cap:
                    continue
                if abs_icc2_guard is not None and float(np.max(abs_icc2_guard[ci, sel_indices])) > INTER_IC_CORR_HARD:
                    continue
            sel_indices.append(ci)
            selected.append((trk, sq))
            used_idx.add(ci)
            seen_sigs.add(sq_sig)
            track_cnt[trk] += 1
            for r in sq_raws:
                raw_counts[r] = raw_counts.get(r, 0) + 1
        if len(selected) >= K_FINAL_SQUADS:
            break

    return selected, track_cnt


def select_spearheads_exact(
    spear_pool: list[tuple[str, list[tuple[int, int, float]]]],
    mu_arr: np.ndarray,
    ir_arr: np.ndarray,
    abs_cs: np.ndarray,
    abs_ic10: np.ndarray,
    abs_icc2: np.ndarray,
    k_max: int,
    mu_min: float,
    ir_min: float,
) -> list[tuple[str, list[tuple[int, int, float]]]]:
    if k_max <= 0 or len(spear_pool) == 0:
        return []
    spear_rank_sc = np.where(
        (mu_arr >= mu_min) & (ir_arr >= ir_min),
        mu_arr * (1.0 + 0.05 * np.minimum(ir_arr, 10.0)),
        -1e9,
    )
    order = np.argsort(-spear_rank_sc)
    picked: list[tuple[str, list[tuple[int, int, float]]]] = []
    sel_idx: list[int] = []
    raw_counts: dict[int, int] = {}
    trk_cnt = {"FixedCal": 0, "TurnEq": 0}
    seen_mem_sets: set[frozenset[tuple[int, int]]] = set()
    max_per_trk = max(2, k_max - 1) if k_max >= 3 else k_max

    for ci_np in order:
        if len(picked) >= k_max:
            break
        ci = int(ci_np)
        if float(spear_rank_sc[ci]) <= 0.0:
            break
        trk, sq = spear_pool[ci]
        if trk_cnt[trk] >= max_per_trk:
            continue
        msig = frozenset((m_i, s_i) for m_i, s_i, _ in sq)
        if msig in seen_mem_sets:
            continue
        sq_raws = [m_i // 50 for m_i, _, _ in sq]
        if any(raw_counts.get(r, 0) + 1 > MAX_RAW_REUSE_FINAL for r in sq_raws):
            continue
        if sel_idx:
            if float(np.max(abs_cs[ci, sel_idx])) > 0.34:
                continue
            if len(sel_idx) >= 5 and float(np.mean(abs_cs[ci, sel_idx])) > 0.165:
                continue
            if float(np.max(abs_ic10[ci, sel_idx])) > 0.54:
                continue
            if float(np.max(abs_icc2[ci, sel_idx])) > INTER_IC_CORR_HARD:
                continue
        sel_idx.append(ci)
        picked.append((trk, sq))
        seen_mem_sets.add(msig)
        trk_cnt[trk] += 1
        for r in sq_raws:
            raw_counts[r] = raw_counts.get(r, 0) + 1

    return picked


def arbitrate_locked_spearheads_plus_core_icir(
    lib: ctypes.CDLL,
    scan: dict,
    ctx: dict,
    spearheads: list[tuple[str, list[tuple[int, int, float]]]],
    core_cands: list[tuple[str, list[tuple[int, int, float]]]],
    gap: int = 5,
) -> tuple[list[tuple[str, list[tuple[int, int, float]]]], list[str]]:
    k_sp = len(spearheads)
    combined = list(spearheads) + deduplicate_tagged_cands(core_cands)
    mu_all, ir_all, abs_cs, abs_ic10, abs_icc2 = eval_cand_pool_exact_mu_ir_corrs(
        lib, scan, ctx, [sq for _, sq in combined], gap=gap
    )

    selected: list[tuple[str, list[tuple[int, int, float]]]] = []
    roles: list[str] = []
    sel_indices: list[int] = []
    raw_counts: dict[int, int] = {}
    track_cnt = {"FixedCal": 0, "TurnEq": 0}
    used_idx: set[int] = set()
    seen_sigs: set[frozenset[tuple[int, int]]] = set()

    for ci in range(k_sp):
        trk, sq = combined[ci]
        sq_sig = frozenset((m_i, s_i) for m_i, s_i, _ in sq)
        sel_indices.append(ci)
        selected.append((trk, sq))
        roles.append("Spearhead_HighIC")
        used_idx.add(ci)
        seen_sigs.add(sq_sig)
        track_cnt[trk] += 1
        for m_i, _, _ in sq:
            r = m_i // 50
            raw_counts[r] = raw_counts.get(r, 0) + 1

    core_indices = np.arange(k_sp, len(combined), dtype=np.int32)
    core_scores = np.where(mu_all[core_indices] > 0.0, ir_all[core_indices], -1e9)
    order_core = core_indices[np.argsort(-core_scores)]

    pass_specs = (
        (0.35, 0.17, 0.550, MAX_TRACK_QUOTA),
        (INTER_MAX_CORR, INTER_MEAN_CORR, 0.575, 30),
        (0.38, 0.19, INTER_IC_CORR_HARD, 45),
    )
    for cs_cap, mean_cs_cap, ic_cap, quota_cap in pass_specs:
        for ci_np in order_core:
            if len(selected) >= K_FINAL_SQUADS:
                break
            ci = int(ci_np)
            if ci in used_idx or float(ir_all[ci]) <= 0.0 or float(mu_all[ci]) <= 0.0:
                continue
            trk, sq = combined[ci]
            if track_cnt[trk] >= quota_cap:
                continue
            sq_sig = frozenset((m_i, s_i) for m_i, s_i, _ in sq)
            if sq_sig in seen_sigs:
                continue
            sq_raws = [m_i // 50 for m_i, _, _ in sq]
            if any(raw_counts.get(r, 0) + 1 > MAX_RAW_REUSE_FINAL for r in sq_raws):
                continue
            if sel_indices:
                cs_vec = abs_cs[ci, sel_indices]
                if float(np.max(cs_vec)) > cs_cap:
                    continue
                if len(sel_indices) >= 5 and float(np.mean(cs_vec)) > mean_cs_cap:
                    continue
                if float(np.max(abs_ic10[ci, sel_indices])) > ic_cap:
                    continue
                if float(np.max(abs_icc2[ci, sel_indices])) > INTER_IC_CORR_HARD:
                    continue
            sel_indices.append(ci)
            selected.append((trk, sq))
            roles.append("Core_PureICIR")
            used_idx.add(ci)
            seen_sigs.add(sq_sig)
            track_cnt[trk] += 1
            for r in sq_raws:
                raw_counts[r] = raw_counts.get(r, 0) + 1
        if len(selected) >= K_FINAL_SQUADS:
            break

    return selected, roles


def evaluate_super_squads_fullhist(
    lib: ctypes.CDLL,
    scan: dict,
    ctx: dict,
    tagged_squads: list[tuple[str, list[tuple[int, int, float]]]],
    gap: int = 5,
    roles: list[str] | None = None,
    compute_captain_attribution: bool = False,
) -> tuple[dict, list[dict]]:
    """在完整全市场全历史上评估入选超因子团（同步输出 10-Barra 与 ColdV2 精确十分组回测及合成配方）。"""
    t_len = ctx["t_len"]
    n_stocks = ctx["n_stocks"]
    n_sq = len(tagged_squads)
    if n_sq == 0:
        return {"gap": gap, "n_squads": 0}, []
    squads = [sq for _, sq in tagged_squads]

    uniq_keys: list[tuple[int, int]] = []
    key_to_idx: dict[tuple[int, int], int] = {}
    for sq in squads:
        for m, sgn, _ in sq:
            k = (m, sgn)
            if k not in key_to_idx:
                key_to_idx[k] = len(uniq_keys)
                uniq_keys.append(k)

    l_uniq = len(uniq_keys)
    specs = np.empty((l_uniq, 4), dtype=np.int32)
    for ui, (m, sgn) in enumerate(uniq_keys):
        rem = m % 50
        specs[ui, 0] = m // 50
        specs[ui, 1] = rem // 25
        specs[ui, 2] = rem % 25
        specs[ui, 3] = sgn

    all_t_idx = np.arange(t_len, dtype=np.int32)
    u_uniq = scan["u16_bank"].materialize_specs(specs, all_t_idx, ctx["win_len_turn"])

    def _run_squad_list(sq_list: list[list[tuple[int, int, float]]]):
        k_n = len(sq_list)
        sq_offsets = np.zeros(k_n + 1, dtype=np.int32)
        u_idx_list, w_list = [], []
        for g, sq in enumerate(sq_list):
            sq_offsets[g] = len(u_idx_list)
            for m, sgn, w in sq:
                u_idx_list.append(key_to_idx[(m, sgn)])
                w_list.append(float(w))
        sq_offsets[k_n] = len(u_idx_list)

        raw_k_p1 = np.empty((k_n + 1, t_len, n_stocks), dtype=np.float32)
        lib.sf_combine_squads_from_uniq(
            ptr_f32(u_uniq), l_uniq, t_len, n_stocks,
            ptr_i32(sq_offsets), ptr_i32(np.asarray(u_idx_list, dtype=np.int32)),
            ptr_f32(np.asarray(w_list, dtype=np.float32)), k_n,
            ptr_f32(raw_k_p1),
        )
        k_bt = k_n + 2
        neu10 = np.empty((k_bt, t_len, n_stocks), dtype=np.float32)
        lib.sf_neutralize_block(
            ptr_f32(raw_k_p1), k_n + 1, t_len, n_stocks,
            ptr_u8(ctx["tradable"]), ptr_f32(ctx["style_q_10_full"]), ctx["n_b10"],
            ptr_f32(neu10[: k_n + 1]),
        )
        lib.sf_fill_post_composite(ptr_f32(neu10), k_n, t_len, n_stocks)

        c2 = np.empty((k_bt, t_len, n_stocks), dtype=np.float32)
        lib.sf_neutralize_block(
            ptr_f32(raw_k_p1), k_n + 1, t_len, n_stocks,
            ptr_u8(ctx["tradable"]), ptr_f32(ctx["style_q_c2_full"]), ctx["n_c2"],
            ptr_f32(c2[: k_n + 1]),
        )
        lib.sf_fill_post_composite(ptr_f32(c2), k_n, t_len, n_stocks)
        del raw_k_p1
        return neu10, c2

    neu10_k47, c2_k47 = _run_squad_list(squads)

    eff_indices = ctx["eff_indices"]
    d_bt = len(eff_indices)
    ret_g = ctx["ret_gap5"] if gap == 5 else ctx["ret_gap1"]
    rsum_g = ctx["ret_sum_gap5"] if gap == 5 else ctx["ret_sum_gap1"]
    idx_ret_eff = ctx["index_ret"][eff_indices].astype(np.float64)

    def _bt_mat(mat_bt: np.ndarray):
        k_bt = mat_bt.shape[0]
        top_r = np.empty((k_bt, d_bt), dtype=np.float64)
        bot_r = np.empty((k_bt, d_bt), dtype=np.float64)
        mkt_r = np.empty((k_bt, d_bt), dtype=np.float64)
        rat_r = np.empty((k_bt, d_bt), dtype=np.float64)
        ic_no = np.empty((k_bt, d_bt), dtype=np.float32)
        ic_dy = np.empty((k_bt, d_bt), dtype=np.float32)
        lib.sf_eval_tail_v4_exact_backtest(
            ptr_f32(mat_bt), k_bt, t_len, n_stocks,
            ptr_f32(ret_g), ptr_f32(rsum_g), ptr_f32(ctx["restrict"]),
            ptr_i32(eff_indices), d_bt, gap, 10,
            ptr_f64(top_r), ptr_f64(bot_r), ptr_f64(mkt_r), ptr_f64(rat_r),
            ptr_f32(ic_no), ptr_f32(ic_dy),
        )
        return top_r, bot_r, mkt_r, ic_no, ic_dy

    top10, bot10, mkt10, ic10, ic10_dy = _bt_mat(neu10_k47)
    topc2, botc2, mktc2, icc2, icc2_dy = _bt_mat(c2_k47)

    cs_mat_10 = compute_cs_corr_matrix_exact(neu10_k47, ctx["tradable"], n_sq)
    cs_mat_c2 = compute_cs_corr_matrix_exact(c2_k47, ctx["tradable"], n_sq)
    orth_10 = {**_compute_orthogonality_45(neu10_k47, ctx["tradable"], n_sq), **_effective_n_from_corr(cs_mat_10)}
    orth_c2 = {**_compute_orthogonality_45(c2_k47, ctx["tradable"], n_sq), **_effective_n_from_corr(cs_mat_c2)}
    del neu10_k47, c2_k47

    ic_cmat_10, ic_corr_10 = pairwise_ic_corr_matrix_strict(ic10_dy[:n_sq])
    ic_cmat_c2, ic_corr_c2 = pairwise_ic_corr_matrix_strict(icc2_dy[:n_sq])
    ic_corr_10.update(_effective_n_from_corr(ic_cmat_10))
    ic_corr_c2.update(_effective_n_from_corr(ic_cmat_c2))

    abs_ic10_off = np.abs(ic_cmat_10).copy()
    np.fill_diagonal(abs_ic10_off, 0.0)
    row_max_ic10 = abs_ic10_off.max(axis=1) if n_sq >= 2 else np.zeros(n_sq)

    abs_icc2_off = np.abs(ic_cmat_c2).copy()
    np.fill_diagonal(abs_icc2_off, 0.0)
    row_max_icc2 = abs_icc2_off.max(axis=1) if n_sq >= 2 else np.zeros(n_sq)

    abs_cs10_off = np.abs(cs_mat_10).copy()
    np.fill_diagonal(abs_cs10_off, 0.0)
    row_max_cs10 = abs_cs10_off.max(axis=1) if n_sq >= 2 else np.zeros(n_sq)

    cap_ic10, cap_ir10, cap_hd10 = None, None, None
    cap_full_10, cap_full_c2 = None, None
    if compute_captain_attribution:
        cap_squads = [[sq[0]] for sq in squads]
        cap_neu10, cap_c2 = _run_squad_list(cap_squads)
        ctop10, cbot10, cmkt10, cic10, _ = _bt_mat(cap_neu10)
        ctopc2, cbotc2, cmktc2, cicc2, _ = _bt_mat(cap_c2)
        del cap_neu10, cap_c2
        cap_full_10 = _pack_bt_metrics(ctop10, cbot10, cmkt10, idx_ret_eff, cic10, gap, n_sq)
        cap_full_c2 = _pack_bt_metrics(ctopc2, cbotc2, cmktc2, idx_ret_eff, cicc2, gap, n_sq)
        cap_ic10 = list(cap_full_10["per_factor_ic"])
        cap_ir10 = list(cap_full_10["per_factor_icir"])
        cap_hd10 = list(cap_full_10["per_factor_hedge_ann"])
        for d_c in (cap_full_10, cap_full_c2):
            for k_rm in ("per_factor_ic", "per_factor_icir", "per_factor_hedge_ann"):
                d_c.pop(k_rm, None)

    del u_uniq

    full_10 = _pack_bt_metrics(top10, bot10, mkt10, idx_ret_eff, ic10, gap, n_sq)
    full_c2 = _pack_bt_metrics(topc2, botc2, mktc2, idx_ret_eff, icc2, gap, n_sq)

    pf_ic10 = np.asarray(full_10["per_factor_ic"], dtype=np.float64)
    pf_ir10 = np.asarray(full_10["per_factor_icir"], dtype=np.float64)
    pf_ic_c2 = np.asarray(full_c2["per_factor_ic"], dtype=np.float64)
    pf_ir_c2 = np.asarray(full_c2["per_factor_icir"], dtype=np.float64)

    top_n_stat = min(5, n_sq)
    dist_10 = {
        "max_ic": float(np.max(pf_ic10)),
        "top5_mean_ic": float(np.mean(np.sort(pf_ic10)[-top_n_stat:])),
        "p75_ic": float(np.percentile(pf_ic10, 75)),
        "median_ic": float(np.median(pf_ic10)),
        "min_ic": float(np.min(pf_ic10)),
        "cnt_ic_ge_0p035": int(np.sum(pf_ic10 >= 0.035)),
        "cnt_ic_ge_0p040": int(np.sum(pf_ic10 >= 0.040)),
        "max_icir": float(np.max(pf_ir10)),
        "top5_mean_icir": float(np.mean(np.sort(pf_ir10)[-top_n_stat:])),
        "median_icir": float(np.median(pf_ir10)),
        "min_icir": float(np.min(pf_ir10)),
        "cnt_icir_ge_6p0": int(np.sum(pf_ir10 >= 6.0)),
        "cnt_icir_ge_7p0": int(np.sum(pf_ir10 >= 7.0)),
    }
    dist_c2 = {
        "max_ic": float(np.max(pf_ic_c2)),
        "top5_mean_ic": float(np.mean(np.sort(pf_ic_c2)[-top_n_stat:])),
        "median_ic": float(np.median(pf_ic_c2)),
        "max_icir": float(np.max(pf_ir_c2)),
        "top5_mean_icir": float(np.mean(np.sort(pf_ir_c2)[-top_n_stat:])),
        "median_icir": float(np.median(pf_ir_c2)),
    }

    recipes_full = []
    raw_use: dict[int, int] = {}
    w_counts = {"1.00": 0, "0.50": 0, "captain_1.00": 0, "helper_1.00": 0, "helper_0.50": 0}
    size_hist = {1: 0, 2: 0, 3: 0, 4: 0}
    track_counts = {"FixedCal": 0, "TurnEq": 0}

    for si, (trk, sq) in enumerate(tagged_squads):
        track_counts[trk] = track_counts.get(trk, 0) + 1
        size_hist[len(sq)] = size_hist.get(len(sq), 0) + 1
        m_recs = []
        for mi_idx, (m, sgn, w) in enumerate(sq):
            raw_use[m // 50] = raw_use.get(m // 50, 0) + 1
            wk = f"{w:.2f}"
            w_counts[wk] = w_counts.get(wk, 0) + 1
            if mi_idx == 0:
                w_counts["captain_1.00"] += 1
            elif abs(w - 1.0) < 1e-3:
                w_counts["helper_1.00"] += 1
            else:
                w_counts["helper_0.50"] += 1
            m_recs.append(decode_col_id(m, scan["raw_names"], sgn, w))

        formula_str = " + ".join(
            f"{'+' if mb['sign'] > 0 else '-'}{mb['weight']:.1f}*{mb['derived_name']}" for mb in m_recs
        )
        role_str = roles[si] if (roles is not None and si < len(roles)) else "Core"
        rec_item = {
            "squad_idx": si + 1,
            "role": role_str,
            "track": trk,
            "size": len(sq),
            "captain_name": m_recs[0]["derived_name"],
            "formula": formula_str,
            "neu10_ic": float(pf_ic10[si]),
            "neu10_icir": float(pf_ir10[si]),
            "neu10_hedge_ann": float(full_10["per_factor_hedge_ann"][si]),
            "coldv2_ic": float(pf_ic_c2[si]),
            "coldv2_icir": float(pf_ir_c2[si]),
            "coldv2_hedge_ann": float(full_c2["per_factor_hedge_ann"][si]),
            "max_abs_ic_corr_10b": float(row_max_ic10[si]),
            "max_abs_ic_corr_c2": float(row_max_icc2[si]),
            "max_abs_cs_corr_10b": float(row_max_cs10[si]),
            "members": m_recs,
        }
        if cap_ic10 is not None:
            rec_item["captain_only_neu10_ic"] = float(cap_ic10[si])
            rec_item["captain_only_neu10_icir"] = float(cap_ir10[si])
            rec_item["captain_only_neu10_hedge_ann"] = float(cap_hd10[si])
            rec_item["delta_neu10_ic"] = float(pf_ic10[si] - cap_ic10[si])
            rec_item["delta_neu10_icir"] = float(pf_ir10[si] - cap_ir10[si])
        recipes_full.append(rec_item)

    for d_m in (full_10, full_c2):
        for k_rm in ("per_factor_ic", "per_factor_icir", "per_factor_hedge_ann"):
            d_m.pop(k_rm, None)

    n_multi = int(sum(1 for _, sq in tagged_squads if len(sq) >= 2))
    n_spear = sum(1 for r in recipes_full if r.get("role") == "Spearhead_HighIC")
    summary = {
        "gap": gap,
        "n_squads": n_sq,
        "n_spearheads": n_spear,
        "n_core_icir": n_sq - n_spear,
        "multi_member_squads": n_multi,
        "singleton_squads": int(n_sq - n_multi),
        "enhanced_ratio": float(n_multi / max(n_sq, 1)),
        "avg_squad_size": float(np.mean([len(sq) for _, sq in tagged_squads])),
        "size_hist": {str(k): int(v) for k, v in sorted(size_hist.items())},
        "track_counts": track_counts,
        "weight_counts": w_counts,
        "unique_raw_factors_used": len(raw_use),
        "dist_10barra": dist_10,
        "dist_coldv2": dist_c2,
        "orthogonality_10barra": orth_10,
        "orthogonality_coldv2": orth_c2,
        "ic_corr_10barra": ic_corr_10,
        "ic_corr_coldv2": ic_corr_c2,
        "full_history": {"eval_10barra": full_10, "eval_coldv2": full_c2},
        "captain_only_45_baseline": (
            {"eval_10barra": cap_full_10, "eval_coldv2": cap_full_c2}
            if cap_full_10 is not None
            else None
        ),
    }
    return summary, recipes_full


def _screen_single_horizon(
    lib: ctypes.CDLL,
    scan: dict,
    ctx: dict,
    gap: int,
    mode: str,
    fast_spear_pool: bool = True,
    compute_captain_attribution: bool = True,
) -> tuple[dict, list[dict]]:
    """对单个预测周期 (`gap=5` 或 `gap=1`) 运行定稿超因子筛选与组装。"""
    if gap == 5:
        n_ev = ctx["n_g5"]
        sub_idx = np.arange(0, n_ev, dtype=np.int32)
        face_cache = VerifFaceCache(lib, scan, ctx, sub_idx, ctx["g5_rows"], max_items=6000)
        ev_mat, q14_mat, ret_ord = ctx["ev_g5"], ctx["q14_g5"], ctx["ret5_ord"]
        ic_floor_adapt = 0.022
    else:
        n_ev = ctx["n_g1"]
        sub_idx = np.arange(0, n_ev, 3, dtype=np.int32)
        face_cache = VerifFaceCache(lib, scan, ctx, sub_idx, ctx["g1_rows"], max_items=4000)
        ev_mat, q14_mat, ret_ord = ctx["ev_g1"], ctx["q14_g1"], ctx["ret1_ord"]
        ic_floor_adapt = 0.011

    ir_sqrt = float(np.sqrt(50.0 if gap == 5 else 250.0))
    mode_norm = mode.strip().lower()

    try:
        if mode_norm in ("pure_icir", "pure_ic", "balanced"):
            obj_mode = {"pure_icir": 0, "pure_ic": 1, "balanced": 2}[mode_norm]
            view = build_fullhist_objective_view(scan, gap, objective_mode=obj_mode)
            cands_fix = assemble_track_with_optional_prelocked(
                lib, scan, ctx, view, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "FixedCal", K_TRACK_CANDS_CORE
            )
            cands_trn = assemble_track_with_optional_prelocked(
                lib, scan, ctx, view, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "TurnEq", K_TRACK_CANDS_CORE
            )
            all_cands = cands_fix + cands_trn
            mu_c, ir_c, cs_c, ic10_c, icc2_c = eval_cand_pool_exact_mu_ir_corrs(
                lib, scan, ctx, [sq for _, sq in all_cands], gap=gap
            )
            sc_c = np.array([_eval_obj_scalar(float(mu_c[i]), float(ir_c[i]), obj_mode) for i in range(len(all_cands))])
            selected, _ = arbitrate_final_45_pure(all_cands, sc_c, cs_c, ic10_c, icc2_c)
            role_tag = {"pure_icir": "Core_PureICIR", "pure_ic": "Spearhead_HighIC", "balanced": "Balanced_Squad"}[mode_norm]
            roles = [role_tag] * len(selected)
            extra_meta: dict = {"mode": mode_norm}
        elif mode_norm in ("hybrid_adapt_icir", "hybrid_top5_icir"):
            v_icir = build_fullhist_objective_view(scan, gap, objective_mode=0)
            v_ic = build_fullhist_objective_view(scan, gap, objective_mode=1)
            v_bal = build_fullhist_objective_view(scan, gap, objective_mode=2)

            k_core_pool = K_TRACK_CANDS_CORE if fast_spear_pool else 58
            cands_icir_fix = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_icir, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "FixedCal", k_core_pool
            )
            cands_icir_trn = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_icir, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "TurnEq", k_core_pool
            )
            cands_icir_all = cands_icir_fix + cands_icir_trn
            mu_icir_all, ir_icir_all, cs_icir, ic10_icir, icc2_icir = eval_cand_pool_exact_mu_ir_corrs(
                lib, scan, ctx, [sq for _, sq in cands_icir_all], gap=gap
            )
            sc_icir = np.where(mu_icir_all > 0.0, ir_icir_all, -1e9)
            sel_icir, _ = arbitrate_final_45_pure(cands_icir_all, sc_icir, cs_icir, ic10_icir, icc2_icir)

            mu_icir45, ir_icir45, _, _, _ = eval_cand_pool_exact_mu_ir_corrs(
                lib, scan, ctx, [sq for _, sq in sel_icir], gap=gap
            )
            med_ic_icir45 = float(np.median(mu_icir45)) if len(mu_icir45) else 0.0
            p80_ic_icir45 = float(np.percentile(mu_icir45, 80)) if len(mu_icir45) else 0.0
            med_ir_icir45 = float(np.median(ir_icir45)) if len(ir_icir45) else 0.0
            ir_floor_spear = float(max(5.0, min(6.2, 0.70 * med_ir_icir45)))
            ic_thresh_adapt = float(max(ic_floor_adapt, 1.15 * p80_ic_icir45))

            k_bal_p = 18 if fast_spear_pool else K_TRACK_CANDS_BAL
            k_pic_p = 18 if fast_spear_pool else K_TRACK_CANDS_PURE_IC
            cands_bal_fix = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_bal, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "FixedCal", k_bal_p
            )
            cands_bal_trn = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_bal, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "TurnEq", k_bal_p
            )
            cands_pic_fix = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_ic, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "FixedCal", k_pic_p
            )
            cands_pic_trn = assemble_track_with_optional_prelocked(
                lib, scan, ctx, v_ic, face_cache, ev_mat, q14_mat, ret_ord, n_ev, "TurnEq", k_pic_p
            )
            spear_pool = deduplicate_tagged_cands(cands_bal_fix + cands_bal_trn + cands_pic_fix + cands_pic_trn)
            mu_sp, ir_sp, cs_sp, ic10_sp, icc2_sp = eval_cand_pool_exact_mu_ir_corrs(
                lib, scan, ctx, [sq for _, sq in spear_pool], gap=gap
            )

            if mode_norm == "hybrid_top5_icir":
                spears = select_spearheads_exact(
                    spear_pool, mu_sp, ir_sp, cs_sp, ic10_sp, icc2_sp,
                    k_max=5, mu_min=med_ic_icir45 * 1.02, ir_min=ir_floor_spear,
                )
            else:
                spears = select_spearheads_exact(
                    spear_pool, mu_sp, ir_sp, cs_sp, ic10_sp, icc2_sp,
                    k_max=8, mu_min=ic_thresh_adapt, ir_min=ir_floor_spear,
                )

            if len(spears) == 0:
                selected, roles = sel_icir, ["Core_PureICIR"] * len(sel_icir)
            else:
                pre_fix = [sq for trk, sq in spears if trk == "FixedCal"]
                pre_trn = [sq for trk, sq in spears if trk == "TurnEq"]
                k_topup = 28 if fast_spear_pool else K_TRACK_CANDS_BAL
                core_fix = assemble_track_with_optional_prelocked(
                    lib, scan, ctx, v_icir, face_cache, ev_mat, q14_mat, ret_ord,
                    n_ev, "FixedCal", k_topup, prelocked_squads=pre_fix,
                )
                core_trn = assemble_track_with_optional_prelocked(
                    lib, scan, ctx, v_icir, face_cache, ev_mat, q14_mat, ret_ord,
                    n_ev, "TurnEq", k_topup, prelocked_squads=pre_trn,
                )
                selected, roles = arbitrate_locked_spearheads_plus_core_icir(
                    lib, scan, ctx, spears, core_fix + core_trn + cands_icir_all, gap=gap
                )

            extra_meta = {
                "mode": mode_norm,
                "ir_floor_spear": round(ir_floor_spear, 3),
                "ic_thresh_adapt": round(ic_thresh_adapt, 5),
                "p80_ic_pure_icir45": round(p80_ic_icir45, 5),
                "med_ic_pure_icir45": round(med_ic_icir45, 5),
                "med_ir_pure_icir45": round(med_ir_icir45, 3),
            }
        else:
            raise ValueError(
                f"未知超因子模式 mode={mode!r}，可选: 'hybrid_adapt_icir', 'hybrid_top5_icir', 'pure_icir', 'balanced', 'pure_ic'"
            )
    finally:
        del face_cache
        _trim_malloc()

    summary, recipes = evaluate_super_squads_fullhist(
        lib,
        scan,
        ctx,
        selected,
        gap=gap,
        roles=roles,
        compute_captain_attribution=compute_captain_attribution,
    )
    summary.update(extra_meta)
    return summary, recipes


def screen_super_factors_from_context(
    ctx: dict,
    store_dir: str,
    raw_names: list[str],
    valid_store_cols: np.ndarray,
    disp: str = "series",
    gaps: Iterable[int] = (5, 1),
    mode: str = "hybrid_adapt_icir",
    slab_cap: int = 4500,
    fast_spear_pool: bool = True,
    n_threads: int | None = None,
    compute_captain_attribution: bool = True,
    must_keep_mask: np.ndarray | None = None,
) -> dict:
    """给定全市场上下文 `ctx` 与 `factor_store` 路径，全自动完成双缓冲扫描、超因子组团与全历史回测。"""
    t0 = time.perf_counter()
    gap_list = [int(g) for g in gaps]
    for g in gap_list:
        if g not in (5, 1):
            raise ValueError(f"仅支持 gap=5 或 gap=1，收到 gap={g}")
    need_gap1 = 1 in gap_list

    lib = load_super_factor_cdll(n_threads=n_threads)
    scan = None
    try:
        scan = stream_scan_super_store(
            lib=lib,
            ctx=ctx,
            disp=disp,
            store_dir=store_dir,
            raw_names=raw_names,
            valid_store_cols=valid_store_cols,
            need_gap1=need_gap1,
            slab_cap=slab_cap,
            must_keep_mask=must_keep_mask,
        )
        results_by_gap: dict[str, dict] = {}
        for g in gap_list:
            summ_g, recs_g = _screen_single_horizon(
                lib=lib,
                scan=scan,
                ctx=ctx,
                gap=g,
                mode=mode,
                fast_spear_pool=fast_spear_pool,
                compute_captain_attribution=compute_captain_attribution if g == 5 else False,
            )
            results_by_gap[f"gap{g}"] = {"summary": summ_g, "squads": recs_g}
        return {
            "disp": disp,
            "store_dir": store_dir,
            "mode": mode,
            "n_raw_factors": scan["n_raw"],
            "n_derived_cols": scan["m_total"],
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            **results_by_gap,
        }
    finally:
        if scan is not None:
            try:
                scan["u16_bank"].clear()
            except Exception:
                pass
            try:
                lib.sf_close_direct_store(ctypes.c_void_p(scan["store_handle"]))
            except Exception:
                pass
            del scan
        _trim_malloc()


def format_super_factor_names_and_descriptions(
    recipes: list[dict],
    gap_tag: str,
) -> tuple[list[str], list[str], dict[str, str]]:
    """根据入选超因子配方生成标准因子名 (`sf5_01_fix_m4__xxx_smooth_1`)、源名与 Word 文档中文定义映射。"""
    sel_names: list[str] = []
    base_names: list[str] = []
    desc_map: dict[str, str] = {}
    for gi, sq in enumerate(recipes):
        trk = str(sq["track"])
        trk_short = "fix" if trk == "FixedCal" else "trn"
        trk_cn = "固定日历窗轨[5,10,20]" if trk == "FixedCal" else "累积换手等权窗轨[15%,30%,60%]"
        lead_raw = sanitize_lead_name(str(sq["members"][0]["raw_name"]))
        bnm = f"{gap_tag}_{gi + 1:02d}_{trk_short}_m{len(sq['members'])}__{lead_raw}"
        fnm = f"{bnm}_smooth_1"
        sel_names.append(fnm)
        base_names.append(bnm)

        parts = []
        for mb in sq["members"]:
            sgn_s = "+" if int(mb["sign"]) > 0 else "-"
            parts.append(f"{sgn_s}{float(mb['weight']):.1f}×{mb['derived_name']}")
        role_tag = sq.get("role", "")
        role_cn = (
            "头部高IC尖刀 · "
            if role_tag == "Spearhead_HighIC"
            else ("纯ICIR稳健主力 · " if role_tag == "Core_PureICIR" else "")
        )
        desc_map[fnm] = f"【{role_cn}{trk_cn} · {len(sq['members'])}成员超因子】 " + "  ".join(parts)
    return sel_names, base_names, desc_map


def materialize_super_recipes_from_store(
    store_dir: str,
    dates: np.ndarray,
    stocks: Iterable[str],
    win_len_turn: np.ndarray,
    gap5_recipes: list[dict] | None = None,
    gap1_recipes: list[dict] | None = None,
    store_factor_names: list[str] | None = None,
    n_threads: int = 32,
) -> dict[str, dict]:
    """仅从 `factor_store` 按需抽取入选配方涉及的约百余列原始因子，毫秒级合成全部超因子 `(K, T, N)` 矩阵。"""
    from rust_pyfunc import factor_store_v5_info

    rec5 = list(gap5_recipes or [])
    rec1 = list(gap1_recipes or [])
    dates_i32 = np.ascontiguousarray(dates, dtype=np.int32)
    stocks_list = [str(s) for s in stocks]
    t_len, n_stocks = len(dates_i32), len(stocks_list)

    uniq_raw_names: list[str] = []
    raw_name_to_local: dict[str, int] = {}
    for sq_list in (rec5, rec1):
        for sq in sq_list:
            for mb in sq["members"]:
                rnm = str(mb["raw_name"])
                if rnm not in raw_name_to_local:
                    raw_name_to_local[rnm] = len(uniq_raw_names)
                    uniq_raw_names.append(rnm)

    if not uniq_raw_names:
        empty_m = np.zeros((0, t_len, n_stocks), dtype=np.float32)
        return {
            "gap5": {"selected_names": [], "base_names": [], "descriptions": {}, "matrices": empty_m},
            "gap1": {"selected_names": [], "base_names": [], "descriptions": {}, "matrices": empty_m},
        }

    all_st_names = (
        store_factor_names
        if store_factor_names is not None
        else [str(nm) for nm in factor_store_v5_info(store_dir)["factor_names"]]
    )
    st_name_to_col = {nm: idx for idx, nm in enumerate(all_st_names)}
    store_col_idxs = np.asarray([st_name_to_col[rnm] for rnm in uniq_raw_names], dtype=np.int32)

    lib = load_super_factor_cdll(n_threads=n_threads)
    n_u = len(uniq_raw_names)
    raw_u16 = np.empty((n_u, t_len, n_stocks), dtype=np.uint16)
    fold_u16 = np.empty((n_u, t_len, n_stocks), dtype=np.uint16)

    stride_s = 16
    stocks_flat = np.zeros((n_stocks, stride_s), dtype=np.uint8)
    for si_b, s_str in enumerate(stocks_list):
        bs = s_str.split(".")[0].encode("utf-8")[:stride_s]
        stocks_flat[si_b, : len(bs)] = list(bs)
    stocks_flat = np.ascontiguousarray(stocks_flat)

    h_store = lib.sf_open_direct_store(
        store_dir.encode("utf-8"),
        ptr_i32(dates_i32),
        t_len,
        ptr_u8(stocks_flat),
        stride_s,
        n_stocks,
    )
    if not h_store:
        raise RuntimeError(f"sf_open_direct_store 打开失败: {store_dir}")
    try:
        rc = lib.sf_load_and_rank_store_cols_u16(
            ctypes.c_void_p(h_store),
            ptr_i32(store_col_idxs),
            n_u,
            ptr_u16(raw_u16),
            ptr_u16(fold_u16),
        )
        if rc != 0:
            raise RuntimeError(f"sf_load_and_rank_store_cols_u16 返回异常码 {rc}")
    finally:
        lib.sf_close_direct_store(ctypes.c_void_p(h_store))

    win_u8 = np.ascontiguousarray(win_len_turn, dtype=np.uint8)
    all_t = np.arange(t_len, dtype=np.int32)

    def _mat_horizon(rec_list: list[dict], gap_tag: str) -> dict:
        if not rec_list:
            return {
                "selected_names": [],
                "base_names": [],
                "descriptions": {},
                "matrices": np.zeros((0, t_len, n_stocks), dtype=np.float32),
            }
        uniq_face_keys: list[tuple[int, int, int, int]] = []
        fkey_to_idx: dict[tuple[int, int, int, int], int] = {}
        for sq in rec_list:
            for mb in sq["members"]:
                fk = (
                    raw_name_to_local[str(mb["raw_name"])],
                    int(mb["var_id"]),
                    int(mb["slot_id"]),
                    int(mb["sign"]),
                )
                if fk not in fkey_to_idx:
                    fkey_to_idx[fk] = len(uniq_face_keys)
                    uniq_face_keys.append(fk)

        l_f = len(uniq_face_keys)
        specs = np.asarray(uniq_face_keys, dtype=np.int32)
        u_uniq = np.empty((l_f, t_len, n_stocks), dtype=np.float32)
        lib.sf_materialize_faces_u16(
            ptr_u16(raw_u16), ptr_u16(fold_u16), n_u, t_len, n_stocks,
            ptr_u8(win_u8), 0, ptr_i32(all_t), t_len,
            None, 0, None, None, 0, None,
            ptr_i32(specs), l_f, ptr_f32(u_uniq),
        )

        n_sq = len(rec_list)
        sq_offsets = np.zeros(n_sq + 1, dtype=np.int32)
        u_idx_l, w_l = [], []
        for gi, sq in enumerate(rec_list):
            sq_offsets[gi] = len(u_idx_l)
            for mb in sq["members"]:
                fk = (
                    raw_name_to_local[str(mb["raw_name"])],
                    int(mb["var_id"]),
                    int(mb["slot_id"]),
                    int(mb["sign"]),
                )
                u_idx_l.append(fkey_to_idx[fk])
                w_l.append(float(mb["weight"]))
        sq_offsets[n_sq] = len(u_idx_l)

        raw_k46 = np.empty((n_sq + 1, t_len, n_stocks), dtype=np.float32)
        lib.sf_combine_squads_from_uniq(
            ptr_f32(u_uniq), l_f, t_len, n_stocks,
            ptr_i32(sq_offsets), ptr_i32(np.asarray(u_idx_l, dtype=np.int32)),
            ptr_f32(np.asarray(w_l, dtype=np.float32)), n_sq, ptr_f32(raw_k46),
        )
        sel_names, base_names, desc_map = format_super_factor_names_and_descriptions(rec_list, gap_tag)
        return {
            "selected_names": sel_names,
            "base_names": base_names,
            "descriptions": desc_map,
            "matrices": raw_k46[:n_sq].copy(),
        }

    out5 = _mat_horizon(rec5, "sf5")
    out1 = _mat_horizon(rec1, "sf1")
    del raw_u16, fold_u16
    _trim_malloc()
    return {"gap5": out5, "gap1": out1}
