"""tail_v4_fulltest_rust 自测：Python 参考实现 vs Rust 实现，逐位对账 + 计时。

用法：
    python tests/selfcheck_tail_v4_fulltest.py            # 合成用例 + 真实数据（若在）
    python tests/selfcheck_tail_v4_fulltest.py 400        # 真实数据只取前 400 个交易日

Python 参考实现 = fulltest_whatever.py 第 1804-1882 行的逐行转写（含 rankdata_nonmiss /
calc_spearman_correlation 的原始定义），它就是要对齐的规格。
"""

import sys
import numpy as np
import pandas as pd
from scipy.stats import rankdata

import rust_pyfunc as rp

DATA_DIR = "/home/chenzongwei/neu_lab/data_yupei_real"


# --------------------------------------------------------------------------
# 规格：fulltest_whatever.py 的原函数（逐行转写）
# --------------------------------------------------------------------------
def rankdata_nonmiss(arr):
    if len(arr) == 0:
        return np.array([])
    rank_arr = arr.copy()
    rank_arr[:] = np.nan
    cond = ~np.isnan(arr) & ~np.isinf(arr)
    rank_arr[cond] = rankdata(arr[cond])
    return rank_arr


def calc_spearman_correlation(x, y):
    n = x.size
    xx = x.argsort().argsort()
    yy = y.argsort().argsort()
    diff = xx - yy
    return 1.0 - 6.0 * np.dot(diff, diff) / (n * (n * n - 1))


def py_ref(signal, ret, ret_sum, restrict, dates, portf_num, H_hori, unique_values=None):
    """signal/ret/ret_sum/restrict: (N_stocks, T_dates) float64。"""
    const_signal_arr = signal
    ret_arr = ret
    ret_arr_sum = ret_sum
    S_restrict_arr = restrict
    if unique_values is None:
        unique_values = np.unique(const_signal_arr[~np.isnan(const_signal_arr)])

    portf_ret = []
    ic_value_list, ic_date_list = [], []
    stock_num_list, stock_num_date_list = [], []
    signal_ori = restrict_ori = None
    for t in range(const_signal_arr.shape[1]):
        if (t % H_hori) == 0:
            signal_ori = const_signal_arr[:, t]
            restrict_ori = S_restrict_arr[:, t]
        ret_arr_d_o = ret_arr[:, t]
        signal_arr_d_o = signal_ori.copy()
        restrict_arr_d_o = restrict_ori.copy()
        b1 = ~np.isnan(signal_arr_d_o)
        b2 = ~np.isnan(ret_arr_d_o)
        b3 = restrict_arr_d_o == 0
        ret_arr_d = ret_arr_d_o[b1 & b2 & b3]
        signal_arr_d = signal_arr_d_o[b1 & b2 & b3]
        if ((t + 1) % H_hori) == 0:
            ret_after_h_hori = ret_arr_sum[:, t][b1 & b2 & b3]
            ic_value_list.append(calc_spearman_correlation(ret_after_h_hori, signal_arr_d))
            ic_date_list.append(dates[t])
        stocks_num = len(signal_arr_d)
        if stocks_num < portf_num:
            portf_ret.append([0 for _ in range(portf_num)])
            continue
        stock_num_list.append(stocks_num)
        stock_num_date_list.append(dates[t])
        portf_ret_d = []
        for k in range(portf_num):
            if len(unique_values) < 10:
                condition = signal_arr_d == unique_values[k]
            else:
                signal_arr_rank_d = rankdata_nonmiss(signal_arr_d) / sum(
                    ~np.isnan(signal_arr_d)
                )
                if k < portf_num - 1:
                    condition = (signal_arr_rank_d >= k / portf_num) & (
                        signal_arr_rank_d < (k + 1) / portf_num
                    )
                else:
                    condition = (signal_arr_rank_d >= k / portf_num) & (
                        signal_arr_rank_d <= (k + 1) / portf_num
                    )
            portf_ret_d.append(np.nanmean(ret_arr_d[condition]))
        portf_ret.append(portf_ret_d)

    portf_ret_arr = np.array(portf_ret).T
    portf_ret_df = pd.DataFrame(portf_ret_arr, index=range(1, portf_num + 1))
    portf_ret_df.fillna(0, inplace=True)
    return {
        "portf_ret": np.ascontiguousarray(portf_ret_df.values),
        "ic_values": list(ic_value_list),
        "ic_dates": list(ic_date_list),
        "stock_num_list": list(stock_num_list),
        "stock_num_dates": list(stock_num_date_list),
    }


# --------------------------------------------------------------------------
# 用例构造
# --------------------------------------------------------------------------
def case_synthetic(seed=20260101, T=61, N=137, H=5, P=10):
    """合成面板：并列值、NaN、±inf、停牌、有效股票不足 10 只的日子都有。"""
    rng = np.random.default_rng(seed)
    # 因子取少量离散档位 → 大量并列（考验平均秩与分组边界）
    levels = rng.choice(np.linspace(-3, 3, 40), size=(T, N))
    signal = np.where(rng.random((T, N)) < 0.05, np.nan, levels).T.copy()  # (N,T)
    signal[3, :] = np.nan
    signal[7, :8] = np.nan
    signal[11, :] = np.inf
    signal[12, 5] = -np.inf
    ret = (rng.standard_normal((T, N)) * 0.02).T
    ret[np.isnan(signal)] = np.nan
    ret[rng.random((N, T)) < 0.04] = np.nan
    ret[20, :] = np.nan          # 该日无有效股票
    ret[33, :3] = np.nan
    restrict = np.zeros((N, T))
    restrict[rng.random((N, T)) < 0.06] = 1.0
    restrict[7, 10:] = np.nan
    ret_sum = np.cumsum(ret, axis=1)
    dates = [f"2020-{(i // 21) + 1:02d}-{(i % 21) + 1:02d}" for i in range(T)]
    return signal, ret, ret_sum, restrict, dates, P, H


def case_real(n_dates, H=5, P=10, factor_idx=0):
    dates = np.load(f"{DATA_DIR}/dates.npy")
    restrict = np.load(f"{DATA_DIR}/restrict.npy").astype(np.float64)
    ret = np.load(f"{DATA_DIR}/ret_gap1.npy").astype(np.float64)
    ret_sum = np.load(f"{DATA_DIR}/ret_sum_gap1.npy").astype(np.float64)
    names = sorted(__import__("glob").glob(f"{DATA_DIR}/factor_*.npy"))
    raw = np.load(names[factor_idx]).astype(np.float64)
    t0 = 0
    sl = slice(t0, min(t0 + n_dates, raw.shape[0]))
    dates = dates[sl]
    signal = raw[sl].T.copy()
    ret = ret[sl].T.copy()
    ret_sum = ret_sum[sl].T.copy()
    restrict = restrict[sl].T.copy()
    dstr = [
        f"{int(d) // 10000:04d}-{int(d) % 10000 // 100:02d}-{int(d) % 100:02d}" for d in dates
    ]
    return signal, ret, ret_sum, restrict, dstr, P, H, names[factor_idx].split("/")[-1]


def run(title, signal, ret, ret_sum, restrict, dates, P, H, repeats=1):
    uniq = np.unique(signal[~np.isnan(signal)])
    print("=" * 78)
    print(f"{title}   shape={signal.shape}  T={len(dates)}  unique={len(uniq)}")
    print("-" * 78)
    if len(uniq) < 10:
        print(f"  unique_values={len(uniq)} < 10：原函数会提前 return，跳过")
        return
    report = rp.tail_v4_fulltest_rust_selfcheck(
        py_ref, signal, ret, ret_sum, restrict, dates, P, H, uniq, True, repeats
    )
    print(report)


def main():
    if not hasattr(rp, "tail_v4_fulltest_rust"):
        print("!! 当前安装的 rust_pyfunc 还没有 tail_v4_fulltest_rust，请先构建（bash dev.sh）")
        sys.exit(1)
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else 300

    run("合成用例（并列/NaN/Inf/停牌/有效股票不足）", *case_synthetic())
    try:
        run("合成用例 H=1（每日换仓）", *case_synthetic(seed=7, T=23, N=64, H=1))
    except Exception as e:  # noqa: BLE001
        print("H=1 用例失败:", e)

    import os

    if os.path.exists(f"{DATA_DIR}/dates.npy"):
        sig, rt, rs, rstr, ds, P, H, nm = case_real(n_real)
        run(f"真实数据 {nm} 前 {n_real} 日", sig, rt, rs, rstr, ds, P, H)
    else:
        print(f"（未找到真实数据目录 {DATA_DIR}，跳过）")


if __name__ == "__main__":
    main()
