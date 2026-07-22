import time

import numpy as np
import pandas as pd
import rust_pyfunc as rp


DATE = 20251231
DATES = [
    20251218, 20251219, 20251222, 20251223, 20251224,
    20251225, 20251226, 20251229, 20251230, 20251231,
]


def as_frame(func, names, date):
    codes, values = func(date)
    data = np.asarray(values, dtype=np.float64).reshape(len(codes), len(names))
    return pd.DataFrame(data, index=pd.Index(codes, name="code"), columns=names)


def test_single_minute_against_numpy():
    minute = 100
    names = rp.py_minute_capm_at_names()
    codes, values = rp.py_minute_capm_at(DATE, minute)
    got = pd.DataFrame(
        np.asarray(values).reshape(len(codes), len(names)), index=codes, columns=names
    )
    close_codes, flat, rows, cols = rp.py_read_minute_data("close", DATE)
    close = np.asarray(flat).reshape(rows, cols)
    code_to_col = {code: i for i, code in enumerate(close_codes)}
    positions = np.fromiter((code_to_col[code] for code in codes), int)
    ret = close[minute, positions] / close[minute - 1, positions] - 1.0
    beta = got["rolling_beta"].to_numpy()
    valid = np.isfinite(ret) & (np.abs(ret) <= 0.30) & np.isfinite(beta)
    x, y = beta[valid], ret[valid]
    mx, my = x.mean(), y.mean()
    sxx = np.square(x - mx).sum()
    slope = ((x - mx) * (y - my)).sum() / sxx
    intercept = my - slope * mx
    fitted = intercept + slope * x
    residual = y - fitted
    leverage = 1.0 / len(x) + np.square(x - mx) / sxx
    sample = got.loc[np.asarray(codes)[valid]]
    np.testing.assert_allclose(sample["capm_intercept"], intercept, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(sample["capm_beta_premium"], slope, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(sample["capm_fitted_return"], fitted, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(sample["capm_residual"], residual, rtol=2e-5, atol=2e-8)
    np.testing.assert_allclose(sample["capm_leverage"], leverage, rtol=2e-5, atol=2e-8)


def test_daily_identities_and_selection():
    all_names = rp.py_minute_capm_all_names()
    selected_names = rp.py_minute_capm_names()
    all_frame = as_frame(rp.py_minute_capm_all, all_names, DATE)
    selected = as_frame(rp.py_minute_capm, selected_names, DATE)
    pd.testing.assert_frame_equal(selected, all_frame[selected_names])
    np.testing.assert_allclose(
        all_frame["minute_capm_residual_mean"],
        all_frame["minute_capm_positive_residual_mean"]
        + all_frame["minute_capm_negative_residual_mean"],
        rtol=2e-5,
        atol=2e-8,
    )
    np.testing.assert_allclose(
        all_frame["minute_capm_abs_residual_mean"],
        all_frame["minute_capm_positive_residual_mean"]
        - all_frame["minute_capm_negative_residual_mean"],
        rtol=2e-5,
        atol=2e-8,
    )


def analyze_correlations():
    names = rp.py_minute_capm_all_names()
    started = time.perf_counter()
    frames = [as_frame(rp.py_minute_capm_all, names, date) for date in DATES]
    stock_means = pd.concat(frames, keys=DATES, names=["date", "code"]).groupby("code").mean()
    individual = names[:22]
    corr = stock_means[individual].corr(method="spearman")
    pairs = []
    for i, left in enumerate(individual):
        for right in individual[i + 1 :]:
            value = corr.loc[left, right]
            if np.isfinite(value) and abs(value) >= 0.95:
                pairs.append((abs(value), value, left, right))
    print(f"stocks={len(stock_means)}, elapsed={time.perf_counter() - started:.3f}s")
    print("cross_stock_metric_means:")
    print(stock_means.mean().to_string())
    print("high_spearman_pairs_abs_ge_0.95:")
    for _, value, left, right in sorted(pairs, reverse=True):
        print(f"{value:+.6f}  {left}  <>  {right}")


if __name__ == "__main__":
    test_single_minute_against_numpy()
    test_daily_identities_and_selection()
    analyze_correlations()
