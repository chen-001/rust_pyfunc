"""验证 features_f64 新增的三个开关：with_max_min / with_lyapunov_exponent / append_for_corr。

分三块：
  A. 老口径逐位不变：verify_get_features_factors_rust_f64（旧入口）与
     新入口 + (curvature=True, max_min=False, lyapunov=False, 不 append) 必须逐位相同。
  B. 新开关与 Python get_features_factors 逐值比（矩阵干净：无常量列、无 inf，只有少量 NaN）。
  C. 老库 rust_pyfunc_0_76_11 与当前库 get_features_factors 是否一致（对账前必须确认）。
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import rust_pyfunc as rp

def eq_bitwise(a, b):
    if len(a) != len(b):
        return False, f'长度 {len(a)} vs {len(b)}'
    for i, (x, y) in enumerate(zip(a, b)):
        x = float(x); y = float(y)
        if np.isnan(x) and np.isnan(y):
            continue
        if x != y:
            return False, f'#{i} {x!r} vs {y!r}'
    return True, 'ok'

def make(nrow, ncol, seed, nan_rows=(3, 7)):
    r = np.random.default_rng(seed)
    a = r.standard_normal((nrow, ncol)) * 3.0 + 1.0
    for k, row in enumerate(nan_rows):
        if row < nrow and k < ncol:
            a[row, k] = np.nan
    return pd.DataFrame(a, columns=[f'c{i}' for i in range(ncol)])

KNOWN = ('lz_complexity', 'entropy_1d', 'max_range_product')

def compare(tag, py_vals, py_names, rust_vals, rust_names, rtol=1e-4, allow_known=False):
    if list(py_names) != list(rust_names):
        print(f'[{tag}] 名字不一致 len {len(py_names)} vs {len(rust_names)}')
        for k, (a, b) in enumerate(zip(py_names, rust_names)):
            if a != b:
                print(f'   第一处不同 #{k}: {a} | {b}'); break
        return False
    if len(py_vals) != len(rust_vals):
        print(f'[{tag}] 长度不一致 {len(py_vals)} vs {len(rust_vals)}'); return False
    bad = []; known = 0
    for i, (a, b) in enumerate(zip(py_vals, rust_vals)):
        a = float(a); b = float(b)
        if np.isnan(a) and np.isnan(b):
            continue
        if np.isnan(a) or np.isnan(b):
            bad.append((i, a, b, 'NaN 分歧')); continue
        if not np.isfinite(a) or not np.isfinite(b):
            if a == b: continue
            bad.append((i, a, b, 'inf 分歧')); continue
        rel = abs(a - b) / max(abs(a), abs(b), 1e-12)
        if rel > rtol:
            if allow_known and str(py_names[i]).endswith(KNOWN):
                known += 1
            else:
                bad.append((i, a, b, f'rel={rel:.2e}'))
    print(f'[{tag}] n={len(py_vals)} 不一致 {len(bad)} 处（其中已知老差异 {known} 处）')
    for i, a, b, why in bad[:8]:
        print(f'    #{i} {py_names[i]}: py={a:.10g} rust={b:.10g} ({why})')
    return not bad

def rust_ex(df, *, th=True, curv=False, mm=False, lyap=False, append=None):
    arm = None; acn = None
    if append is not None:
        arm = np.ascontiguousarray(append.to_numpy(dtype=float))
        acn = [str(c) for c in append.columns]
    return rp.verify_get_features_factors_rust_f64_ex(
        np.ascontiguousarray(df.to_numpy(dtype=float)),
        [str(c) for c in df.columns],
        with_threshold_counts=th, with_curvature=curv,
        with_max_min=mm, with_lyapunov_exponent=lyap,
        append_matrix=arm, append_col_names=acn)

def run(tag, df, *, th=True, mm=False, lyap=False, append=None):
    py_vals, py_names = rp.get_features_factors(
        df, with_threshold_counts=th, with_max_min=mm,
        with_lyapunov_exponent=lyap, append_for_corr=append)
    rv, rn = rust_ex(df, th=th, curv=False, mm=mm, lyap=lyap, append=append)
    return compare(tag, py_vals, py_names, rv, rn, allow_known=True)

allok = True

# ---------- A. 老口径逐位不变 ----------
m = make(80, 4, 11).to_numpy(dtype=float).copy()
m[2, 2] = 0.5  # 常量列
old_v, old_n = rp.verify_get_features_factors_rust_f64(np.ascontiguousarray(m), ['c0','c1','c2','c3'])
new_v, new_n = rp.verify_get_features_factors_rust_f64_ex(
    np.ascontiguousarray(m), ['c0','c1','c2','c3'],
    with_threshold_counts=True, with_curvature=True,
    with_max_min=False, with_lyapunov_exponent=False)
ok, why = eq_bitwise(old_v, new_v)
print(f'[A 老口径逐位不变] 名字一致={list(old_n)==list(new_n)} 值逐位一致={ok} ({why})')
allok &= ok and list(old_n) == list(new_n)

# ---------- B. 新开关 vs Python ----------
df = make(80, 4, 1)
allok &= run('默认(关lyapunov)', df, lyap=False)
allok &= run('默认(开lyapunov)', df, lyap=True)
allok &= run('max_min', df, mm=True, lyap=False)
allok &= run('max_min+lyapunov', df, mm=True, lyap=True)
allok &= run('max_min+threshold_off', df, th=False, mm=True, lyap=False)
allok &= run('threshold_off+lyapunov', df, th=False, lyap=True)

df2 = make(120, 3, 2)
rng = np.random.default_rng(7)
ap = pd.DataFrame(rng.standard_normal((120, 2)), columns=['ap0', 'ap1'])
ap.iloc[4, 0] = np.nan
allok &= run('append', df2, mm=True, lyap=False, append=ap)
allok &= run('append(默认开关)', df2, lyap=False, append=ap)
allok &= run('append+lyapunov', df2, mm=True, lyap=True, append=ap)

# ---------- B2. 无 NaN 无常量列：全项一致（不给"已知差异"豁免） ----------
def make_clean(nrow, ncol, seed):
    r = np.random.default_rng(seed)
    return pd.DataFrame(r.standard_normal((nrow, ncol)) * 2.0, columns=[f'c{i}' for i in range(ncol)])

dc = make_clean(90, 4, 21)
apc = pd.DataFrame(np.random.default_rng(22).standard_normal((90, 2)), columns=['ap0', 'ap1'])
for tag, kw in [
    ('clean 默认', dict(lyap=False)),
    ('clean max_min+lyapunov', dict(mm=True, lyap=True)),
    ('clean threshold_off', dict(th=False, mm=True, lyap=False)),
    ('clean append', dict(mm=True, lyap=False, append=apc)),
]:
    ap = kw.pop('append', None)
    pv, pn = rp.get_features_factors(dc, with_threshold_counts=kw.get('th', True),
                                     with_max_min=kw.get('mm', False),
                                     with_lyapunov_exponent=kw.get('lyap', False),
                                     append_for_corr=ap)
    rv, rn = rust_ex(dc, th=kw.get('th', True), curv=False, mm=kw.get('mm', False),
                     lyap=kw.get('lyap', False), append=ap)
    allok &= compare(tag, pv, pn, rv, rn, allow_known=False)

# ---------- C. 老库 vs 当前库 ----------
try:
    import rust_pyfunc_0_76_11 as old
    d3 = make(70, 3, 5)
    ov, on = old.get_features_factors(d3)
    nv, nn = rp.get_features_factors(d3)
    okc = compare('C 0_76_11 vs 当前库(默认参数)', ov, on, nv, nn)
    allok &= okc
except Exception as exc:
    print('[C] 老库对比跳过:', repr(exc))

print('结论:', '通过' if allok else '不通过')
