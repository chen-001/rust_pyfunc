"""第三版验收（只读 Rust 产物）：逐字节、确定性、撤单空值、2016 读取修复、覆盖、大小。
用法: python validate_v3.py
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"
import filecmp
import hashlib
import json
import numpy as np

R = "results/v3full"


def load(d, date, eng):
    b = f"{R}/{d}/v3_{date}_{eng}"
    names = json.load(open(b + ".names.json"))
    codes = json.load(open(b + ".codes.json"))
    X = np.fromfile(b + ".f32", dtype="<f4").reshape(len(codes), len(names))
    return names, codes, X, b


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


print("== 逐字节：提速版 vs 冻结基准（同一天全部股票）==")
for date in (20241231, 20160630, 20211105):
    pb = f"{R}/base/v3_{date}_base"
    pf = f"{R}/fast1/v3_{date}_fast"
    if not os.path.exists(pb + ".f32") or not os.path.exists(pf + ".f32"):
        print(date, "缺文件")
        continue
    same = filecmp.cmp(pb + ".f32", pf + ".f32", shallow=False)
    same_n = json.load(open(pb + ".names.json")) == json.load(open(pf + ".names.json"))
    same_c = json.load(open(pb + ".codes.json")) == json.load(open(pf + ".codes.json"))
    print(date, "值逐字节相同", same, "列名相同", same_n, "股票相同", same_c, "sha", sha(pf + ".f32"))
print("== 确定性：提速版两次运行 ==")
a, b = f"{R}/fast1/v3_20241231_fast.f32", f"{R}/fast2/v3_20241231_fast.f32"
print("20241231 两次逐字节相同", filecmp.cmp(a, b, shallow=False), sha(a), sha(b))

for date in (20241231, 20160630, 20211105):
    names, codes, X, base = load("fast1", date, "fast")
    print(f"\n== {date}: 股票 {len(codes)} 列 {len(names)} 文件 {os.path.getsize(base + '.f32') / 1e6:.1f} MB ==")
    codes = np.array(codes)
    sh = np.array([c.startswith("6") for c in codes])
    sz = np.array([c[0] in "03" for c in codes])
    cx = np.array(["cxl" in n for n in names])
    fin = np.isfinite(X)
    print("含 cxl 列", int(cx.sum()), "| 沪市股票", int(sh.sum()), "深市股票", int(sz.sum()))
    print("cxl 列有值的格子：沪市", int(fin[np.ix_(sh, cx)].sum()), "深市", int(fin[np.ix_(sz, cx)].sum()))
    print("cxl 列里写 0 的格子（应为空值而非 0 的检查，仅沪市 20211105 前）:",
          int((X[np.ix_(sh, cx)] == 0).sum()) if date < 20211105 else "不适用")
    print("非 cxl 列覆盖率中位：沪市", round(float(np.median(fin[np.ix_(sh, ~cx)].mean(0))), 3),
          "深市", round(float(np.median(fin[np.ix_(sz, ~cx)].mean(0))), 3))
    for g in ("mir", "wd", "var", "xday", "ind", "csrank", "peerx", "naive", "audit"):
        idx = [i for i, n in enumerate(names) if n.split("_")[0] == g]
        c = fin[:, idx].mean(0)
        print(f"  {g:7s} 列数 {len(idx):5d} 覆盖率中位 {np.median(c):.3f} 最低 {c.min():.3f} 全空列 {int((c == 0).sum())}")
    if "000001" in codes:
        r = X[list(codes).index("000001")]
        print("000001 有值列数", int(np.isfinite(r).sum()), "/", len(names), "cxl 有值", int(np.isfinite(r[cx]).sum()))
    else:
        print("000001 不在股票列表")
