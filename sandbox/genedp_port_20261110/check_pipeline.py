"""正式库 pipeline 端到端验收：一天全市场 → 临时直写库 → 逐列读回 → 与沙箱第三版产物逐位比较。
用法: python check_pipeline.py <系列名> <日期>
系列名: genedp_day_vs_day_align / genedp_self_repeat_align / genedp_mirror_align / genedp_buysell_fold"""
import os, sys, json, time, shutil
import numpy as np
import rust_pyfunc as rp

S, D = sys.argv[1], int(sys.argv[2])
SB = "/home/chenzongwei/rust_pyfunc"
store = f"/hdd/user_home_unsafe/chenzongwei/tmp_genedp_check_{S}_{D}_direct"
SKIP = len(sys.argv) > 3 and sys.argv[3] == "--skip-run"
if os.path.exists(store) and not SKIP:
    shutil.rmtree(store)

names = getattr(rp, f"py_{S}_names")()
print(f"{S} {D} 列数 {len(names)}，含 cxl {sum('cxl' in n for n in names)}", flush=True)

# ---- 沙箱参考产物（行=股票，列=因子）----
def ref_matrix():
    if S == "genedp_day_vs_day_align":
        meta = json.load(open(f"{SB}/sandbox_align_day_vs_day/results_v3/base_{D}.json"))
        m = np.fromfile(f"{SB}/sandbox_align_day_vs_day/results_v3/base_{D}.f32", dtype="<f4").reshape(meta["rows"], meta["cols"])
        return meta["codes"], meta["names"], m
    if S == "genedp_self_repeat_align":
        meta = json.load(open(f"{SB}/sandbox_align_self_repeat/results/v3_{D}_fast2.meta.json"))
        m = np.fromfile(f"{SB}/sandbox_align_self_repeat/results/v3_{D}_fast2.f32", dtype="<f4").reshape(meta["n_cols"], meta["n_codes"]).T
        return meta["codes"], meta["names"], m
    if S == "genedp_mirror_align":
        p = f"{SB}/sandbox_align_mirror/results/v3full/fast1/v3_{D}_fast"
        codes, nm = json.load(open(p + ".codes.json")), json.load(open(p + ".names.json"))
        return codes, nm, np.fromfile(p + ".f32", dtype="<f4").reshape(len(codes), len(nm))
    if S == "genedp_buysell_fold":
        p = f"{SB}/sandbox_align_fold/out/v3_fast/{D}"
        meta = json.load(open(p + "/meta.json"))
        return meta["codes"], meta["names"], np.fromfile(p + "/vals.bin", dtype="<f4").reshape(meta["n_rows"], meta["n_cols"])
codes_ref, names_ref, mat_ref = ref_matrix()
assert names_ref == names, "列名与沙箱不一致"
print("沙箱参考:", mat_ref.shape, flush=True)

# ---- 正式 pipeline（单 worker × 50 线程）----
t = time.time()
if not SKIP: rp.run_factor_pipeline_cross_section(
    pipeline=S, tasks=[D], n_jobs=50, n_workers=1,
    expected_result_length=len(names), trading_days=list(rp.td.trading_days),
    store_dir=store, store_factor_names=names, update_mode=True, direct_write=True,
    write_buffer_gb=32.0, bind_cores=False,
)
print(f"pipeline 用时 {time.time() - t:.1f}s", flush=True)
info = rp.factor_store_v5_info(store)
print({k: info[k] for k in info if k in ("storage_layout", "direct_segment_count", "n_factors", "n_records", "is_projected")}, flush=True)

# ---- 逐列读回比较（库里不存空值：读回缺的格子当空值；非空值按位比较）----
if len(sys.argv) > 3 and sys.argv[3] == "--skip-run":
    pass
tmpl = rp.factor_store_v5_template(store)
stocks = [s.split(".")[0] for s in tmpl["stocks"]]
assert sorted(stocks) == sorted(codes_ref), f"股票集合不同: 库 {len(stocks)} 沙箱 {len(codes_ref)}"
rows = np.array([codes_ref.index(c) for c in stocks])
t = time.time()
bad = []
STEP = int(os.environ.get("CHECK_STEP", "1"))
cols = list(range(0, len(names), STEP))
if STEP > 1:  # 抽样时再加撤单列的抽样和最后一列
    cxl = [j for j, n in enumerate(names) if "cxl" in n]
    cols = sorted(set(cols) | set(cxl[::STEP]) | {len(names) - 1})
print(f"比较 {len(cols)} 列（步长 {STEP}）", flush=True)
for j in cols:
    tri = rp.factor_store_v5_read_factor(store, j)
    got = np.full(len(stocks), np.nan, dtype=np.float32)
    got[np.asarray(tri["code_id"]).astype(int)] = np.asarray(tri["factor"], dtype=np.float32)
    ref = mat_ref[rows, j]
    nan_g, nan_r = np.isnan(got), np.isnan(ref)
    if not (np.array_equal(nan_g, nan_r) and np.array_equal(got[~nan_g].view(np.uint32), ref[~nan_r].view(np.uint32))):
        bad.append(j)
print(f"逐列比较 {len(cols)} 列 × {len(stocks)} 股，用时 {time.time() - t:.0f}s；不一致 {len(bad)} 列", [names[b] for b in bad[:10]], flush=True)
