"""P0/P1 修复的验证测试（组合 store / 读取闭环 / 判定器 pairwise 样本数）。

运行: python -m pytest tests/test_supplement_fixes.py -q
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

rp = pytest.importorskip("rust_pyfunc")
dw = pytest.importorskip("design_whatever")

PROD_STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_cross_yhyb"


def _make_tiny_style_parquet(path: Path, dates: list, codes: list) -> str:
    """造一个最小合法风格 parquet（date + code + 41 列浮点），避免测试加载 840MB 生产文件。

    41 列用互不相关的随机值（全零会导致回归矩阵退化、日期转换失败）。
    """
    rng = np.random.default_rng(11)
    rows = pd.DataFrame(
        [{"date": int(d), "code": c} for d in dates for c in codes]
    )
    for i in range(41):
        rows[f"c{i}"] = rng.standard_normal(len(rows)) + i * 0.1
    rows.to_parquet(path, index=False)
    return str(path)


# ---------- P1-8: copy_subset（accepted group 构造） ----------

def test_copy_subset_and_guards():
    root = Path(tempfile.mkdtemp(prefix="copy_subset_"))
    src = root / "src"
    src.mkdir()
    assert rp.factor_store_v5_smoke_proj(str(src), 24, 3)[0]

    dst = root / "dst"
    info = rp.factor_store_v5_copy_subset(str(src), str(dst), ["f1", "f2"])
    assert info["factor_count"] == 2
    assert info["is_projected"] is True
    assert info["factor_names"] == ["f1", "f2"]

    # 值与源完全一致（三元组逐项相等）
    a = rp.factor_store_v5_read_factor(str(src), 1)
    b = rp.factor_store_v5_read_factor(str(dst), 0)
    np.testing.assert_array_equal(a["date_id"], b["date_id"])
    np.testing.assert_array_equal(a["code_id"], b["code_id"])
    np.testing.assert_allclose(a["factor"], b["factor"], rtol=0, atol=0)

    # 请求名不在 src → 报错，不静默漏复制
    with pytest.raises(ValueError, match="不在 src store"):
        rp.factor_store_v5_copy_subset(str(src), str(root / "dst2"), ["nope"])

    # dst 非空 → 拒绝覆盖
    dst3 = root / "dst3"
    dst3.mkdir()
    (dst3 / "x").write_text("x")
    with pytest.raises(ValueError, match="拒绝覆盖"):
        rp.factor_store_v5_copy_subset(str(src), str(dst3), ["f0"])

    # 组合 store 根目录（有 factor_groups.json）→ 拒绝
    combo = root / "combo"
    combo.mkdir()
    (combo / "factor_groups.json").write_text("{}")
    with pytest.raises(ValueError, match="组合 store 根目录"):
        rp.factor_store_v5_copy_subset(str(combo), str(root / "dst4"), ["f0"])

    # dst 已被组合 manifest 引用 → 拒绝（防覆盖已注册 group 布局）
    combo2 = root / "combo2"
    (combo2 / "taken_dir").mkdir(parents=True)
    (combo2 / "factor_groups.json").write_text(
        '{"version": 1, "groups": [{"name": "base", "dir": "."}, {"name": "taken", "dir": "taken_dir"}]}'
    )
    with pytest.raises(ValueError, match="已被组合 manifest 注册"):
        rp.factor_store_v5_copy_subset(str(src), str(combo2 / "taken_dir"), ["f0"])


# ---------- P1-9: update_mode=False 的 force_clear 安全闸门 ----------

def test_cross_section_force_clear_gate():
    d = Path(tempfile.mkdtemp(prefix="force_clear_"))
    store = d / "existing"
    store.mkdir()
    (store / "factors.idx").write_bytes(b"x")
    common = dict(
        pipeline="cross_section_example",
        tasks=[],
        n_jobs=2,
        expected_result_length=1,
        trading_days=[20230103],
        store_dir=str(store),
        store_factor_names=["f0"],
        update_mode=False,
    )
    # 非空目录 + 未显式 force_clear → 拒绝
    with pytest.raises(ValueError, match="拒绝清空重建"):
        rp.run_factor_pipeline_cross_section(**common)
    # 显式 force_clear=True → 允许重建
    rp.run_factor_pipeline_cross_section(force_clear=True, **common)


# ---------- P0-1: 读取失败必须带因子名报错，不得静默遗漏 ----------

@pytest.mark.skipif(
    not os.path.isdir(PROD_STORE), reason="生产 colblk store 不存在"
)
def test_engine_read_failure_not_silent():
    tmp = Path(tempfile.mkdtemp(prefix="read_closure_"))
    tmpl = rp.factor_store_v5_template(PROD_STORE)
    dates = [int(d) for d in tmpl["dates"][:30]]
    stocks = [str(s) for s in tmpl["stocks"][:100]]
    bare_codes = [s.split(".")[0] for s in stocks]
    style_path = _make_tiny_style_parquet(
        tmp / "style.parquet", dates[:5], bare_codes[:30]
    )
    n_dates, n_stocks = len(dates), len(stocks)
    inputs = tmp / "inputs"
    inputs.mkdir()
    rng = np.random.default_rng(7)
    np.save(inputs / "ret_gap1.npy", rng.standard_normal((n_dates, n_stocks)).astype(np.float32))
    np.save(inputs / "ret_sum_gap1.npy", rng.standard_normal((n_dates, n_stocks)).astype(np.float32))
    np.save(inputs / "ret_gap5.npy", rng.standard_normal((n_dates, n_stocks)).astype(np.float32))
    np.save(inputs / "ret_sum_gap5.npy", rng.standard_normal((n_dates, n_stocks)).astype(np.float32))
    np.save(inputs / "restrict.npy", np.zeros((n_dates, n_stocks), dtype=np.float32))
    np.save(inputs / "index_ret.npy", rng.standard_normal(n_dates).astype(np.float32))

    def run(factor_path):
        return rp.tail_backtest_engine(
            colblk_store_dir=PROD_STORE,
            factor_names=["__bad_factor__"],
            factor_paths=[factor_path],
            dates=dates,
            stocks=stocks,
            windows=[5],
            fold=False,
            n_jobs=2,
            min_valid=5,
            cache_root=str(tmp / "cache"),
            style_data_path=style_path,
            ret_gap1_path=str(inputs / "ret_gap1.npy"),
            ret_sum_gap1_path=str(inputs / "ret_sum_gap1.npy"),
            ret_gap5_path=str(inputs / "ret_gap5.npy"),
            ret_sum_gap5_path=str(inputs / "ret_sum_gap5.npy"),
            restrict_path=str(inputs / "restrict.npy"),
            index_ret_path=str(inputs / "index_ret.npy"),
            backtest_start=dates[0],
            cover_rate=0.5,
            ret_point_neu_gap5=0.055,
            ret_point_neu_gap1=0.08,
            ic_point_neu_gap5=0.01,
            ic_point_neu_gap1=0.006,
            ret_point_gap5=0.1,
            ret_point_gap1=0.13,
            ic_point_gap5=0.03,
            ic_point_gap1=0.02,
            ic_more_important_gap5=0.02,
            ic_more_important_gap1=0.012,
            majority_count_threshold=10000.0,
            zero_max_threshold=0.12,
            nan_max_threshold=0.04,
            industry_neutralize=False,
            industry_matrix=np.zeros((n_dates, n_stocks), dtype=np.float64),
        )

    # col_idx 解析失败 → 带因子名报错
    with pytest.raises(RuntimeError, match="__bad_factor__.*解析 col_idx 失败"):
        run("no_colon_here")
    # 读取失败（col_idx 越界）→ 带因子名报错
    with pytest.raises(RuntimeError, match="__bad_factor__.*读取因子失败"):
        run(f"{PROD_STORE}::99999999")


# ---------- P0-3: 相关性阻挡用 pairwise 共同有效样本数 ----------

def _write_synthetic_cache(root: Path, *, metrics_only: bool) -> None:
    from design_whatever.tail_v3 import _resolve_selection_config

    meta = root / "meta"
    metrics = root / "metrics"
    ic = root / "ic_ts"
    selected = root / "selected"
    for d in (meta, metrics, ic, selected):
        d.mkdir(parents=True, exist_ok=True)

    dates = np.arange(100, dtype=np.int32)  # 100 个日期槽位
    np.save(meta / "dates.npy", dates)

    cfg = dict(
        logic_version="legacy_backtest_v4_exact_neu_sorted_codes",
        config_version=2,
        engine_logic_version="tail_backtest_engine_v7_metrics_v3",
        ver="synthetic",
        source_dir="/tmp/synthetic_store",
        windows=[5, 10, 20],
        fold=True,
        start_date="2015-01-01",
        backtest_start_date="2015-02-01",
        end_date="2026-07-17",
        selection_kwargs=_resolve_selection_config(
            {"cover_rate": 0.97, "ret_point_neu_gap5": 0.055}
        ),
        preflight={
            "majority_count_threshold": 10000.0,
            "zero_max_threshold": 0.12,
            "nan_max_threshold": 0.04,
        },
        metrics_only=metrics_only,
        industry_neutralize=True,
        index_name="000905",
        style_data_path="/tmp/style.parquet",
        min_valid=12,
        industry_data_path="/tmp/industry.csv",
        engine_build_sha256="abc123",
        engine_repo_commit=None,
    )
    (meta / "tail_v4_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False), encoding="utf-8"
    )
    (meta / "input_fingerprints.json").write_text(
        json.dumps({"style_data_path": {"sha256": "f" * 64, "size": 1}}),
        encoding="utf-8",
    )


def test_pairwise_corr_n_obs():
    root = Path(tempfile.mkdtemp(prefix="pairwise_"))
    base = root / "base"
    supp = root / "supp"
    rng = np.random.default_rng(3)

    # ---- 初版：5 个入选因子 ----
    # B1/B2/B5 全程有效（neu_ret 通道）；B4 只有 50 个有效日（稀疏初版因子）；
    # B3 是唯一"纯 neu_IC 入选"（下限 = abs(b3.mean()) ≈ 0.02）。
    _write_synthetic_cache(base, metrics_only=False)
    dates = np.load(base / "meta" / "dates.npy")
    u = rng.normal(0, 1, 100)
    w = rng.normal(0, 1, 100)
    b1 = 0.08 + 0.01 * u / u.std()                          # abs_IC 0.08
    b2 = 0.03 + rng.normal(0, 0.005, 100)
    b3 = 0.02 + rng.normal(0, 0.003, 100)
    b4 = np.full(100, np.nan)                               # 稀疏：仅前 50 天
    b4[:50] = 0.09 + rng.normal(0, 0.01, 50)
    b5 = 0.085 + 0.01 * w / w.std()                         # abs_IC 0.085
    np.save(base / "ic_ts" / "ic_neu_gap5.npy", np.stack([b1, b2, b3, b4, b5], axis=1))
    np.save(base / "ic_ts" / "ic_neu_gap5_dates.npy", dates)
    (base / "ic_ts" / "ic_neu_gap5_names.json").write_text(
        json.dumps(["B1", "B2", "B3", "B4", "B5"]), encoding="utf-8"
    )
    base_summary = [
        {"factor_name": n, "stage": "neu", "IC_mean": float(v), "ratio_mean": 0.99,
         "hedge_annualized_return": 0.3, "source_factor": n}
        for n, v in (("B1", b1.mean()), ("B2", b2.mean()), ("B3", b3.mean()),
                     ("B4", 0.09), ("B5", b5.mean()))
    ]
    (base / "metrics" / "summary_neu_gap5_candidates.json").write_text(
        json.dumps(base_summary), encoding="utf-8"
    )
    audit_rows = []
    for n, ic in (("B1", b1.mean()), ("B2", b2.mean()), ("B4", 0.09), ("B5", b5.mean())):
        audit_rows.append(
            {"factor_name": n, "gap": 5, "channel": "neu_ret", "also_neu_ret": True,
             "is_pure_neu_ic": False, "abs_IC_mean": abs(ic),
             "IC_mean": ic, "hedge_annualized_return": 0.3, "ratio_mean": 0.99}
        )
    audit_rows.append(
        {"factor_name": "B3", "gap": 5, "channel": "neu_ic", "also_neu_ret": False,
         "is_pure_neu_ic": True, "abs_IC_mean": abs(b3.mean()),
         "IC_mean": b3.mean(), "hedge_annualized_return": 0.3, "ratio_mean": 0.99}
    )
    pd.DataFrame(audit_rows).to_parquet(
        base / "selected" / "selection_audit.parquet", index=False
    )
    pd.DataFrame({"factor_name": ["B1", "B2", "B3", "B4", "B5"]}).to_parquet(
        base / "selected" / "gap5_selected.parquet", index=False
    )

    # ---- 补充：6 个场景 ----
    # S1: 自身 100 天有效，仅与稀疏 B4 共享 50 天（pair 样本不足不阻挡 → 通过）
    # S2: 与 B1 完全相同 → 被 B1 阻挡（n=100）
    # S3: 与 B1 高相关且 80 个有效日 → 被 B1 阻挡（n=80）
    # S4: 低于 IC 下限 → 不过第一关
    # S5: 自身只有 40 个有效日 → insufficient_ic_history
    # S6: 同时与 B1(corr≈0.6) 和 B5(corr≈0.8) 高相关 → 阻挡因子应为 B5（更高相关）
    _write_synthetic_cache(supp, metrics_only=True)
    s1 = np.full(100, np.nan)
    s1[:50] = b4[:50] + rng.normal(0, 0.001, 50)   # 与 B4 仅共享 50 天
    s1[50:] = rng.normal(0, 0.01, 50)              # 后半段独立噪声（自身样本充足）
    s2 = b1.copy()
    s3 = np.full(100, np.nan)
    s3[:80] = b1[:80] + rng.normal(0, 0.001, 80)
    s4 = 0.01 + rng.normal(0, 0.001, 100)
    s5 = np.full(100, np.nan)
    s5[:40] = 0.05 + rng.normal(0, 0.002, 40)      # 自身仅 40 个有效日
    s6 = 0.6 * u / u.std() + 0.8 * w / w.std() + rng.normal(0, 0.02, 100)
    supp_mat = np.stack([s1, s2, s3, s4, s5, s6], axis=1)
    np.save(supp / "ic_ts" / "ic_neu_gap5_all.npy", supp_mat)
    np.save(supp / "ic_ts" / "ic_neu_gap5_all_dates.npy", dates)
    (supp / "ic_ts" / "ic_neu_gap5_all_names.json").write_text(
        json.dumps(["S1", "S2", "S3", "S4", "S5", "S6"]), encoding="utf-8"
    )
    supp_summary = [
        {"factor_name": n, "stage": "neu", "source_factor": n, "IC_mean": float(v),
         "ratio_mean": 0.99, "preflight_passed": True}
        for n, v in (("S1", 0.05), ("S2", 0.05), ("S3", 0.05),
                     ("S4", 0.01), ("S5", 0.05), ("S6", 0.05))
    ]
    (supp / "metrics" / "summary_neu_gap5_all.json").write_text(
        json.dumps(supp_summary), encoding="utf-8"
    )

    result, meta = dw.evaluate_supplement_factors(
        base_ver="b", supplement_ver="s",
        base_temp_root=str(base), supplement_temp_root=str(supp),
        selection_kwargs=None,
        min_corr_obs=60,
        min_ic_obs=60,
        expected_supplement_names=["S1", "S2", "S3", "S4", "S5", "S6"],
    )
    by = {r.source_factor: r for r in result.itertuples(index=False)}
    assert meta["ic_floor"] == pytest.approx(abs(b3.mean()), abs=1e-9)
    # S1：自身样本充足、与 B4 的 pair 样本只有 50 < 60 → 不阻挡 → 通过
    assert by["S1"].worth_supplementing is True, "pair 样本不足 60 不得产生阻挡结论"
    assert by["S1"].ic_n_obs == 100
    # S2：被 B1 阻挡，阻挡对样本数 100
    assert by["S2"].worth_supplementing is False
    assert by["S2"].blocker == "high_corr_higher_ic"
    assert by["S2"].blocker_factor == "B1"
    assert by["S2"].corr_n_obs == 100, "阻挡对的 corr_n_obs 必须是 pairwise 有效点数"
    # S3：80 个样本
    assert by["S3"].corr_n_obs == 80
    assert by["S3"].worth_supplementing is False
    # S4：不过 IC 下限
    assert by["S4"].passes_ic_floor is False
    assert by["S4"].worth_supplementing is False
    # S5：自身 IC 历史不足 60 → insufficient_ic_history
    assert by["S5"].blocker == "insufficient_ic_history"
    assert by["S5"].ic_n_obs == 40
    assert by["S5"].worth_supplementing is False
    # S6：两个阻挡因子（B1 corr≈0.6、B5 corr≈0.8）→ 必须取相关性更高的 B5
    assert by["S6"].blocker == "high_corr_higher_ic"
    assert by["S6"].blocker_factor == "B5", "必须选相关性更高的阻挡因子"
    assert by["S6"].worth_supplementing is False

    # 外部 selection_kwargs 与基线 config 不一致 → 直接报错
    with pytest.raises(ValueError, match="不一致"):
        dw.evaluate_supplement_factors(
            base_ver="b", supplement_ver="s",
            base_temp_root=str(base), supplement_temp_root=str(supp),
            selection_kwargs={"cover_rate": 0.5},
            expected_supplement_names=["S1"],
        )
