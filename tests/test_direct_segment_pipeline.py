"""Opt-in deployed-worker and backtest integration, with every artifact on HDD.

After ./alter.sh: DIRECT_SEGMENT_PIPELINES=1 python -m pytest -p no:cacheprovider
    tests/test_direct_segment_pipeline.py -q
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_direct_segment_store import sb, work

pytestmark = pytest.mark.skipif(os.environ.get("DIRECT_SEGMENT_PIPELINES") != "1",
                                reason="requires the deployed new library and worker")


def test_real_worker_two_dates_two_stocks(work):
    import rust_pyfunc as rp

    days, codes = [20220819, 20220822], ["000001", "000002"]
    data = work / "level2"
    for date in days:
        for sub in ["transaction", "market_data"]:
            folder = data / str(date) / sub
            folder.mkdir(parents=True)
            for code in codes:
                source = Path("/ssd_data/stock") / str(date) / sub / f"{code}_{date}_{sub}.csv"
                assert source.exists(), source
                (folder / source.name).symlink_to(source)
    for kind, ncols in [("cross", 3), ("stock", 26976)]:
        baseline = None
        for direct in [False, True]:
            store = work / f"{kind}_{direct}"
            opts = dict(n_jobs=2, expected_result_length=ncols, trading_days=days,
                        bind_cores=False, store_dir=str(store), store_factor_names=[f"f{i}" for i in range(ncols)],
                        update_mode=True, data_root=str(data), direct_write=direct,
                        write_buffer_gb=0.00011)  # Force the stock case to publish multiple segments.
            if kind == "cross":
                run = rp.run_factor_pipeline_cross_section
                opts.update(pipeline="cross_section_example", tasks=days)
            else:
                run = rp.run_factor_pipeline
                opts.update(pipeline="extreme_point_fit", tasks=[[d, c] for d in days for c in codes],
                            backup_file=str(work / "unused.bin"), params={"with_curvature": False},
                            export_n_jobs=2)
            run(**opts)
            assert rp.factor_store_v5_info(str(store))["record_count"] == 4
            matrices = sb.read_matrix(str(store), list(range(ncols)), True)[2]
            if baseline is None:
                baseline = matrices
            else:
                np.testing.assert_array_equal(matrices, baseline)
                names_before = sorted(p.name for p in store.iterdir())
                run(**opts)  # Resume only published rows/dates, without duplicate segments.
                assert rp.factor_store_v5_info(str(store))["record_count"] == 4
                assert sorted(p.name for p in store.iterdir()) == names_before
                assert not any(p.name.endswith(".colblk") or "proj" in p.name or "wal" in p.name.lower()
                               for p in store.rglob("*"))
                assert not (store / "_completed_dates").exists()
        if kind == "stock":
            threaded = work / "stock_threaded"
            opts.update(store_dir=str(threaded), mode="thread")
            run(**opts)
            np.testing.assert_array_equal(sb.read_matrix(str(threaded), list(range(ncols)), True)[2], baseline)
            run(**opts)
            assert rp.factor_store_v5_info(str(threaded))["record_count"] == 4
    assert not (work / "unused.bin").exists()


def test_pipeline_preflight_preserves_existing_store(work):
    import rust_pyfunc as rp

    for cross in [False, True]:
        run = rp.run_factor_pipeline_cross_section if cross else rp.run_factor_pipeline
        target = work / f"invalid_{cross}"
        opts = dict(pipeline="cross_section_example" if cross else "extreme_point_fit",
                    tasks=[], n_jobs=1, expected_result_length=3, trading_days=[20220819],
                    store_dir=str(target), store_factor_names=["f0", "f1", "f2"],
                    direct_write=True, update_mode=True)
        if not cross:
            opts["backup_file"] = ""
        invalid = [dict(write_buffer_gb=float("nan")), dict(write_buffer_gb=0),
                   dict(store_dir=None), dict(incremental_projection=True), dict(n_jobs=0),
                   dict(store_factor_names=["wrong_length"])]
        invalid.append(dict(n_workers=0) if cross else dict(mode="wrong_mode"))
        for change in invalid:
            with pytest.raises(ValueError):
                run(**(opts | change))
            assert not target.exists()
    target = work / "existing_cross"
    writer = sb.Writer(str(target), ["f0", "f1", "f2"], 36)
    writer.append_date(20220819, ["000001"], [[1, 2, 3]])
    writer.finish()
    writer.close()
    before = (target / "segments.manifest.json").read_bytes()
    for force_clear in [False, True]:
        with pytest.raises(ValueError, match="不支持清空非空目录"):
            rp.run_factor_pipeline_cross_section(pipeline="cross_section_example", tasks=[], n_jobs=1,
                expected_result_length=3, trading_days=[20220819], store_dir=str(target),
                store_factor_names=["f0", "f1", "f2"], direct_write=True, update_mode=False,
                force_clear=force_clear)
        assert (target / "segments.manifest.json").read_bytes() == before


def test_legacy_and_direct_backtest_metrics_identical(work):
    import rust_pyfunc as rp
    import h5py

    dates = [int(d.strftime("%Y%m%d")) for d in pd.date_range("2022-01-01", periods=60)]
    codes = [f"{i:06}" for i in range(1, 65)]
    rng = np.random.default_rng(37)
    names = [f"f{i}" for i in range(16)]
    values = rng.normal(size=(len(dates) * len(codes), 16)).astype(np.float32)
    values[::101, ::3] = np.nan
    row_dates, row_codes = np.repeat(dates, len(codes)).tolist(), codes * len(dates)
    style = work / "style"
    szba = style / "SzBa"
    szba.mkdir(parents=True)
    pd.DataFrame({"date_min": dates}).to_csv(szba / "calendar_map.csv", index=False)
    pd.DataFrame({"symbol": codes, "pos": list(range(len(codes)))}).to_csv(szba / "symbol_map.csv", index=False)
    style_rng = np.random.default_rng(11)
    fields = ["residual_volatility", "book_to_price", "size", "momentum", "leverage",
              "earnings_yield", "growth", "liquidity", "beta", "non_linear_size"]
    for i, field in enumerate(fields + ["industry"]):
        with h5py.File(szba / f"{field}.h5", "w") as h5:
            vals = np.tile((np.arange(64) % 31 + 1).astype(np.float64), (60, 1)) if field == "industry" else style_rng.standard_normal((60, 64)) + i * .1
            h5.create_dataset("data", data=vals)
    inputs = work / "inputs"
    inputs.mkdir()
    ret_keys = ["ret_gap1", "ret_sum_gap1", "ret_gap5", "ret_sum_gap5", "restrict", "index_ret"]
    for key in ret_keys[:4]:
        np.save(inputs / f"{key}.npy", rng.normal(0, .02, (len(dates), len(codes))).astype(np.float32))
    np.save(inputs / "restrict.npy", np.zeros((len(dates), len(codes)), np.float32))
    np.save(inputs / "index_ret.npy", rng.normal(0, .01, len(dates)).astype(np.float32))
    results = []
    for kind in ["legacy", "direct"]:
        store = work / kind
        if kind == "legacy":
            sb.write_legacy(str(store), row_dates, row_codes, values.tolist(), names)
        else:
            writer = sb.Writer(str(store), names, 32 * 1024)
            writer.append_rows(row_dates, row_codes, values.tolist())
            writer.finish()
            writer.close()
        opts = dict(colblk_store_dir=str(store), factor_names=names,
                    factor_paths=[f"{store}::{i}" for i in range(16)], dates=dates,
                    stocks=[c + ".SZ" for c in codes], windows=[5, 10], fold=False, n_jobs=4,
                    min_valid=5, style_vars_dir=str(style), backtest_start=dates[0], cover_rate=.5,
                    majority_count_threshold=200., zero_max_threshold=1., nan_max_threshold=1.,
                    save_all_metrics=True, industry_neutralize=True,
                    industry_matrix=np.tile((np.arange(64) % 31 + 1).astype(np.float64), (60, 1)),
                    ic_only=True, ret_point_neu_gap5=0., ret_point_neu_gap1=0.,
                    ic_point_neu_gap5=0., ic_point_neu_gap1=0., ret_point_gap5=0., ret_point_gap1=0.,
                    ic_point_gap5=0., ic_point_gap1=0., ic_more_important_gap5=None,
                    ic_more_important_gap1=None, io_batch_size=4,
                    cache_root=str(work / f"engine_{kind}"),
                    **{f"{key}_path": str(inputs / f"{key}.npy") for key in ret_keys})
        results.append(rp.tail_backtest_engine(**opts))
        assert results[-1]["processed_sources"] == 16 and results[-1]["io_read_groups"] == 4
    assert results[0]["candidate_counts"] == results[1]["candidate_counts"]
    assert results[0]["candidate_counts"]["neu_gap5"] > 0
    left, right = work / "engine_legacy", work / "engine_direct"
    checked = 0
    for file in left.rglob("*.parquet"):
        other = right / file.relative_to(left)
        assert other.exists()
        a, b = pd.read_parquet(file), pd.read_parquet(other)
        if "factor_name" in a.columns:
            a, b = a.set_index("factor_name").sort_index(), b.set_index("factor_name").sort_index()
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        checked += 1
    for file in left.rglob("*.npy"):
        np.testing.assert_array_equal(np.load(file, allow_pickle=True),
                                      np.load(right / file.relative_to(left), allow_pickle=True))
        checked += 1
    assert checked > 0
