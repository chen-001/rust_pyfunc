"""生产引擎耗时占比实测: ic_only 模式跑 40 因子, 收集 PROF 输出。"""
import numpy as np
import rust_pyfunc as rp
import sys

sys.path.insert(0, "/home/chenzongwei/design_whatever")
import design_whatever as dw

STORE = "/hdd/user_home_unsafe/chenzongwei/factor_store_挂单猫0701c"
BARRA = "/home/chenzongwei/database/barra/barra_daily_together_jason.parquet"
BP = "/home/chenzongwei/pythoncode/_tail_v2_shared/backtest_inputs/000905_20170103_20260522_5438_6b8884a67f05d221/"


def main():
    info = rp.factor_store_v5_info(STORE)
    names = list(info["factor_names"])[:40]
    tmpl = rp.factor_store_v5_template(STORE)
    dates = np.asarray(tmpl["dates"], dtype=np.int64)
    stocks = list(tmpl["stocks"])
    ind = np.ascontiguousarray(dw.tail_v4._load_industry_matrix(dates, stocks))

    rp.tail_backtest_engine(
        colblk_store_dir=STORE,
        factor_names=names,
        factor_paths=[f"{STORE}::{i}" for i in range(40)],
        dates=dates.astype(np.int32).tolist(),
        stocks=stocks,
        windows=[5, 10, 20],
        fold=False,
        n_jobs=8,
        min_valid=12,
        cache_root="/tmp/rankic_prof_cache",
        style_data_path=BARRA,
        ret_gap1_path=BP + "ret_gap1.npy",
        ret_sum_gap1_path=BP + "ret_sum_gap1.npy",
        ret_gap5_path=BP + "ret_gap5.npy",
        ret_sum_gap5_path=BP + "ret_sum_gap5.npy",
        restrict_path=BP + "restrict.npy",
        index_ret_path=BP + "index_ret.npy",
        backtest_start=20170201,
        cover_rate=0.0,
        ret_point_neu_gap5=0.0, ret_point_neu_gap1=0.0,
        ic_point_neu_gap5=0.0, ic_point_neu_gap1=0.0,
        ret_point_gap5=0.0, ret_point_gap1=0.0,
        ic_point_gap5=0.0, ic_point_gap1=0.0,
        ic_more_important_gap5=None, ic_more_important_gap1=None,
        majority_count_threshold=1e9, zero_max_threshold=1.0, nan_max_threshold=1.0,
        save_all_metrics=False,
        industry_neutralize=True,
        industry_matrix=ind,
        ic_only=True,
    )
    print("done")


if __name__ == "__main__":
    main()
