"""用一个已有的横截面 pipeline 入口（cross_section_example，走 worker 进程）验证读取修复。

用法：python verify_pipeline.py <标签>
  对 20160630、20241231 两天跑一次，产物写到 results/pipe_<标签>/，
  再把每个因子读回来，按 (日期, 代码) 排序后记录行数与哈希到 results/pipe_<标签>.json。
对比：python verify_pipeline.py compare <标签A> <标签B>
"""
import hashlib
import json
import os
import shutil
import sys

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import rust_pyfunc as rp

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
DATES = [20160630, 20241231]


def run(tag):
    names = list(rp.py_cross_section_example_names())
    store = os.path.join(RES, f"pipe_{tag}")
    if os.path.exists(store):
        shutil.rmtree(store)
    os.makedirs(RES, exist_ok=True)
    rp.run_factor_pipeline_cross_section(
        "cross_section_example",
        DATES,
        n_jobs=1,
        expected_result_length=len(names),
        trading_days=DATES,
        n_workers=1,
        bind_cores=False,
        store_dir=store,
        store_factor_names=names,
        force_clear=True,
    )
    with open(os.path.join(RES, f"pipe_{tag}.json"), "w") as f:
        json.dump(digest_store(store), f, indent=1)
    out = json.load(open(os.path.join(RES, f"pipe_{tag}.json")))
    for name, per in out["cols"].items():
        print(tag, name, {k: (x["rows"], x["finite"]) for k, x in per.items()})


def digest_store(store):
    """按 (日期, 代码字符串) 排序后哈希。代码编号是 store 内部按出现顺序分配的，
    两次运行读到的股票集合不同时编号会变，所以必须换成代码字符串再比。"""
    info = rp.factor_store_v5_info(store)
    tmpl = rp.factor_store_v5_template(store)
    dates_axis = np.asarray(tmpl["dates"], dtype=np.int64)
    codes_axis = np.asarray(list(tmpl["stocks"]), dtype="U6")
    out = {"names": list(info["factor_names"]), "cols": {}}
    for i, name in enumerate(info["factor_names"]):
        r = rp.factor_store_v5_read_factor(store, i)
        d = dates_axis[np.asarray(r["date_id"])]
        c = codes_axis[np.asarray(r["code_id"])]
        v = np.asarray(r["factor"], dtype=np.float32)
        order = np.lexsort((c, d))
        d, c, v = d[order], c[order], v[order]
        per_date = {}
        for day in np.unique(d):
            m = d == day
            per_date[str(int(day))] = {
                "rows": int(m.sum()),
                "finite": int(np.isfinite(v[m]).sum()),
                "sha1": hashlib.sha1(c[m].tobytes() + v[m].tobytes()).hexdigest(),
            }
        out["cols"][name] = per_date
    return out


def compare(a, b):
    A = digest_store(os.path.join(RES, f"pipe_{a}"))
    B = digest_store(os.path.join(RES, f"pipe_{b}"))
    assert A["names"] == B["names"], "列名不同"
    for name in A["names"]:
        for day in sorted(set(A["cols"][name]) | set(B["cols"][name])):
            x, y = A["cols"][name].get(day), B["cols"][name].get(day)
            same = x == y
            print(f"{name} {day}: {a} {x and (x['rows'], x['finite'])} / "
                  f"{b} {y and (y['rows'], y['finite'])} {'逐字节相同' if same else '不同'}")


if __name__ == "__main__":
    if sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
    else:
        run(sys.argv[1])
