"""回测引擎“低覆盖放行”的验证。

用法：
  python verify_engine.py run <标签> <head|new> <store> <因子名逗号分隔> [标记词逗号分隔]
      head = 用未修改的 design_whatever（dw_head/ 下的 git HEAD 导出）；new = 用工作区版本。
      标记词为空 = 不开启放行（默认路径）。
      产物写到 results/eng_<标签>/，selected parquet 写到 results/eng_<标签>_sel/。
  python verify_engine.py compare <标签A> <标签B>
      对比两次运行的产物文件（ic/、metrics/、selected/ 下全部文件的字节哈希）。
"""
import hashlib
import json
import os
import sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[k] = "1"

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
# 只比较这些子目录：配置文件里有构建指纹、日志里有时间，不算产物
CMP_DIRS = ("ic_ts", "metrics", "selected", "task_results")


def run(tag, which, store, names, markers):
    if which == "head":
        sys.path.insert(0, str(HERE / "dw_head"))
    else:
        sys.path.insert(0, "/home/chenzongwei")
    import design_whatever.tail_v4 as tv4
    print("design_whatever 来源:", tv4.__file__, flush=True)
    kw = dict(
        colblk_store_dir=store,
        ver=f"lowcov_check_{tag}",
        names=names,
        temp_root=str(RES / f"eng_{tag}"),
        selected_parquet_dir=str(RES / f"eng_{tag}_sel"),
        n_jobs=min(len(names), 8),
        force_restart=True,
    )
    if markers:
        kw["low_coverage_markers"] = markers
    out = tv4.run_tail_pipeline_engine(**kw)
    print("入选 gap5/gap1:", len(out[0]), len(out[1]), out[0][:5], out[1][:5])


def digest(tag):
    root = RES / f"eng_{tag}"
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and any(part in CMP_DIRS for part in p.relative_to(root).parts[:-1]):
            files[str(p.relative_to(root))] = hashlib.sha1(p.read_bytes()).hexdigest()
    return files


def compare(a, b):
    A, B = digest(a), digest(b)
    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if A.get(k) != B.get(k)]
    print(f"比较文件 {len(keys)} 个（{', '.join(CMP_DIRS)} 下），不同 {len(diff)} 个")
    for k in diff:
        print("  不同:", k, A.get(k, "缺"), B.get(k, "缺"))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        tag, which, store = sys.argv[2], sys.argv[3], sys.argv[4]
        names = [n for n in sys.argv[5].split(",") if n]
        markers = [m for m in (sys.argv[6].split(",") if len(sys.argv) > 6 else []) if m]
        run(tag, which, store, names, markers)
    else:
        compare(sys.argv[2], sys.argv[3])
