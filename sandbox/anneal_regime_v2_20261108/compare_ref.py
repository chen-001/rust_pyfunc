"""与冻结参考输出逐位比对 + 测速（AV2_TIMING 打开时打印分段耗时）。"""
import sys, time, numpy as np
sys.path.insert(0, "package")
import anneal_regime_v2_sbx as sb
for c in ["000001", "600519", "300750"]:
    ref = np.load(f"results/ref_{c}.npy")
    best = 1e9
    for _ in range(3):
        t = time.time(); v = np.array(sb.py_anneal_regime_excess(c, 20220819), dtype=np.float32); best = min(best, time.time() - t)
    same = np.array_equal(v.view(np.uint32), ref.view(np.uint32))
    print(c, f"最快{best:.3f}s", "与参考逐位相同" if same else f"不同! 差异列数 {np.sum(~((v == ref) | (np.isnan(v) & np.isnan(ref))))}")
