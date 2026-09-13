"""sandbox 验证：池版聚合口径 vs 初版（池=全市场逐位对照）+ 真实池输出检查 + 速度实测。
用法：/home/chenzongwei/.conda/envs/chenzongwei311/bin/python verify_ext.py
"""
import time
import numpy as np
import sandbox_yhyb_indtop as sb

DATE = 20241231
names = sb.py_yhyb_ext_names()
assert len(names) == 4176, f"因子名数 {len(names)} != 4176"
print(f"names: {len(names)}（G1a 1740 + G1b 1740 + G3 696）")

# ---- 1) 口径对照：mode=1（池=全市场）vs 初版无 L4（py_yhyb_params 默认参数）----
prm = [10.0, 1200.0, 2, 180.0, 2, 0.99, 100, 2, 0.001, 0.5, 0.3, 2.0, 1.0, 0.5, 0.05, 60.0, 0.05]
t0 = time.time()
c1, v1 = sb.py_yhyb(20220819, False)
print(f"初版(含L4) 20220819: {len(c1)} 股 {time.time()-t0:.1f}s")
t0 = time.time()
c2, v2 = sb.py_yhyb_params(20220819, prm)
print(f"初版(无L4) 20220819: {len(c2)} 股 {time.time()-t0:.1f}s")

t0 = time.time()
c3, v3 = sb.py_yhyb_ext(20220819, 1)  # 池=全市场
print(f"池版(池=全市场) 20220819: {len(c3)} 股 {time.time()-t0:.1f}s")

# 对齐 codes 逐位比较 1740 因子
pos2 = {c: i for i, c in enumerate(c2)}
a = np.array(v2).reshape(len(c2), -1)
b = np.array(v3).reshape(len(c3), -1)
if len(c2) != len(c3):
    common = [c for c in c2 if c in set(c3)]
    a = a[[pos2[c] for c in common]]
    b = b[[pos2[c] for c in common]]
diff = np.abs(a - b)
nan_mask = np.isnan(a) & np.isnan(b)
mismatch = (~nan_mask) & (diff > 1e-4) & (~(np.isnan(a) | np.isnan(b)))
print(f"逐位对照: 比较 {a.shape} 值, NaN一致 {nan_mask.sum()}, 不一致 {mismatch.sum()}")
if mismatch.sum() > 0:
    idx = np.argwhere(mismatch)
    print("不一致位置(前10):", idx[:10].tolist())
    for i, j in idx[:5]:
        print(f"  ({i},{j}): 初版={a[i,j]} 池版={b[i,j]}")
