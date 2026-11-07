# 逐位比较两次结果里同名列：python -I compare.py base trim [date]
import sys, os, numpy as np
D = os.path.dirname(os.path.abspath(__file__)) + "/results"
date = sys.argv[3] if len(sys.argv) > 3 else "20241231"
def load(t):
    c, n = open(f"{D}/{t}_{date}.meta").read().split("\n##\n")
    return np.load(f"{D}/{t}_{date}.npy"), c.split("\n"), n.split("\n")
av, ac, an = load(sys.argv[1]); bv, bc, bn = load(sys.argv[2])
assert ac == bc, "股票代码不同"
pos = {x: i for i, x in enumerate(an)}
idx = np.array([pos[x] for x in bn])
same = av[:, idx].view(np.uint32) == bv.view(np.uint32)
print(f"{sys.argv[1]} {av.shape} vs {sys.argv[2]} {bv.shape}: 同名列逐位一致={bool(same.all())} 不一致格数={int((~same).sum())}")
