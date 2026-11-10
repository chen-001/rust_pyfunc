"""第二轮 Q4：用已落盘的每日 IC 矩阵，测“互补对”和“以已选因子为条件的增量”哪个样本外更好。
只读回测产物，不写回测目录。线程上限 40。
用法: python q4_complement_test.py <版本目录名> [K]
"""
import os
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "40"
import json
import sys
import time
import numpy as np

BASE = "/nas197/user_home_unsafe/chenzongwei"
SPLIT = int(os.environ.get("Q4_SPLIT", "20221231"))
GAP = os.environ.get("Q4_GAP", "5")
SEED = 20261109
ver = sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
ic_dir = f"{BASE}/{ver}/ic_ts"
t0 = time.time()
M = np.load(f"{ic_dir}/ic_neu_gap{GAP}.npy", mmap_mode="r")
names = json.load(open(f"{ic_dir}/ic_neu_gap{GAP}_names.json"))
dates = np.load(f"{ic_dir}/ic_neu_gap{GAP}_dates.npy").astype(int)
print(ver, "IC矩阵", M.shape, "因子", len(names), "日期", len(dates), dates[0], dates[-1])
assert M.shape == (len(dates), len(names))
M = np.array(M, dtype=np.float32)
print(f"读入 {time.time() - t0:.1f}s, {M.nbytes / 1e9:.2f}GB")

tr = dates <= SPLIT
te = ~tr
valid = np.isfinite(M).mean(0)
keep0 = np.where(valid > 0.7)[0]
X = np.nan_to_num(M[:, keep0], nan=0.0)
Xtr, Xte = X[tr], X[te]
print("有效因子", len(keep0), "训练期", tr.sum(), "样本外", te.sum())
sign = np.sign(Xtr.mean(0))
sign[sign == 0] = 1
Xtr = Xtr * sign
Xte = Xte * sign
mtr, str_ = Xtr.mean(0), Xtr.std(0)
mte, ste = Xte.mean(0), Xte.std(0)
ir = lambda m, s: m / np.maximum(s, 1e-9)
irtr, irte = ir(mtr, str_), ir(mte, ste)
print("样本内→样本外 单因子 IR 秩相关", np.round(np.corrcoef(irtr.argsort().argsort(), irte.argsort().argsort())[0, 1], 3))
order = np.argsort(-np.abs(mtr))


def greedy(order, X, thr=0.6, n=45):
    """训练期内复现流程：|IC| 降序、IC 序列相关 <0.6 贪心取 n 个（不含 TLQ 门禁）"""
    Z = (X - X.mean(0)) / np.maximum(X.std(0), 1e-9)
    sel = []
    for i in order:
        if not sel:
            sel.append(i)
            continue
        c = (Z[:, sel].T @ Z[:, i]) / len(Z)
        if np.abs(c).max() < thr:
            sel.append(i)
            if len(sel) >= n:
                break
    return np.array(sel)


t = time.time()
S = greedy(order, Xtr)
print(f"单因子贪心选 {len(S)} 个 用时 {time.time() - t:.1f}s; S 的训练期 |IC| 范围 {np.abs(mtr[S]).min():.4f}~{np.abs(mtr[S]).max():.4f}")

# 模拟“单独不够突出”的候选池：不在 S 中，随机 K 个
rng = np.random.default_rng(SEED)
pool = np.setdiff1d(np.arange(Xtr.shape[1]), S)
pool = rng.choice(pool, size=min(K, len(pool)), replace=False)


def pair_scan(A, mA):
    """全部配对的 IR 增益：组合 IR - max(单因子 IR)。只需协方差矩阵。"""
    n = len(A)
    B = A - A.mean(0)
    cov = (B.T @ B) / n
    var = np.diag(cov).copy()
    ir1 = mA / np.sqrt(np.maximum(var, 1e-18))
    denom = np.sqrt(np.maximum(var[:, None] + var[None, :] + 2 * cov, 1e-18))
    pir = (mA[:, None] + mA[None, :]) / denom
    best = np.maximum(ir1[:, None], ir1[None, :])
    return pir, pir - best, ir1


def pair_ir(Xs, i, j):
    y = Xs[:, i] + Xs[:, j]
    return y.mean() / max(y.std(), 1e-9)


t = time.time()
Ap = Xtr[:, pool]
pir, gain, ir1 = pair_scan(Ap, Ap.mean(0))
iu = np.triu_indices(len(pool), 1)
gv = gain[iu]
print(f"\n配对扫描 {len(pool)} 因子 = {len(gv):,} 对, 用时 {time.time() - t:.1f}s")
top_idx = np.argsort(-gv)[:200]
top_pairs = [(pool[iu[0][k]], pool[iu[1][k]]) for k in top_idx]
te_gain = []
te_ir = []
for i, j in top_pairs:
    p = pair_ir(Xte, i, j)
    te_ir.append(p)
    te_gain.append(p - max(irte[i], irte[j]))
te_gain = np.array(te_gain)
print("训练期 top200 对: IR 增益中位", np.round(np.median(gv[top_idx]), 3), " 样本外增益中位", np.round(np.median(te_gain), 3),
      " 样本外增益>0 占比", np.round((te_gain > 0).mean(), 3), " 样本外组合 IR 中位", np.round(np.median(te_ir), 3))

# 随机配对对照
rk = rng.choice(len(gv), 20000, replace=False)
rg = np.array([pair_ir(Xte, pool[iu[0][k]], pool[iu[1][k]]) - max(irte[pool[iu[0][k]]], irte[pool[iu[1][k]]]) for k in rk])
print("随机对: 样本外增益中位", np.round(np.median(rg), 3), " >0 占比", np.round((rg > 0).mean(), 3))

# 多重检验：把每个因子的 IC 序列各自循环错位，破坏配对关系、保留自相关，看“最佳增益”靠运气能到多少
t = time.time()
null_max = []
null_q = []
sub = rng.choice(len(pool), min(2500, len(pool)), replace=False)
obs_sub = gain[np.ix_(sub, sub)]
obs_sub = obs_sub[np.triu_indices(len(sub), 1)]
for r in range(5):
    shift = rng.integers(0, Ap.shape[0], len(sub))
    A = np.stack([np.roll(Ap[:, sub[k]], shift[k]) for k in range(len(sub))], 1)
    _, g0, _ = pair_scan(A, A.mean(0))
    g0 = g0[np.triu_indices(len(sub), 1)]
    null_max.append(g0.max())
    null_q.append(np.quantile(g0, 0.9999))
print(f"错位检验(5次, {len(sub)} 因子, {len(obs_sub):,} 对, {time.time() - t:.0f}s): 真实最大增益 {obs_sub.max():.3f}, 错位最大增益 {np.round(null_max, 3)}; "
      f"真实 99.99% 分位 {np.quantile(obs_sub, 0.9999):.3f}, 错位 {np.round(null_q, 3)}")

# 以已选 45 为条件的增量：组合是已选集合的等权 IC 序列
def comp_ir(Xs, idx):
    y = Xs[:, idx].mean(1)
    return y.mean() / max(y.std(), 1e-9), y.mean()


base_ir_tr, base_ic_tr = comp_ir(Xtr, S)
base_ir_te, base_ic_te = comp_ir(Xte, S)
print(f"\n已选 45 等权组合 IR: 训练 {base_ir_tr:.3f} 样本外 {base_ir_te:.3f}; 平均 IC 训练 {base_ic_tr:.4f} 样本外 {base_ic_te:.4f}")


def forward_add(cands, m=20):
    """以当前集合为条件，每步在候选里加一个使训练期组合 IR 最大的因子"""
    cur = list(S)
    Xc = Xtr[:, cands]
    picked = []
    for _ in range(m):
        s = Xtr[:, cur].sum(1)
        n = len(cur)
        # 候选 c 加入后组合 = (s + Xc)/(n+1)，IR 与缩放无关
        y = s[:, None] + Xc
        irs = y.mean(0) / np.maximum(y.std(0), 1e-9)
        if picked:
            irs[[list(cands).index(p) for p in picked]] = -1e9
        k = int(np.argmax(irs))
        picked.append(cands[k])
        cur.append(cands[k])
    return picked


t = time.time()
inc = forward_add(pool)
print(f"条件增量前向选择 20 个 用时 {time.time() - t:.1f}s")
irs_inc_tr, _ = comp_ir(Xtr, list(S) + list(inc))
irs_inc_te, ic_inc_te = comp_ir(Xte, list(S) + list(inc))
pair_members = []
for i, j in top_pairs:
    for x in (i, j):
        if x not in pair_members and x not in S:
            pair_members.append(x)
        if len(pair_members) >= 20:
            break
    if len(pair_members) >= 20:
        break
ir_pair_tr, _ = comp_ir(Xtr, list(S) + pair_members)
ir_pair_te, ic_pair_te = comp_ir(Xte, list(S) + pair_members)
rand = []
for r in range(200):
    pick = list(rng.choice(pool, 20, replace=False))
    rand.append(comp_ir(Xte, list(S) + pick)[0])
rand = np.array(rand)
# 名次顺延对照：按单因子 |IC| 接着往下取 20 个（相当于把 45 放宽到 65）
next20 = [i for i in order if i not in set(S.tolist())][:20]
ir_next_te, ic_next_te = comp_ir(Xte, list(S) + next20)
print("加 20 个后的组合 IR（训练 → 样本外；基准样本外 %.3f）" % base_ir_te)
print(f"  条件增量: {irs_inc_tr:.3f} → {irs_inc_te:.3f}（平均IC {ic_inc_te:.4f}）")
print(f"  互补对成员: {ir_pair_tr:.3f} → {ir_pair_te:.3f}（平均IC {ic_pair_te:.4f}）")
print(f"  单因子名次顺延: → {ir_next_te:.3f}（平均IC {ic_next_te:.4f}）")
print(f"  随机 20 个: 中位 {np.median(rand):.3f}, 95% 分位 {np.quantile(rand, 0.95):.3f}, 增量胜过随机的分位 {(rand < irs_inc_te).mean():.3f}, 互补对胜过随机的分位 {(rand < ir_pair_te).mean():.3f}")
print("  单因子 |IC|（训练）: 入选的条件增量因子 中位 %.4f；互补对成员 中位 %.4f；45 入选中位 %.4f" % (
    np.median(np.abs(mtr[inc])), np.median(np.abs(mtr[pair_members])), np.median(np.abs(mtr[S]))))
print(f"总用时 {time.time() - t0:.0f}s")
