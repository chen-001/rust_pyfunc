"""新版沙箱基本验收：长度、名字唯一、速度、确定性、各模块空值比例、订单数干扰。"""
import sys, time, numpy as np
sys.path.insert(0, "package")
import anneal_regime_v2_sbx as sb
names = sb.py_anneal_regime_excess_names()
print("因子数", len(names), "名字唯一", len(set(names)) == len(names))
# 1) 单条随机序列：打乱对照应让超额值接近 0、且不随长度变化
rng = np.random.default_rng(0)
for n in [10, 50, 200, 1000, 5000]:
    ex = []
    raw = []
    for rep in range(30):
        x = rng.lognormal(5, 1, n).astype(np.float32)
        r, nl, o = sb.py_anneal_seq_real_null_debug(x.tolist(), max(200, 20 * n))
        raw.append(r[1]); ex.append(o[1])
    print(f"随机序列 n={n:5d}  真实final_r均值={np.mean(raw):.3f}  超额final_r均值={np.mean(ex):+.4f}  超额标准差={np.std(ex):.4f}")
# 2) 真实股票
codes = ["000001", "600519", "300750", "002594", "601318", "000858", "600000", "688981"]
ix = {k: i for i, k in enumerate(names)}
for c in codes:
    t = time.time(); v = np.array(sb.py_anneal_regime_excess(c, 20220819), dtype=np.float64); dt = time.time() - t
    v2 = np.array(sb.py_anneal_regime_excess(c, 20220819), dtype=np.float64)
    same = np.array_equal(np.nan_to_num(v, nan=1e30), np.nan_to_num(v2, nan=1e30))
    keys = ["s2_cond_m1_cnt_freq_bid_final_r_ex_state_corr", "s2_cond_m1_cnt_freq_bid_steps_r80_norm_ex_state_corr",
            "s2_cond_m1_cnt_freq_bid_inertia_norm_ex_state_corr", "s2_cond_s30_cnt_freq_bid_final_r_ex_state_corr"]
    print(c, f"耗时{dt:.2f}s", "两次一致" if same else "两次不一致!", "NaN比例", round(np.isnan(v).mean(), 3),
          "成交笔数-还原度相关", [round(v[ix[k]], 2) for k in keys])
