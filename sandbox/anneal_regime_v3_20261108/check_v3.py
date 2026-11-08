"""v3 沙箱验收：①16 个原始指标与 v1 原版逐位对拍；②长度/名字唯一/确定性/速度；③真实数据上订单数干扰检查。"""
import sys, time, numpy as np
sys.path.insert(0, "package")
import anneal_regime_v3_sbx as sb
rng = np.random.default_rng(1)
bad = 0; tot = 0
for n in [3, 4, 7, 15, 31, 32, 60, 200, 1000, 5000]:
    for cap in [6000, 12000, 20000, 30000]:
        for kind in range(4):
            if kind == 0: x = rng.lognormal(5, 1, n)
            elif kind == 1: x = np.round(rng.lognormal(5, 1, n), -2)        # 整百股，大量相同值
            elif kind == 2: x = np.sort(rng.lognormal(5, 1, n))              # 单调
            else: x = np.full(n, 100.0); x[: max(1, n // 3)] = 300.0         # 极少几种取值
            o, v = sb.py_compare_old_v3(x.astype(np.float32).tolist(), cap)
            o = np.array(o, np.float32); v = np.array(v[:16], np.float32)
            tot += 1
            if not np.array_equal(o.view(np.uint32), v.view(np.uint32)):
                bad += 1; print("不一致", n, cap, kind, o, v)
print(f"原始 16 指标与 v1 原版逐位对拍：{tot - bad}/{tot} 组一致")
names = sb.py_anneal_regime_raw_excess_names()
print("因子数", len(names), "名字唯一", len(set(names)) == len(names))
ix = {k: i for i, k in enumerate(names)}
for c in ["000001", "600519", "300750", "002594", "601318", "688981"]:
    best = 1e9
    for _ in range(2):
        t = time.time(); v = np.array(sb.py_anneal_regime_raw_excess(c, 20220819), np.float32); best = min(best, time.time() - t)
    v2 = np.array(sb.py_anneal_regime_raw_excess(c, 20220819), np.float32)
    same = np.array_equal(v.view(np.uint32), v2.view(np.uint32))
    ks = ["s3_cond_m1_cnt_freq_bid_final_r_state_corr", "s3_cond_m1_cnt_freq_bid_final_r_ex_state_corr", "s3_cond_m1_cnt_freq_bid_tie_ratio_state_corr"]
    print(c, f"最快{best:.2f}s", "两次一致" if same else "两次不一致!", "NaN比例", round(float(np.isnan(v).mean()), 3),
          "笔数-还原度相关[原始, 超额, 相同量比例]", [round(float(v[ix[k]]), 2) for k in ks])
