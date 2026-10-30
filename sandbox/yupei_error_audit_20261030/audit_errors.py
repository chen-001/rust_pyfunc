"""Read-only source audit and tiny numerical counterexamples; no market scan.

Python ports plus a one-core executable compiled in a temporary directory from
the actual private functions. Source assertions tie examples to this version.
No main-project build, install or production behavior change.
"""
import hashlib
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FILES = ["src/yupei_multi/mod.rs", "src/yupei_pair/split_mat.rs",
         "src/yupei_pair/split.rs", "src/yupei_pair/reduce.rs",
         "src/yupei_dist/matrix_stage.rs", "src/yupei_pair/net.rs"]
src = {name: (ROOT / name).read_text() for name in FILES}
multi, mat, split, reduce, stage, net = [src[name] for name in FILES]
findings = []


def add(name, evidence, example, affected, remedy, speed):
    findings.append(dict(name=name, evidence=evidence, example=example,
                         affected=affected, remedy=remedy, speed=speed))


assert "bucketize(side_pts),\n        otsu_out," in multi
assert "cells_otsu.push(c);\n        cells_side.push(d);" in multi
add("oc_/os_ 载体标签颠倒", "源码已证实；不需要全天重算",
    {"生成顺序": ["base", "order", "order_side", "otsu"],
     "消费顺序": ["base", "order", "otsu", "order_side"]},
    "全部 oc_ 与 os_ 列的定义。目前 oc_ 实际为订单方向，os_ 实际为聚簇。",
    "同步交换载体接线和矩阵复用位置；若只修标签，历史输出可按整组前缀重命名。",
    "纯纠正标签/接线，不会明显提速，仍保留订单方向矩阵复用。")

assert "N_BUCKETS: usize = 1_422_000" in stage
assert "N_BLOCKS: usize = N_BUCKETS / BLOCK_BUCKETS" in stage
assert "for block in 0..N_BLOCKS" in mat
n_buckets, block = 1_422_000, 4096
covered = (n_buckets // block) * block
add("末尾 6.88 秒未进入交互矩阵", "源码常数与整数运算已证实；实际当日影响数未重算",
    {"全天桶数": n_buckets, "实际块数": n_buckets // block,
     "覆盖桶数": covered, "漏桶数": n_buckets - covered,
     "漏秒数": (n_buckets - covered) / 100,
     "单笔落在最后桶时": {"当前自能量": 0, "应有自能量": 1}},
    "四组链和领先度；SVD 独立覆盖全部 14220 秒，不受此处截断。",
    "块数向上取整并正确处理最后部分块；共享常数同时影响 yupei_dist 等入口。",
    "属于补齐计算，预计微增耗时；不能作为提速措施。")

assert "p[PM] += up * gpk[1]" in mat
assert "p[MP] += um * gpk[0]" in mat
assert "*raw[t].0.add(j * n + i) = pack[i * 4 + t]" in mat
assert "if ap && bm {\n        v += m[PM][k]" in split
# Exact f32 recurrence of the scalar hot loop for four synthetic events.
events = [{0: (1, 0), 100: (0, 1)}, {1: (0, 1), 300: (1, 0)}]
decay = np.exp(np.float32(-0.01), dtype=np.float32)
m4 = np.zeros((4, 2, 2), np.float32)
for j in range(2):
    ap = am = np.float32(0)
    for bucket in range(301):
        jp, jm = map(np.float32, events[j].get(bucket, (0, 0)))
        gp, gm = np.float32(ap + np.float32(.5) * jp), np.float32(am + np.float32(.5) * jm)
        for i in range(2):
            ip, im = map(np.float32, events[i].get(bucket, (0, 0)))
            m4[:, j, i] += np.array([ip * gp, ip * gm, im * gp, im * gm], np.float32)
        ap, am = np.float32((ap + jp) * decay), np.float32((am + jm) * decay)
wrong_rho = float(m4[1, 0, 1]) + float(m4[2, 1, 0])
right_rho = float(m4[2, 0, 1]) + float(m4[1, 1, 0])
assert wrong_rho < .14 and right_rho > .99
add("PM/MP 槽方向与消费定义相反", "源码、Python 小样本和原矩阵核心单核复现均通过；未重算完整因子",
    {"A": "P@0ms, M@1000ms", "B": "M@10ms, P@3000ms",
     "R(P,M)[A,B]当前表达式": wrong_rho,
     "R(P,M)[A,B]按定义应有": right_rho,
     "说明": "两股两方向自能量都为 1，示例没有缺失方向或掩码干扰。"},
    "每组 spppa、spAAp、sppma、spmpm；spppp 只用 PP 不受此修复影响。领先度用四槽总和、SVD 用基础笔数，也不受交换 PM/MP 影响。",
    "统一矩阵写入/读取的先动方向和后动方向；只交换 PM/MP 槽可保持计算量。",
    "没有直接提速，但优先级高于给错误方向加速；历史对应因子需重算/重评。")

assert "for (tw, sp, sm) in agg.into_values()" in multi
assert "ev.sort_by_key(|x| x.0)" in multi
values = np.sqrt(np.array([1, 2, 2], np.float32))
sums = {}
for order in itertools.permutations(range(3)):
    value = np.float32(0)
    for i in order:
        value = np.float32(value + values[i])
    sums[','.join(map(str, order))] = float(value)
assert len(set(sums.values())) == 2
add("vol 同桶累加顺序不确定", "源码 HashMap 遍历 + 稳定桶排序已证实，三笔整数成交量反例已复现",
    {"三个独立订单同桶成交量": [1, 2, 2], "不同顺序的平方根量之和": sums},
    "weight=vol 的 order 与当前 oc_ 组；默认 cnt 的整数累加不受此最小反例影响。",
    "按稳定订单键排序后落桶，或同桶内再按订单键排序；不能只按桶号排序。",
    "确定性修复略增排序成本；可结合一次排序派生双通道减少冗余分配，未测收益。")

assert "static SOFT: OnceLock<SoftCfg>" in reduce
assert "let _ = SOFT.set(cfg)" in reduce
add("sig_temp/sig_thr 被进程首次调用锁定", "源码明确写明重复设置被忽略",
    {"同进程调用顺序": ["第一次 sig_thr=2", "第二次 sig_thr=5"],
     "第二次实际阈值": 2,
     "跨函数影响": "yupei_pair/split/multi 共享同一 SOFT"},
    "阈值计数、软质量、显著性相关列；默认参数且全新进程不受影响。",
    "把参数显式传递到本次任务，不用进程全局可变状态；直接改全局锁会引入并发调用串参。",
    "参数正确性修复，暂无可见提速收益。")

assert "if total == 0 {\n            continue;\n        }" in mat
old = np.float32(1)
correct = np.float32(1)
for _ in range(block):
    correct = np.float32(correct * decay)
add("SSE 全市场空块跳过历史衰减", "源码路径已证实；单日 5195 股是否存在空块未验证",
    {"块长秒": block / 100, "块前残留": 1,
     "空块后当前残留": float(old), "按相同逐桶递推应有残留": float(correct)},
    "空块后重新出现成交时的矩阵；主要是稀疏/小股票池边界，全市场默认路径触发未知。",
    "没有新成交也要推进历史衰减；要求逐位一致时保留递推顺序。",
    "修复漏算，不应宣称提速；若设计可接受舍入差异的跳桶算法须单独验证。")

# A separate role validity issue: one direction's mask is used for B, C and X.
assert "let mask: Vec<bool> = e_a.iter().map(|&v| v > 0.0).collect()" in multi
assert "if !mask_ref[c]" in net
findings[2]["additional_direction_boundary"] = {
    "问题": "混合方向共用 B 的有效股票掩码",
    "证据": "源码已证实，未扫描实际全日触发股票数，也未运行独立核心测试。",
    "小例": "sppma 的 C 只看 M，只有主卖的 C 本应有能量，却因共用 P 掩码被排除。",
    "建议": "B、C、X 应分别定义有效掩码；修正后候选可能增减，需单独重验。"}

native = r'''
use rayon;
mod yupei_dist { pub mod matrix_stage {
    pub const N_BUCKETS: usize = 1_422_000;
    pub const BLOCK_BUCKETS: usize = 4096;
    pub const N_BLOCKS: usize = N_BUCKETS / BLOCK_BUCKETS;
    pub const BUCKET_US: i64 = 10000;
    pub struct StockPrep { pub cells_pm: Vec<(u32, [f32; 4])> }
} }
mod fast_csv_reader {
    pub struct TradeRecord { pub time_us: i64, pub volume: f32,
        pub flag: u8, pub ask_order: i64, pub bid_order: i64 }
}
#[path = "__SPLIT_PATH__"] mod split_mat;
mod audit_cells {
    use crate::yupei_dist::matrix_stage::{N_BUCKETS, BUCKET_US};
    __CELL_SOURCE__
    pub fn run() {
        let r: Vec<_> = [1.0, 2.0, 2.0].iter().enumerate().map(|(i, &v)|
            crate::fast_csv_reader::TradeRecord { time_us: 100000, volume: v,
                flag: 66, ask_order: i as i64 + 1, bid_order: 0 }).collect();
        let mut sums = std::collections::BTreeSet::new();
        for _ in 0..1000 {
            let (base, order, side, otsu) = build_all_cells(&r, 0);
            assert_eq!(order[0].1[0], 3.0);
            assert_eq!(side[0].1[1], 3.0);
            assert_eq!(base, otsu);
            sums.insert(order[0].1[2].to_bits());
        }
        assert_eq!(sums.len(), 2);
        println!("vol_unique_f32_bits={:?}", sums.into_iter().collect::<Vec<_>>());
    }
}
mod audit_soft {
    use std::sync::OnceLock;
    __SOFT_SOURCE__
    pub fn run() {
        set_soft(SoftCfg { temp: 0.3, thr: 2.0 });
        set_soft(SoftCfg { temp: 0.3, thr: 5.0 });
        assert_eq!(soft().thr, 2.0);
        println!("soft_second_actual_threshold={}", soft().thr);
    }
}
fn main() {
    use split_mat::{compute_split_mats_simd as fast, compute_split_mats_from_cells as reference, PmWeight};
    let p = [1., 0., 1., 0.];
    let m = [0., 1., 0., 1.];
    let v = fast(&[vec![(0, p), (100, m)], vec![(1, m), (300, p)]], 1., PmWeight::Cnt);
    let wrong = v[1][1] + v[2][2];
    let correct = v[2][1] + v[1][2];
    assert!(wrong < 0.14 && correct > 0.99);
    println!("pm_mp_current={wrong}");
    println!("pm_mp_correct={correct}");
    let tail = fast(&[vec![(1_421_999, p)]], 1., PmWeight::Cnt);
    assert_eq!(tail[0][0], 0.);
    println!("tail_self_energy_current={}", 2. * tail[0][0]);
    let sparse = [vec![(4095, p)], vec![(8192, p)]];
    let bad = fast(&sparse, 1., PmWeight::Cnt)[0][1];
    let good = reference(&sparse, 1., PmWeight::Cnt)[0][1];
    assert!(bad > 0.99 && good < 1e-17);
    println!("empty_block_current={bad}");
    println!("empty_block_scalar_reference={good}");
    audit_cells::run();
    audit_soft::run();
}
'''
cells_start = multi.index("#[derive(Clone, Copy, PartialEq, Eq)]\npub enum CellMode")
cells_end = multi.index("/// 用一套桶单元跑 15 条链")
soft_start = reduce.index("#[derive(Clone, Copy)]\npub(crate) struct SoftCfg")
soft_end = reduce.index("// ============================================================", soft_start)
native = native.replace("__SPLIT_PATH__", str(ROOT / "src/yupei_pair/split_mat.rs"))
native = native.replace("__CELL_SOURCE__", multi[cells_start:cells_end])
native = native.replace("__SOFT_SOURCE__", reduce[soft_start:soft_end])
deps = ROOT / "target/release-fast/deps"
rayon = max(deps.glob("librayon-*.rlib"), key=lambda p: p.stat().st_mtime)
with tempfile.TemporaryDirectory(prefix="yupei_error_audit_") as tmp:
    tmp = Path(tmp)
    (tmp / "audit.rs").write_text(native)
    subprocess.run(["/home/chenzongwei/.cargo/bin/rustc", "+1.97.0", "--edition", "2021", "-Awarnings",
                    "-C", "opt-level=2", "-L", "dependency=" + str(deps),
                    "--extern", "rayon=" + str(rayon), str(tmp / "audit.rs"),
                    "-o", str(tmp / "audit")], check=True, text=True)
    run = subprocess.run([str(tmp / "audit")], check=True, capture_output=True, text=True,
                         env={**os.environ, "RAYON_NUM_THREADS": "1"})
native_results = {line.split("=", 1)[0]: json.loads(line.split("=", 1)[1])
                  for line in run.stdout.splitlines() if "=" in line}
for i in [0, 1, 2, 3, 4, 5]:
    findings[i]["native_validation"] = "独立单核可执行程序直接编入/抽取当前对应源码，断言通过；没有安装或重建主库。"
result = {"validation_scope": "源码 + Python 小例 + 原函数独立单核复现；未跑全日、未改变生产结果。",
          "native_results": native_results,
          "native_environment": {"compiler": "rustc 1.97.0", "rayon_threads": 1,
                                 "rayon_dependency": str(rayon), "production_install_changed": False},
          "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in FILES},
          "findings": findings}
output = ROOT / "reports/yupei_multi_ks5_20_20261030/error_audit.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"output": str(output), "findings": len(findings), "pm_mp_example": [wrong_rho, right_rho]}, ensure_ascii=False))
