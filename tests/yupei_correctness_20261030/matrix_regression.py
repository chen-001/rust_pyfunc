"""One-core tests of real matrix kernels against independent f32 recurrences.

Only a tiny temporary executable is compiled; the installed package is untouched.
Empty intervals use numpy.multiply.accumulate (one rounding per bucket), not pow.
Optional first argument selects the source root (default: correctness sandbox).
"""
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'sandbox/yupei_correctness_20261030'
BASE = SOURCE_ROOT / 'src'
N, BLOCK = 1_422_000, 4096
TAIL = N // BLOCK * BLOCK
p, m = [1., 0., 1., 0.], [0., 1., 0., 1.]
cases = {
    'empty_input': [],
    'empty_stocks': [[], [], []],
    'direction_audit_example': [[(0, p), (100, m)], [(1, m), (300, p)]],
    'empty_block_audit_example': [[(4095, p)], [(8192, p)]],
    'last_bucket_audit_example': [[(N - 1, p)]],
    'odd_tail_and_day_limit': [[(TAIL - 1, [2., 1., 1.25, 3.5]), (N, p)],
                               [(TAIL, m), (N - 1, p)],
                               [(TAIL + 17, [1., 3., 2.5, 1.125]), (N - 1, m)]],
}
rng = np.random.default_rng(20261030)
cases['mixed_gaps_odd_universe'] = [
    [(k, [float(rng.integers(0, 6)), float(rng.integers(0, 6)),
           float(np.float32(rng.uniform(.1, 5))), float(np.float32(rng.uniform(.1, 5)))])
     for k in [0, 1, 19, 4095, 4096, 4099, 8192, 12301, TAIL, N - 1]]
    for _ in range(5)]
libm = ctypes.CDLL('libm.so.6')
libm.expf.argtypes = [ctypes.c_float]
libm.expf.restype = ctypes.c_float
libm.fmaf.argtypes = [ctypes.c_float] * 3
libm.fmaf.restype = ctypes.c_float


def decay(tau):
    x = np.float32(-10000.) / np.float32(np.float32(tau) * np.float32(1e6))
    return np.float32(libm.expf(float(x)))


def advance(acc, d, count):
    if count == 0 or not np.any(acc):
        return acc.copy()
    # Seeding with each current accumulator gives exactly count sequential products.
    steps = np.full((count + 1, acc.size), d, np.float32)
    steps[0] = acc.ravel()
    return np.multiply.accumulate(steps, axis=0)[-1].reshape(acc.shape)


def reference(cells, tau, weight, fused=False):
    n = len(cells)
    mats = np.zeros((4, n, n), np.float32)
    acc = np.zeros((n, 2), np.float32)
    events = {}
    for i, stock in enumerate(cells):
        for k, lanes in stock:
            if 0 <= k < N:
                events.setdefault(k, np.zeros((n, 2), np.float32))[i] += np.array(
                    lanes[2 * weight:2 * weight + 2], np.float32)
    prev, d = 0, decay(tau)
    for k, current in sorted(events.items()):
        acc = advance(acc, d, k - prev)
        history = acc + np.float32(.5) * current
        for first in range(2):
            for last in range(2):
                lane = first * 2 + last
                if fused:
                    for j in range(n):
                        for i in range(n):
                            mats[lane, j, i] = libm.fmaf(float(current[i, last]),
                                float(history[j, first]), float(mats[lane, j, i]))
                else:
                    mats[lane] += history[:, first, None] * current[None, :, last]
        acc = (acc + current) * d
        prev = k + 1
    return mats


def rust_cells(cells):
    return 'vec![' + ','.join('vec![' + ','.join(
        f'({k}u32,[{",".join(repr(float(v)) + "f32" for v in u)}])' for k, u in stock)
        + ']' for stock in cells) + ']'


stage_path = BASE / 'yupei_dist/matrix_stage.rs'
stage = stage_path.read_text()
# Keep the actual complete module; compile its scalar fallback on x86 for comparison.
stage = stage.replace('#[cfg(any(not(target_arch = "x86_64"), test))]', '')
stage += r'''
pub fn audit_tail_phases() {
    let k0 = ((N_BLOCKS - 1) * BLOCK_BUCKETS) as u32;
    let len = N_BUCKETS - k0 as usize;
    let buf = vec![(k0, [1.25; N_U]), (k0 + 17, [2.5; N_U]),
                   (N_BUCKETS as u32 - 1, [0.375; N_U])];
    let mut gs = vec![-123.; BLOCK_BUCKETS * GP_STRIDE];
    let mut gv = gs.clone();
    let mut a_s = vec![3.25; GP_STRIDE]; let mut a_v = a_s.clone();
    phase1_scalar::<{ FAM_ALL }>(&mut gs, &mut a_s, &buf, 3, 0, k0);
    phase1_simd::<{ FAM_ALL }>(&mut gv, &mut a_v, &buf, 3, 0, k0);
    assert!(gs.iter().zip(&gv).all(|(a,b)|a.to_bits()==b.to_bits()));
    assert!(a_s.iter().zip(&a_v).all(|(a,b)|a.to_bits()==b.to_bits()));
    assert!(gs[len * GP_STRIDE..].iter().all(|&v|v == -123.));
    let mut ps=vec![0.375;PACK_W]; let mut pv=ps.clone();
    phase2_scalar::<{FAM_ALL}>(&mut ps,&gs,&buf,&[3],&[0,3],1,k0);
    phase2_simd::<{FAM_ALL}>(&mut pv,&gv,&buf,&[3],&[0,3],1,k0);
    assert!(ps.iter().zip(&pv).all(|(a,b)|a.to_bits()==b.to_bits()));
    println!("stage_tail_phase_checks=true");
    println!("stage_tail_cnt_tau1_acc={}", a_s[1].to_bits());
}
'''
native = r'''
mod fast_csv_reader { pub struct TradeRecord {
    pub time_us:i64, pub price:f32, pub volume:f32, pub turnover:f32,
    pub flag:i32, pub bid_order:i64, pub ask_order:i64,
} }
mod yupei_dist { #[path="__STAGE__"] pub mod matrix_stage; }
#[path="__SPLIT__"] mod split_mat;
fn run(name:&str,cells:Vec<Vec<(u32,[f32;4])>>,tau:f32) {
    use split_mat::*;
    for (label,w) in [("cnt",PmWeight::Cnt),("vol",PmWeight::Vol)] {
        let scalar=compute_split_mats_from_cells(&cells,tau,w);
        std::env::set_var("YUPEI_MATRIX_TWO_TARGET","0");
        let sse=compute_split_mats_simd(&cells,tau,w);
        std::env::set_var("YUPEI_MATRIX_TWO_TARGET","1");
        let avx=compute_split_mats_simd(&cells,tau,w);
        let swapped:Vec<_>=cells.iter().map(|c|c.iter().map(|&(k,u)|
            (k,[u[1],u[0],u[3],u[2]])).collect()).collect();
        let flip=compute_split_mats_simd(&swapped,tau,w);
        for t in 0..4 { for q in 0..scalar[t].len() {
            assert_eq!(scalar[t][q].to_bits(),sse[t][q].to_bits(),"SSE {name}/{label}");
            assert_eq!(scalar[t][q].to_bits(),avx[t][q].to_bits(),"AVX {name}/{label}");
            assert_eq!(scalar[3-t][q].to_bits(),flip[t][q].to_bits(),"flip {name}/{label}");
        } }
        println!("{name}/{label}={:?}",scalar.iter().flatten().map(|v|v.to_bits()).collect::<Vec<_>>());
    }
}
fn main() {
    assert!(std::arch::is_x86_feature_detected!("avx2"));
    assert!(std::arch::is_x86_feature_detected!("fma"));
    __RUNS__
    use yupei_dist::matrix_stage::*;
    assert_eq!(N_BLOCKS,348);
    audit_tail_phases();
    let mut st=StockPrep::empty("tail".into());
    st.cells=vec![(N_BUCKETS as u32-1,[1.;N_U]),(N_BUCKETS as u32,[100.;N_U])];
    let (mat,_)=compute_matrices(&[st],FAM_ALL);
    println!("stage_last_bucket_all28={:?}",mat.iter().map(|m|m[0].to_bits()).collect::<Vec<_>>());
    println!("avx2_hardware_exercised=true");
}
'''
runs = '\n'.join(f'run("{name}",{rust_cells(cells)},{30.0 if name == "mixed_gaps_odd_universe" else 1.0}f32);'
                 for name, cells in cases.items())
deps = ROOT / 'target/release-fast/deps'
rayon = deps / 'librayon-5a66e49cbfeb6155.rlib'
with tempfile.TemporaryDirectory(prefix='yupei_matrix_regression_') as d:
    tmp = Path(d)
    (tmp / 'matrix_stage.rs').write_text(stage)
    native = native.replace('__STAGE__', str(tmp / 'matrix_stage.rs')).replace(
        '__SPLIT__', str(BASE / 'yupei_pair/split_mat.rs')).replace('__RUNS__', runs)
    (tmp / 'main.rs').write_text(native)
    subprocess.run(['/home/chenzongwei/.cargo/bin/rustc', '+1.97.0', '--edition=2021',
                    '-Awarnings', '-C', 'opt-level=2', '-C', 'target-cpu=native',
                    '-L', f'dependency={deps}', '--extern', f'rayon={rayon}',
                    str(tmp / 'main.rs'), '-o', str(tmp / 'audit')], check=True)
    output = subprocess.check_output([str(tmp / 'audit')], text=True,
        env={**os.environ, 'RAYON_NUM_THREADS': '1'})
results = {k: json.loads(v) for k, v in (line.split('=', 1) for line in output.splitlines())}
checks = []
for name, cells in cases.items():
    for weight, label in enumerate(['cnt', 'vol']):
        tau = 30. if name == 'mixed_gaps_odd_universe' else 1.
        expected = reference(cells, tau, weight).view(np.uint32).ravel()
        actual = np.array(results[f'{name}/{label}'], np.uint32)
        assert np.array_equal(actual, expected), (name, label, actual, expected)
        checks.append(dict(case=name, weight=label, tau=tau, stocks=len(cells),
                           values=int(expected.size), python_bitwise=True,
                           scalar_sse_avx2_bitwise=True, swapped_pm_bitwise=True))
# Independent cnt accumulator at the end of the 688-bucket partial block.
a, d = np.float32(3.25), decay(1.)
for k in range(N - TAIL):
    a = np.float32((a + np.float32({0:1.25,17:2.5,687:.375}.get(k,0.))) * d)
assert int(a.view(np.uint32)) == results['stage_tail_cnt_tau1_acc']
expected28 = np.array([.5] * 10 + [1.] * 18, np.float32).view(np.uint32).tolist()
assert results['stage_last_bucket_all28'] == expected28
v = np.array(results['direction_audit_example/cnt'], np.uint32).view(np.float32).reshape(4,2,2)
assert v[1,0,1] + v[2,1,0] > .99
path = ROOT / 'reports/yupei_correctness_20261030/matrix_regression.json'
path.parent.mkdir(parents=True, exist_ok=True)
files = ['yupei_pair/split_mat.rs','yupei_pair/split_mat_two_target.rs','yupei_dist/matrix_stage.rs']
report = dict(source_root=str(SOURCE_ROOT),
    scope='真实矩阵源码单核小样本；独立 Python f32 递推；未重算全市场全天',
    cases=checks, compared_values=sum(c['values'] for c in checks),
    direction_rho_after=float(v[1,0,1]+v[2,1,0]),
    tail_self_energy_after=1., empty_block_interaction_after=float(np.array(
        results['empty_block_audit_example/cnt'],np.uint32).view(np.float32)[1]),
    blocks=348, final_block_buckets=688, excludes_bucket_N_and_after=True,
    stage_tail_scalar_simd_bitwise=True, stage_tail_unused_buffer_untouched=True,
    stage_tail_python_acc_bitwise=True, stage_all28_last_bucket_correct=True,
    flip_reuse='交换 P/M 和 volP/volM 后，重算矩阵逐位等于原矩阵 [MM,MP,PM,PP]；cnt/vol 均覆盖',
    interface_changes=False, rayon_threads=1, hardware='AVX2/FMA 检测后实跑 SSE 和双目标 AVX2',
    hashes={f:hashlib.sha256((BASE/f).read_bytes()).hexdigest() for f in files},
    limitation='测试为稀疏合成输入与此前 audit_errors.py 三个最小反例，不代表真实全天受影响列数。')
path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'report':str(path),'checks':len(checks),'values':report['compared_values']},ensure_ascii=False))
