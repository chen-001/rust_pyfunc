"""Actual reducer modules: isolated legacy equivalence, A→B→A, nested Rayon and threads.

Only the unused NetMats transport type is stubbed; no numerical function is mocked.
Builds a small independent crate in /tmp, without importing or installing rust_pyfunc.
Optional first argument selects the corrected source root (default: correctness sandbox).
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
NEW = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / 'sandbox/yupei_correctness_20261030'
OLD = ROOT / 'sandbox/yupei_multi_ks5_20_20261030'
PAIR = Path('src/yupei_pair')
net = (NEW / PAIR / 'net.rs').read_text()
net = net[net.index('pub(crate) struct NetMats {'):net.index('\nstruct Shared {')]
net = 'type TopkAcc = ();\n' + net
fixture = (NEW / PAIR / 'split.rs').read_text()
fixture = fixture[fixture.index('fn fake_matset('):fixture.index('/// 15 条链的因子列名')]
# Exercise both positive and negative Z thresholds, including specialised pairs.
fixture = fixture.replace('z: f(nn),', 'z: f(nn).into_iter().map(|v| v*8.0-4.0).collect(),')
fixture = fixture.replace('zs: f(nn),', 'zs: f(nn).into_iter().map(|v| v*8.0-4.0).collect(),')
modules = []
for module, source in [('legacy', OLD), ('corrected', NEW)]:
    paths = '\n'.join(f'#[path={json.dumps(str(source / PAIR / (name + ".rs")))}] mod {name};'
                      for name in ['reduce', 'reduce2', 'simd', 'topk_heap', 'bitonic_sort'])
    setter = 'reduce::set_soft(sig);' if module == 'legacy' else ''
    config = '' if module == 'legacy' else 'sig, '
    modules.append(f'''mod {module} {{
fn verbose() -> bool {{ false }}
mod net {{ {net} }}
{paths}
use reduce::{{MatSet, Out, SoftCfg}};
{fixture}
pub fn run(temp: f64, thr: f64) -> Vec<u64> {{
    let sig = SoftCfg {{ temp, thr }};
    {setter}
    let n = 64;
    let ms = fake_matset(n);
    let ok = vec![true; n];
    let industry: Vec<i32> = (0..n).map(|i| (i % 7) as i32).collect();
    let lnt: Vec<f64> = (0..n).map(|i| ((i+1) as f64).ln()).collect();
    let mut out = Out::new();
    reduce::produce_one_mat(&ms, "test_", n, &ok, Some(&industry), Some(&lnt), true, {config}&mut out);
    out.cols.into_iter().flatten().map(f64::to_bits).collect()
}}
}}''')
main = r'''
use rayon::prelude::*;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args[1].as_str() {
        "legacy" | "corrected" => {
            let temp = args[2].parse().unwrap(); let thr = args[3].parse().unwrap();
            let run = if args[1] == "legacy" { legacy::run } else { corrected::run };
            println!("{:?}", run(temp, thr));
        }
        "aba" => {
            let configs = [(0.3,2.0),(1.4,0.25),(0.3,2.0),(0.9,2.0),(0.3,2.0)];
            let outputs: Vec<_> = configs.iter().map(|&(t,h)|corrected::run(t,h)).collect();
            println!("{:?}", outputs);
        }
        "rayon" => {
            let outputs: Vec<_> = (0..8).into_par_iter().map(|i| {
                if i%2 == 0 { corrected::run(0.3,2.0) } else { corrected::run(1.4,0.25) }
            }).collect();
            println!("{:?}", outputs);
        }
        "threads" => {
            let threads: Vec<_> = (0..8).map(|i|std::thread::spawn(move || {
                if i%2 == 0 { corrected::run(0.3,2.0) } else { corrected::run(1.4,0.25) }
            })).collect();
            let outputs: Vec<_> = threads.into_iter().map(|t|t.join().unwrap()).collect();
            println!("{:?}", outputs);
        }
        _ => panic!("unknown mode"),
    }
}
'''
env = os.environ.copy()
env['PATH'] = '/home/chenzongwei/.cargo/bin:' + env['PATH']
env['RAYON_NUM_THREADS'] = '4'
report = {'source': str(NEW), 'baseline': str(OLD), 'rayon_threads': 4, 'checks': []}
with tempfile.TemporaryDirectory(prefix='yupei_soft_config_') as tmp:
    tmp = Path(tmp)
    (tmp/'src').mkdir()
    (tmp/'Cargo.toml').write_text('[package]\nname="yupei_soft_config_check"\nversion="0.1.0"\nedition="2021"\n[dependencies]\nrayon="1.8"\n[profile.release]\nlto=false\ncodegen-units=16\n')
    (tmp/'src/main.rs').write_text('\n'.join(modules) + main)
    target = '/tmp/yupei_soft_config_target'
    subprocess.run(['cargo', 'build', '--release', '--offline', '--jobs', '2', '--target-dir', target],
                   cwd=tmp, env=env, check=True, stdout=subprocess.DEVNULL)
    binary = f'{target}/release/yupei_soft_config_check'
    def run(*args):
        return json.loads(subprocess.check_output([binary, *map(str, args)], env=env, text=True))
    isolated = []
    for label, temp, thr in [('A',0.3,2.0), ('B',1.4,0.25), ('C',0.9,2.0)]:
        old, new = run('legacy',temp,thr), run('corrected',temp,thr)
        assert old == new, f'{label}: single-call numerical behavior changed'
        isolated.append(new)
        report['checks'].append({'case': f'isolated_legacy_equals_corrected_{label}', 'values': len(new), 'bit_mismatches': 0})
    assert isolated[0] != isolated[1] and isolated[0] != isolated[2], 'parameters must actually affect outputs'
    aba = run('aba')
    assert aba == [isolated[i] for i in [0,1,0,2,0]]
    report['checks'].append({'case':'same_process_A_B_A_C_A', 'calls':5, 'bit_mismatches':0})
    for mode in ['rayon', 'threads']:
        outputs = run(mode)
        assert outputs == [isolated[i%2] for i in range(8)], f'{mode}: parameter contamination'
        report['checks'].append({'case':f'concurrent_{mode}', 'calls':8, 'bit_mismatches':0})
    report['changed_values_A_to_B'] = sum(a!=b for a,b in zip(*isolated[:2]))
    report['changed_values_temperature_only_A_to_C'] = sum(a!=b for a,b in zip(isolated[0],isolated[2]))
report['source_sha256'] = {str(p.relative_to(NEW)):hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in [NEW/PAIR/'reduce.rs', NEW/PAIR/'reduce2.rs']}
out = ROOT/'reports/yupei_correctness_20261030/soft_config_regression.json'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
