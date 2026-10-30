"""Compile the two actual source prefixes; compare exact rho/e bits, without importing the library."""
from pathlib import Path
import json
import subprocess
import tempfile

root = Path(__file__).resolve().parents[3]
net = (root / 'src/yupei_pair/net.rs').read_text()
new = (root / 'sandbox/yupei_multi_ks5_20_20261030/src/yupei_pair/reduce.rs').read_text()
net = net[net.index('pub(crate) fn net_one_mat('):]
net = net[net.index('    let e: Vec<f64>'):net.index('    let rhot = &rho;')]
new = new[new.index('pub(crate) fn prepare_shared_rho_reduce('):]
new = new[new.index('    let e: Vec<f64>'):new.index('    let single = best_single_of(')]
functions = []
for name, body in [('reference', net), ('candidate', new)]:
    body = body.replace('.par_chunks_mut(', '.chunks_mut(')
    functions.append(f'fn {name}(sdir: &[f32], n: usize) -> (Vec<f64>, Vec<f32>) {{\n{body}\n(e, rho)\n}}')
main = r'''
fn main() {
    let mut seed = 0xb40be71d6e3c9a31u64;
    let mut cases = 0;
    let mut values = 0;
    for n in [1usize, 2, 17, 65, 128] {
        for trial in 0..12 {
            let mut sdir = vec![0f32; n*n];
            for x in &mut sdir {
                seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17;
                *x = ((seed >> 32) as f32 / (u32::MAX as f32)) * 1e7;
            }
            if trial % 3 == 0 { sdir[0] = 0.0; }
            if trial % 3 == 1 { sdir[0] = f32::MIN_POSITIVE; }
            let (ea, ra) = reference(&sdir, n);
            let (eb, rb) = candidate(&sdir, n);
            assert!(ea.iter().zip(&eb).all(|(a,b)| a.to_bits() == b.to_bits()));
            assert!(ra.iter().zip(&rb).all(|(a,b)| a.to_bits() == b.to_bits()));
            cases += 1; values += n*n+n;
        }
    }
    println!("{{\"cases\":{cases},\"rho_and_e_values_compared\":{values},\"bit_mismatches\":0}}");
}
'''
with tempfile.TemporaryDirectory(prefix='yupei_shared_roundtrip_') as tmp:
    src = Path(tmp) / 'roundtrip.rs'
    binary = Path(tmp) / 'roundtrip'
    src.write_text('\n'.join(functions) + main)
    subprocess.run(['/home/chenzongwei/.cargo/bin/rustc', '-Awarnings', '-O', str(src), '-o', str(binary)], check=True)
    report = json.loads(subprocess.check_output([str(binary)], text=True))
report.update(reference_source='src/yupei_pair/net.rs', candidate_source='sandbox/yupei_multi_ks5_20_20261030/src/yupei_pair/reduce.rs', note='Extracted the actual e/rho construction statements, replacing only Rayon chunk iteration with standard chunk iteration.')
(root/'sandbox/yupei_shared_reduce_20261030/roundtrip_evidence.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
