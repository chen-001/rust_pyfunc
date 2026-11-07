"""Compile the archived candidate 3 SharedRho helper; never build the library.

Checks all family input bit patterns and rejects dangling views/source mutation
at compile time. No market data or performance benchmark is used.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT / "results/speed/candidate3_sources.json"
manifest = {row["path"]: row["sha256"] for row in json.loads(manifest_path.read_text())}
source_path = ROOT / "frozen/candidate3_src/src/yupei_gpt/network.rs"
source_bytes = source_path.read_bytes()
assert hashlib.sha256(source_bytes).hexdigest() == manifest["src/yupei_gpt/network.rs"], "Archived candidate 3 source hash mismatch"
source = source_bytes.decode()
baseline = (ROOT / "frozen/candidate2_src/src/yupei_gpt/network.rs").read_text()
helper = source.split("// BEGIN SHARED_RHO_EXACT_INPUTS\n", 1)[1].split("// END SHARED_RHO_EXACT_INPUTS", 1)[0]
scan = lambda text: text.split("fn scan_pairs(", 1)[1].split("/// Moment/Z/fit products", 1)[0]
assert scan(source) == scan(baseline), "The exact pair scan or Acc merge changed"
prefix = "#![allow(dead_code, private_interfaces)]\nstruct Kernel { rho: [Vec<f32>; 9] }\n" + helper
checks = r'''
fn assert_send_sync<T: Send + Sync>() {}
fn main() {
    assert_send_sync::<SharedRho<'_>>();
    let patterns = [0u32, 0x8000_0000, 1, 0x8000_0001, 0x3f80_0000,
        0xbf80_0000, 0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000,
        0x7fc0_0001, 0xffc0_0002, 0x7f80_0001];
    let families = [(0,0,0), (0,0,1), (0,0,2), (1,1,0), (1,1,1), (1,1,2),
        (1,2,0), (1,2,1), (1,2,2), (2,2,0), (2,2,1), (2,2,2)];
    for n in [0usize, 1, 2, 8, 17, 64] {
        let kernel = Kernel { rho: std::array::from_fn(|k| (0..n*n)
            .map(|j| f32::from_bits(patterns[(j + k) % patterns.len()])).collect()) };
        let shared = SharedRho::new(&kernel);
        assert!(shared.matches(&kernel));
        let other = Kernel { rho: kernel.rho.clone() };
        assert!(!shared.matches(&other));
        for (a, b, c) in families {
            for k in [a*3+c, b*3+c] {
                let old: Vec<f32> = kernel.rho[k].iter()
                    .map(|&v| if v.is_finite() { v } else { 0. }).collect();
                let new = shared.matrix(k);
                assert_eq!(old.len(), new.len());
                for (&x, &y) in old.iter().zip(new) { assert_eq!(x.to_bits(), y.to_bits()); }
                // Repeated family access returns the same physical allocation.
                assert_eq!(new.as_ptr(), shared.matrix(k).as_ptr());
            }
            if a == b { assert_eq!(shared.matrix(a*3+c).as_ptr(), shared.matrix(b*3+c).as_ptr()); }
        }
    }
    println!("{{\"size_cases\":6,\"families_per_size\":12,\"input_bits_equal\":true,\"same_input_allocation_reused\":true,\"send_sync\":true,\"wrong_kernel_rejected\":true}}");
}
'''
dangling = r'''
fn leak(kernel: &Kernel) -> &[f32] {
    let shared = SharedRho::new(kernel);
    shared.matrix(0)
}
fn main() {}
'''
mutation = r'''
fn main() {
    let mut kernel = Kernel { rho: std::array::from_fn(|_| vec![0.]) };
    let shared = SharedRho::new(&kernel);
    kernel.rho[0][0] = 1.;
    println!("{}", shared.matrix(0)[0]);
}
'''
with tempfile.TemporaryDirectory(prefix="gpt_shared_rho_") as tmp:
    directory = Path(tmp)
    results = {}
    for name, body, expected_error in [("exact", checks, None), ("dangling", dangling, "E0515"), ("mutation", mutation, "E0502")]:
        rs, binary = directory / f"{name}.rs", directory / name
        rs.write_text(prefix + body)
        compiled = subprocess.run(["rustc", "--edition=2021", "-O", str(rs), "-o", str(binary)], capture_output=True, text=True)
        results[f"{name}_compile_exit_code"] = compiled.returncode
        if expected_error:
            assert compiled.returncode != 0 and expected_error in compiled.stderr, compiled.stderr
            results[f"{name}_rejected_with"] = expected_error
        else:
            assert compiled.returncode == 0, compiled.stderr
            checked = subprocess.run([str(binary)], capture_output=True, text=True)
            results["test_exit_code"] = checked.returncode
            assert checked.returncode == 0, checked.stderr
            results.update(json.loads(checked.stdout))
results.update({"scope": "isolated_actual_shared_rho_helper_no_market_data_no_timing",
    "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(), "scan_pairs_source_unchanged": True,
    "source_path": str(source_path), "source_sha256": manifest["src/yupei_gpt/network.rs"],
    "source_manifest_path": str(manifest_path)})
target = ROOT / "results/speed/gpt_shared_rho_exact.json"
target.write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results))
