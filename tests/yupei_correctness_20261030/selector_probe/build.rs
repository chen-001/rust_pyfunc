use std::{env, fs, path::PathBuf};
fn main() {
    let manifest=PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let workspace=manifest.ancestors().nth(3).unwrap();
    let root=env::var_os("YUPEI_SELECTOR_SOURCE_ROOT").map(PathBuf::from)
        .unwrap_or_else(||workspace.join("sandbox/yupei_correctness_20261030"));
    let baseline=env::var_os("YUPEI_SELECTOR_BASELINE_ROOT").map(PathBuf::from)
        .unwrap_or_else(||workspace.to_owned());
    let mut modules=String::new();
    for (name,file,base) in [("bitonic_sort","bitonic_sort.rs",&root),("correct","simd.rs",&root),
        ("net","net.rs",&root),("topk","topk.rs",&root),("mirror_stage","mirror_stage.rs",&root),
        ("legacy","simd.rs",&baseline)] {
        let path=base.join("src/yupei_pair").join(file).canonicalize().unwrap();
        modules.push_str(&format!("#[path = {path:?}] mod {name};\n"));
        println!("cargo:rerun-if-changed={}",path.display());
    }
    fs::write(PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("source_modules.rs"),modules).unwrap();
    println!("cargo:rerun-if-env-changed=YUPEI_SELECTOR_SOURCE_ROOT");
    println!("cargo:rerun-if-env-changed=YUPEI_SELECTOR_BASELINE_ROOT");
}
