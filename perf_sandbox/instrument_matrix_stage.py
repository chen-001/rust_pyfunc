import io
p = '/home/chenzongwei/rust_pyfunc/src/yupei_dist/matrix_stage.rs'
s = open(p).read()
subs = []

if 't_gather' not in s:
    subs.append(("    for block in 0..N_BLOCKS {\n        let k0 = block * BLOCK_BUCKETS;\n        let k1 = k0 + BLOCK_BUCKETS;\n        // ---- 1.",
                 "    let prof = std::env::var(\"YUPEI_PROF\").is_ok();\n    let t_all = std::time::Instant::now();\n    let phase1_ns = std::sync::atomic::AtomicU64::new(0);\n    let phase2_ns = std::sync::atomic::AtomicU64::new(0);\n    let mut gather_ns: u128 = 0;\n    for block in 0..N_BLOCKS {\n        let k0 = block * BLOCK_BUCKETS;\n        let k1 = k0 + BLOCK_BUCKETS;\n        let t_gather = std::time::Instant::now();\n        // ---- 1."))
if 'gather_ns +=' not in s:
    subs.append(("        for i in 0..n {\n            cursors[i] += counts[i];\n        }\n\n        // ---- 2. tile 并行",
                 "        for i in 0..n {\n            cursors[i] += counts[i];\n        }\n        gather_ns += t_gather.elapsed().as_nanos();\n\n        // ---- 2. tile 并行"))
if 'phase1_ns.fetch_add' not in s:
    subs.append(("            #[cfg(target_arch = \"x86_64\")]\n            phase1_simd(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);\n            #[cfg(not(target_arch = \"x86_64\"))]\n            phase1_scalar(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);",
                 "            let _t1 = std::time::Instant::now();\n            #[cfg(target_arch = \"x86_64\")]\n            phase1_simd(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);\n            #[cfg(not(target_arch = \"x86_64\"))]\n            phase1_scalar(gp, acc_mem, block_buf_ref, bcnt, boff, k0 as u32);\n            phase1_ns.fetch_add(_t1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);"))
if 'phase2_ns.fetch_add' not in s:
    subs.append(("            #[cfg(target_arch = \"x86_64\")]\n            phase2_simd(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);\n            #[cfg(not(target_arch = \"x86_64\"))]\n            phase2_scalar(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);\n        });\n    }",
                 "            let _t2 = std::time::Instant::now();\n            #[cfg(target_arch = \"x86_64\")]\n            phase2_simd(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);\n            #[cfg(not(target_arch = \"x86_64\"))]\n            phase2_scalar(pack, gp, block_buf_ref, counts_ref, offs_ref, n, k0 as u32);\n            phase2_ns.fetch_add(_t2.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n        });\n    }"))
if 't_flush' not in s:
    subs.append(("    // ---- 日末刷入 S[m][a][b]（并行; 每 tile 只写第 b 列, 列互不相交）----",
                 "    let t_flush = std::time::Instant::now();\n    // ---- 日末刷入 S[m][a][b]（并行; 每 tile 只写第 b 列, 列互不相交）----"))
if 'prof] compute_matrices' not in s:
    subs.append(("    (mats, stats)\n}\n#[cfg(test)]",
                 "    if prof {\n        eprintln!(\n            \"[prof] compute_matrices 总 {:.1}s | 收集单元 {:.1}s | phase1 线程累计 {:.1}s | phase2 线程累计 {:.1}s | 日末刷入 {:.1}s\",\n            t_all.elapsed().as_secs_f64(), gather_ns as f64 / 1e9,\n            phase1_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,\n            phase2_ns.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9,\n            t_flush.elapsed().as_secs_f64(),\n        );\n    }\n    (mats, stats)\n}\n#[cfg(test)]"))
done = []
for old, new in subs:
    if old not in s:
        done.append('MISS:' + old[:40]); continue
    s = s.replace(old, new, 1); done.append('OK:' + new[:28])
open(p, 'w').write(s)
print('\n'.join(done))
print('markers:', all(k in s for k in ['t_gather','gather_ns +=','phase1_ns.fetch_add','phase2_ns.fetch_add','t_flush','prof] compute_matrices']))
