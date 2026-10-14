p = '/home/chenzongwei/rust_pyfunc/src/yupei_dist/compute.rs'
s = open(p).read()
subs = []
if 'prof_t_read' not in s:
    subs.append(("    // ---- 2. 并行读 + 每股预处理（universe 固定；失败的股票以空行保留）----\n    let preps: Vec<Option<StockPrep>> = codes",
                 "    let prof = std::env::var(\"YUPEI_PROF\").is_ok();\n    let prof_t_read = std::time::Instant::now();\n    // ---- 2. 并行读 + 每股预处理（universe 固定；失败的股票以空行保留）----\n    let preps: Vec<Option<StockPrep>> = codes"))
if 'prof_t_mats' not in s:
    subs.append(("    // ---- 3. 37 张矩阵 + 统计量 ----\n    let (mats, stats) = compute_matrices(&stocks);",
                 "    if prof { eprintln!(\"[prof] 读+每股预处理 {:.1}s\", prof_t_read.elapsed().as_secs_f64()); }\n    // ---- 3. 37 张矩阵 + 统计量 ----\n    let prof_t_mats = std::time::Instant::now();\n    let (mats, stats) = compute_matrices(&stocks);\n    if prof { eprintln!(\"[prof] compute_matrices {:.1}s\", prof_t_mats.elapsed().as_secs_f64()); }"))
if 'prof_t_ind' not in s:
    subs.append(("    // ---- 4. 行业（直读 h5, 进程内只读一次; 读不到直接报错）----\n    let industry = super::industry::industry_of(date, &codes_out)?;",
                 "    // ---- 4. 行业（直读 h5, 进程内只读一次; 读不到直接报错）----\n    let prof_t_ind = std::time::Instant::now();\n    let industry = super::industry::industry_of(date, &codes_out)?;\n    if prof { eprintln!(\"[prof] 行业 {:.1}s\", prof_t_ind.elapsed().as_secs_f64()); }"))
done = []
for old, new in subs:
    if old not in s: done.append('MISS:' + old[:50]); continue
    s = s.replace(old, new, 1); done.append('OK:' + new[:24])
open(p, 'w').write(s)
print('\n'.join(done))
print('markers:', all(k in s for k in ['prof_t_read','prof_t_mats','prof_t_ind']))

q = '/home/chenzongwei/rust_pyfunc/src/yupei_pair/mod.rs'
t = open(q).read()
if 'check 用时' not in t:
    old = "            let fails = check::check_net(&net, n, &mask, mat);"
    new = "            let _tc = std::time::Instant::now();\n            let fails = check::check_net(&net, n, &mask, mat);\n            eprintln!(\"[prof] check[{mat}] {:.2}s\", _tc.elapsed().as_secs_f64());"
    assert old in t
    t = t.replace(old, new, 1)
    open(q, 'w').write(t)
    print('OK: check 计时')
else:
    print('SKIP: check 计时已有')
