p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/compute.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep("""    // ---- 2. 并行读 + 每股预处理（universe 固定；失败的股票以空行保留）----
    let preps: Vec<Option<StockPrep>> = codes""",
"""    let prof = std::env::var("YUPEI_PROF").is_ok();
    let prof_read = std::time::Instant::now();
    // ---- 2. 并行读 + 每股预处理（universe 固定；失败的股票以空行保留）----
    let preps: Vec<Option<StockPrep>> = codes""", 't read')
rep("""    // ---- 3. 37 张矩阵 + 统计量 ----
    let (mats, stats) = compute_matrices(&stocks, need);""",
"""    if prof { eprintln!("[prof] 读数据+每股预处理 {:.1}s", prof_read.elapsed().as_secs_f64()); }
    // ---- 3. 37 张矩阵 + 统计量 ----
    let prof_mats = std::time::Instant::now();
    let (mats, stats) = compute_matrices(&stocks, need);
    if prof { eprintln!("[prof] 算矩阵 {:.1}s", prof_mats.elapsed().as_secs_f64()); }""", 't mats')
rep("""    // ---- 4. 行业（直读 h5, 进程内只读一次; 读不到直接报错）----
    let industry = super::industry::industry_of(date, &codes_out)?;""",
"""    // ---- 4. 行业（直读 h5, 进程内只读一次; 读不到直接报错）----
    let prof_ind = std::time::Instant::now();
    let industry = super::industry::industry_of(date, &codes_out)?;
    if prof { eprintln!("[prof] 行业 {:.1}s", prof_ind.elapsed().as_secs_f64()); }""", 't ind')
open(p,'w').write(s)
print('\n'.join(done))
