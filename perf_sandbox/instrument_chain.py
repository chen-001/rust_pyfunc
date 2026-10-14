p = '/home/chenzongwei/rust_pyfunc/src/yupei_pair/mod.rs'
s = open(p).read()
done = []
if 't_chain' not in s:
    old = "    type MatOut = io::Result<(Vec<String>, Vec<Vec<f32>>)>;\n    let results: Vec<MatOut> = mats"
    new = ("    let prof = std::env::var(\"YUPEI_PROF\").is_ok();\n"
           "    let t_e = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_net = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_chk = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_red = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_top = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_mrg = std::sync::atomic::AtomicU64::new(0);\n"
           "    let t_chain = std::sync::atomic::AtomicU64::new(0);\n"
           "    type MatOut = io::Result<(Vec<String>, Vec<Vec<f32>>)>;\n    let results: Vec<MatOut> = mats")
    assert old in s; s = s.replace(old, new, 1); done.append('init')
if 't_e.fetch_add' not in s:
    old = "            // 自能量 E>0 = 当天有成交的股票 = 配对参与集与候选 X 集\n            let e: Vec<f64>"
    new = "            let _t0 = std::time::Instant::now();\n            // 自能量 E>0 = 当天有成交的股票 = 配对参与集与候选 X 集\n            let e: Vec<f64>"
    assert old in s; s = s.replace(old, new, 1)
    old2 = "            // ---- net ----\n            let net = net::net_one_mat"
    new2 = ("            let _t1 = std::time::Instant::now();\n"
            "            t_e.fetch_add(_t0.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n"
            "            // ---- net ----\n            let net = net::net_one_mat")
    assert old2 in s; s = s.replace(old2, new2, 1)
    old3 = "            let _tc = std::time::Instant::now();\n            let fails = check::check_net"
    new3 = ("            let _t2 = std::time::Instant::now();\n"
            "            t_net.fetch_add(_t1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n"
            "            let _tc = std::time::Instant::now();\n            let fails = check::check_net")
    assert old3 in s; s = s.replace(old3, new3, 1)
    old4 = "            let ms = matset_from_net(net, n);"
    new4 = ("            let _t3 = std::time::Instant::now();\n"
            "            let ms = matset_from_net(net, n);")
    assert old4 in s; s = s.replace(old4, new4, 1)
    old5 = "            // ---- topk：K 档 × 双口径（rho/e 复用 ms 里的）----\n            let (tn, tc) = topk_one_mat"
    new5 = ("            let _t4 = std::time::Instant::now();\n"
            "            t_red.fetch_add((_tc.elapsed().as_nanos() + _t3.elapsed().as_nanos()) as u64, std::sync::atomic::Ordering::Relaxed);\n"
            "            // ---- topk：K 档 × 双口径（rho/e 复用 ms 里的）----\n            let (tn, tc) = topk_one_mat")
    assert old5 in s; s = s.replace(old5, new5, 1)
    old6 = "            // 合并 names/cols（reduce cols 是 f64，topk cols 是 f32，统一成 f32）"
    new6 = ("            let _t5 = std::time::Instant::now();\n"
            "            t_top.fetch_add(_t4.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n"
            "            // 合并 names/cols（reduce cols 是 f64，topk cols 是 f32，统一成 f32）")
    assert old6 in s; s = s.replace(old6, new6, 1)
    old7 = "            Ok((names, cols))\n        })\n        .collect();"
    new7 = ("            t_mrg.fetch_add(_t5.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);\n"
            "            Ok((names, cols))\n        })\n        .collect();")
    assert old7 in s; s = s.replace(old7, new7, 1)
    done.append('stages')
if 'prof] 链总计' not in s:
    old = "    let f = names.len();\n    if f == 0 {"
    new = ("    if prof {\n"
           "        let g = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9;\n"
           "        eprintln!(\"[prof] 链各段墙钟之和 {:.1}s = e/mask {:.1} + net {:.1} + check+reduce {:.1} + topk {:.1} + 合并 {:.1}\",\n"
           "            t_chain.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e9, g(&t_e), g(&t_net), g(&t_red), g(&t_top), g(&t_mrg));\n"
           "    }\n"
           "    let f = names.len();\n    if f == 0 {")
    assert old in s; s = s.replace(old, new, 1); done.append('print')
open(p, 'w').write(s)
print('patched:', done, 'markers', all(k in s for k in ['t_chain','t_e.fetch_add','t_net.fetch_add','prof] 链总计']))
