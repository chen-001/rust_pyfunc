import io
P = '/home/chenzongwei/rust_pyfunc/perf_sandbox/hm076_bench/src/hm076.rs'
s = io.open(P, encoding='utf-8').read()
old = '''    T[5].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let mut results = vec![f64::NAN; n * 27];
    for i in 0..n {'''
new = '''    let mut results = vec![f64::NAN; n * 27];
    let __t5 = std::time::Instant::now();
    for i in 0..n {'''
assert s.count(old) == 1
s = s.replace(old, new)
old2 = '''    }
    results
}

/// hm076 全是逐股因子'''
if s.count(old2) != 1:
    # 找 ultra_sorted 的结尾（后面紧跟 price_stats_v3 注释）
    old2 = '''    }
    results
}

// ────────── rp.calculate_trade_price_statistics_by_volume_v3 ──────────'''
assert s.count(old2) == 1, s.count(old2)
s = s.replace(old2, '''    }
    T[5].fetch_add(__t5.elapsed().as_nanos() as u64, Ordering::Relaxed);
    results
}

// ────────── rp.calculate_trade_price_statistics_by_volume_v3 ──────────''')
io.open(P, 'w', encoding='utf-8').write(s)
print('T[5] 位置修正完成')
