# -*- coding: utf-8 -*-
"""给 sandbox 里的 hm076.rs 副本插桩计时；只改 perf_sandbox/hm076_bench/src/hm076.rs。"""
import io, sys

P = '/home/chenzongwei/rust_pyfunc/perf_sandbox/hm076_bench/src/hm076.rs'
s = io.open(P, encoding='utf-8').read()

def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, 'anchor count %d != %d for: %r' % (c, n, old[:70])
    s = s.replace(old, new)

# 1) 计时用的静态计数器
rep('use crate::features_f64;\n',
    'use crate::features_f64;\n'
    'use std::sync::atomic::{AtomicU64, Ordering};\n'
    'pub static T: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];\n'
    'pub static C_GROUPS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_RECORDS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_ELEMS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_SORTS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_ALLOCS: AtomicU64 = AtomicU64::new(0);\n')

# 2) compute_stock 三个阶段
rep('''    let rows = read_day(code, date)?;
    let day = build_day(rows);
    let res1 = one(&day, "same");
    let res2 = one(&day, "diff");''',
'''    let __t = std::time::Instant::now();
    let rows = read_day(code, date)?;
    T[0].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let __t = std::time::Instant::now();
    let day = build_day(rows);
    T[1].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let __t = std::time::Instant::now();
    let res1 = one(&day, "same");
    T[2].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let __t = std::time::Instant::now();
    let res2 = one(&day, "diff");
    T[2].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);''')

# 3) 组内主循环的计数
rep('''        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }''',
'''        let group_size = self.indices.len();
        if group_size < min_count {
            return;
        }
        C_GROUPS.fetch_add(1, Ordering::Relaxed);''')
rep('''            if target_indices.len() < min_count {
                continue;
            }''',
'''            if target_indices.len() < min_count {
                continue;
            }
            C_RECORDS.fetch_add(1, Ordering::Relaxed);
            C_ELEMS.fetch_add(target_indices.len() as u64, Ordering::Relaxed);
            C_ALLOCS.fetch_add(2, Ordering::Relaxed);''')
rep('''        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());''',
'''        C_SORTS.fetch_add(1, Ordering::Relaxed);
        time_distances.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());''')
rep('''    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());''',
'''    C_SORTS.fetch_add(1, Ordering::Relaxed);
    C_ALLOCS.fetch_add(1, Ordering::Relaxed);
    C_ELEMS.fetch_add(total_count as u64, Ordering::Relaxed);
    all_prices.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());''')

# 4) ultra_sorted 三段
rep('    let exchtime_data: Vec<f64> = exchtime_ns.iter().map(|&t| t as f64 / 1e9).collect();',
    '    let __t = std::time::Instant::now();\n    let exchtime_data: Vec<f64> = exchtime_ns.iter().map(|&t| t as f64 / 1e9).collect();')
rep('    let mut order_results = vec![f64::NAN; orders.len() * 22];',
    '    T[3].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);\n'
    '    let __t = std::time::Instant::now();\n'
    '    let mut order_results = vec![f64::NAN; orders.len() * 22];')
rep('''    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast(&mut order_results, min_count, use_flag);
    }''',
'''    for group in order_groups.iter() {
        group.compute_all_indicators_ultra_fast(&mut order_results, min_count, use_flag);
    }
    T[4].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let __t = std::time::Instant::now();''')
rep('    let mut results = vec![f64::NAN; n * 27];',
    '    T[5].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);\n'
    '    let mut results = vec![f64::NAN; n * 27];')

# 5) one() 内部四段
rep('''        direction,
    );

    // dropna''', '''        direction,
    );

    let __t = std::time::Instant::now();
    // dropna''')
rep('    let mut res: Vec<f64> = Vec::with_capacity(ONE_LEN);',
    '    T[6].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);\n'
    '    let __t = std::time::Instant::now();\n'
    '    let mut res: Vec<f64> = Vec::with_capacity(ONE_LEN);')
rep('    // ── v3 那 420 个值',
    '    T[9].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);\n'
    '    let __t = std::time::Instant::now();\n'
    '    // ── v3 那 420 个值')
rep('''    let (means, stds) = price_stats_v3(&d.volume, &d.ns, &price_v, &d.flag, 10, direction);''',
'''    let (means, stds) = price_stats_v3(&d.volume, &d.ns, &price_v, &d.flag, 10, direction);
    T[7].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let __t = std::time::Instant::now();''')
rep('''        }
    }
    res
}

// ────────────────────────── 入口''',
'''        }
    }
    T[8].fetch_add(__t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    res
}

// ────────────────────────── 入口''')

io.open(P, 'w', encoding='utf-8').write(s)
print('instrumented ok, lines =', s.count(chr(10)) + 1)
