# -*- coding: utf-8 -*-
"""第二层插桩：把 price_stats_v3 拆成 组准备 / 逐记录主循环。"""
import io
P = '/home/chenzongwei/rust_pyfunc/perf_sandbox/hm076_bench/src/hm076.rs'
s = io.open(P, encoding='utf-8').read()

def rep(old, new, n=1):
    global s
    c = s.count(old)
    assert c == n, 'anchor %d != %d: %r' % (c, n, old[:60])
    s = s.replace(old, new)

rep('pub static C_ALLOCS: AtomicU64 = AtomicU64::new(0);\n',
    'pub static C_ALLOCS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static P: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];\n'
    'pub static C_P_GROUPS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_P_RECORDS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_P_ELEMS: AtomicU64 = AtomicU64::new(0);\n'
    'pub static C_P_SORTS: AtomicU64 = AtomicU64::new(0);\n')

rep('''    let volume_ranges = find_volume_ranges(volume);
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

    for (_, start_idx, end_idx) in volume_ranges.iter() {''',
'''    let volume_ranges = find_volume_ranges(volume);
    let percentages = [0.01, 0.02, 0.03, 0.04, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50];

    let mut __tp = std::time::Instant::now();
    for (_, start_idx, end_idx) in volume_ranges.iter() {''')

rep('''        buy_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sell_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let max_records = buy_records.len().max(sell_records.len());''',
'''        buy_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        sell_records.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        P[0].fetch_add(__tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
        C_P_GROUPS.fetch_add(1, Ordering::Relaxed);
        C_P_SORTS.fetch_add(2, Ordering::Relaxed);
        let __tm = std::time::Instant::now();

        let max_records = buy_records.len().max(sell_records.len());''')

rep('''            let available = distances_buffer.len();
            if available < min_count {
                continue;
            }''',
'''            let available = distances_buffer.len();
            C_P_RECORDS.fetch_add(1, Ordering::Relaxed);
            C_P_ELEMS.fetch_add(available as u64, Ordering::Relaxed);
            if available < min_count {
                continue;
            }''')

rep('''            if max_needed < available {
                distances_buffer
                    .select_nth_unstable_by(max_needed, |a, b| a.0.partial_cmp(&b.0).unwrap());
                distances_buffer[..=max_needed]
                    .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            } else {
                distances_buffer.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            }''',
'''            if max_needed < available {
                distances_buffer
                    .select_nth_unstable_by(max_needed, |a, b| a.0.partial_cmp(&b.0).unwrap());
                distances_buffer[..=max_needed]
                    .sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                C_P_SORTS.fetch_add(2, Ordering::Relaxed);
            } else {
                distances_buffer.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                C_P_SORTS.fetch_add(1, Ordering::Relaxed);
            }''')

rep('''                stds[global_idx * 10 + pct_idx] = variance.max(0.0).sqrt();
            }
        }
    }
    (means, stds)''',
'''                stds[global_idx * 10 + pct_idx] = variance.max(0.0).sqrt();
            }
        }
        P[1].fetch_add(__tm.elapsed().as_nanos() as u64, Ordering::Relaxed);
        __tp = std::time::Instant::now();
    }
    (means, stds)''')

io.open(P, 'w', encoding='utf-8').write(s)
print('price_stats_v3 instrumented')
