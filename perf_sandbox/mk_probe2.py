import pathlib
src = pathlib.Path('.v5_net.rs').read_text()
def sub1(t, old, new, tag):
    assert t.count(old) == 1, (tag, t.count(old))
    return t.replace(old, new)

STATICS = '''
// [PROBE2] topk 块内部分段（只在 YUPEI_PROF=1 时累加）
use std::sync::atomic::{AtomicU64, Ordering};
static P2_OTHER_NS: AtomicU64 = AtomicU64::new(0);
static P2_KH_NS: AtomicU64 = AtomicU64::new(0);
static P2_KG_NS: AtomicU64 = AtomicU64::new(0);
static P2_ABS_NS: AtomicU64 = AtomicU64::new(0);
'''
src = sub1(src, 'use super::topk::{absorb_topk, Acc as TopkAcc};',
           'use super::topk::{absorb_topk, Acc as TopkAcc};\n' + STATICS, 'statics')
src = sub1(src, '        let parts: Vec<Mass> = bounds',
           '        let probe = std::env::var_os("YUPEI_PROF").is_some();\n        let parts: Vec<Mass> = bounds', 'flag')

src = sub1(src, '                        let (mx_h, mx_g, sh_pos, sg_pos, sqg_pos, npos) = simd::fused_hg_stats(',
'''                        let _t_other = if probe { Some(std::time::Instant::now()) } else { None };
                        let (mx_h, mx_g, sh_pos, sg_pos, sqg_pos, npos) = simd::fused_hg_stats(''', 'other-head')
src = sub1(src, '''                            sh_pos > 0.0,
                            sg_pos > 0.0,
                        );''',
'''                            sh_pos > 0.0,
                            sg_pos > 0.0,
                        );
                        if let Some(t) = _t_other {
                            P2_OTHER_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }''', 'other-tail')

src = sub1(src, '                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----',
'''                        let _t_kh = if probe { Some(std::time::Instant::now()) } else { None };
                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----''', 'kh-head')
src = sub1(src, '''                                kh.select_nth_unstable(take_t - 1);
                                kh[..take_t].sort_unstable();''',
'''                                kh.select_nth_unstable(take_t - 1);
                                kh[..take_t].sort_unstable();
                                if let Some(t) = _t_kh {
                                    P2_KH_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                }
                                let _t_kg = if probe { Some(std::time::Instant::now()) } else { None };''', 'kh-tail')
src = sub1(src, '''                                kg.select_nth_unstable(take_t - 1);
                                kg[..take_t].sort_unstable();''',
'''                                kg.select_nth_unstable(take_t - 1);
                                kg[..take_t].sort_unstable();
                                if let Some(t) = _t_kg {
                                    P2_KG_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                }
                                let _t_abs = if probe { Some(std::time::Instant::now()) } else { None };''', 'kg-tail')
src = sub1(src, '''                                absorb_topk(t, n, 1, nlev_t, ks_ref, &gidx[..take_t], &hb, &gb, &pref);''',
'''                                absorb_topk(t, n, 1, nlev_t, ks_ref, &gidx[..take_t], &hb, &gb, &pref);
                                if let Some(t) = _t_abs {
                                    P2_ABS_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                                }''', 'abs-tail')

src = sub1(src, '''    eprintln!(
        "[net:{mat_name}] 有效配对 {npair_eff}（自能量非正 {n_bad}），扫描 {:.1}s",
        t0.elapsed().as_secs_f32()
    );''',
'''    eprintln!(
        "[net:{mat_name}] 有效配对 {npair_eff}（自能量非正 {n_bad}），扫描 {:.1}s",
        t0.elapsed().as_secs_f32()
    );
    if std::env::var_os("YUPEI_PROF").is_some() {
        let g = |v: &AtomicU64| v.load(Ordering::Relaxed) as f64 / 1e9;
        eprintln!(
            "[probe2:{mat_name}] 累计线程秒：非topk全段 {:.1}s，打包选kh {:.1}s，打包选kg {:.1}s，取前K累加 {:.1}s",
            g(&P2_OTHER_NS), g(&P2_KH_NS), g(&P2_KG_NS), g(&P2_ABS_NS)
        );
    }''', 'eprintln')

pathlib.Path('.probe2_net.rs').write_text(src)
print('probe2 写好')
