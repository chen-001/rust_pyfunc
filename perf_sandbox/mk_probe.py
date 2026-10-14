import re, pathlib
src = pathlib.Path('.v5_net.rs').read_text()

def sub1(text, old, new, tag):
    assert text.count(old) == 1, (tag, text.count(old))
    return text.replace(old, new)

STATICS = '''
// [PROBE] 分段计时用（只在 YUPEI_PROF=1 时累加；正式路径不受影响）
use std::sync::atomic::{AtomicU64, Ordering};
static PROBE_FUSED_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_TOPK_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_IDX_NS: AtomicU64 = AtomicU64::new(0);
static PROBE_MASS_NS: AtomicU64 = AtomicU64::new(0);
'''
src = sub1(src, 'use super::topk::{absorb_topk, Acc as TopkAcc};',
           'use super::topk::{absorb_topk, Acc as TopkAcc};\n' + STATICS, 'statics')

src = sub1(src, '        let parts: Vec<Mass> = bounds',
           '        let probe = std::env::var_os("YUPEI_PROF").is_some();\n        let parts: Vec<Mass> = bounds', 'probe-flag')

src = sub1(src, '''                        let (mx_h, mx_g, sh_pos, sg_pos, sqg_pos, npos) = simd::fused_hg_stats(
                            &mut hb, &mut gb, wb, wc, p, q, &addm, b, c,
                        );''',
'''                        let _t_fused = if probe { Some(std::time::Instant::now()) } else { None };
                        let (mx_h, mx_g, sh_pos, sg_pos, sqg_pos, npos) = simd::fused_hg_stats(
                            &mut hb, &mut gb, wb, wc, p, q, &addm, b, c,
                        );
                        if let Some(t) = _t_fused {
                            PROBE_FUSED_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }''', 'fused')

src = sub1(src, '                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----',
           '''                        let _t_topk = if probe { Some(std::time::Instant::now()) } else { None };
                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----''', 'topk-head')

src = sub1(src, '''                                t.pairs += 1;
                            }
                        }
                        if mx_h == f32::NEG_INFINITY {''',
'''                                t.pairs += 1;
                            }
                        }
                        if let Some(t) = _t_topk {
                            PROBE_TOPK_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }
                        if mx_h == f32::NEG_INFINITY {''', 'topk-tail')

src = sub1(src, '                        let (bi, bsi) = simd::argmax_first_two(&hb, &gb, mx_h, mx_g);',
'''                        let _t_idx = if probe { Some(std::time::Instant::now()) } else { None };
                        let (bi, bsi) = simd::argmax_first_two(&hb, &gb, mx_h, mx_g);
                        if let Some(t) = _t_idx {
                            PROBE_IDX_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }''', 'idx')

src = sub1(src, '''                        simd::mass_acc_hg(
                            &mut em, &mut gm, &mut g2m,
                            &mut sm, &mut sgm, &mut sg2m,
                            &hb, &gb,
                            if sh_pos > 0.0 { 1.0 / sh_pos } else { 1.0 },
                            if sg_pos > 0.0 { 1.0 / sg_pos } else { 1.0 },
                            sh_pos > 0.0,
                            sg_pos > 0.0,
                        );''',
'''                        let _t_mass = if probe { Some(std::time::Instant::now()) } else { None };
                        simd::mass_acc_hg(
                            &mut em, &mut gm, &mut g2m,
                            &mut sm, &mut sgm, &mut sg2m,
                            &hb, &gb,
                            if sh_pos > 0.0 { 1.0 / sh_pos } else { 1.0 },
                            if sg_pos > 0.0 { 1.0 / sg_pos } else { 1.0 },
                            sh_pos > 0.0,
                            sg_pos > 0.0,
                        );
                        if let Some(t) = _t_mass {
                            PROBE_MASS_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        }''', 'mass')

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
            "[probe:{mat_name}] 累计线程秒：合算h/g+统计 {:.1}s，topk块 {:.1}s，取下标 {:.1}s，矩累加 {:.1}s",
            g(&PROBE_FUSED_NS), g(&PROBE_TOPK_NS), g(&PROBE_IDX_NS), g(&PROBE_MASS_NS)
        );
    }''', 'eprintln')

pathlib.Path('.probe_net.rs').write_text(src)
print('probe 文件写好，行数', len(src.splitlines()))
