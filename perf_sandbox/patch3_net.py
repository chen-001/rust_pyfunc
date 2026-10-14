p='/home/chenzongwei/rust_pyfunc/src/yupei_pair/net.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)

rep("use super::simd;", "use super::simd;\nuse super::topk::{absorb_topk, Acc as TopkAcc};", 'imports')

rep("""    #[allow(dead_code)]
    pub strata_s: Vec<(f64, f64, usize)>,
}""",
"""    #[allow(dead_code)]
    pub strata_s: Vec<(f64, f64, usize)>,
    /// 与 topk 合并后，同一趟配对扫描里累计出来的 topk 结果（None = 没启用合并）。
    pub topk_total: Option<TopkAcc>,
}""", 'netmats field')

rep("""    null_mc: usize,
    exz_bins: usize,
    mat_name: &str,
) -> NetMats {""",
"""    null_mc: usize,
    exz_bins: usize,
    mat_name: &str,
    topk_ks: Option<&[usize]>,
) -> NetMats {""", 'net sig')

rep("        type Mass = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64);",
    "        type Mass = (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, u64, Vec<TopkAcc>);", 'mass type')

rep("""        let n_stocks = mask.iter().filter(|&&v| v).count();
        let n_valid_pair = (n_stocks as u32).saturating_sub(2);""",
"""        let n_stocks = mask.iter().filter(|&&v| v).count();
        let n_valid_pair = (n_stocks as u32).saturating_sub(2);
        // ---- topk 合并到这一趟扫描：参数与 topk_one_mat 完全一致 ----
        let topk_on = topk_ks.is_some();
        let ks_ref: &[usize] = topk_ks.unwrap_or(&[]);
        let nlev_t = ks_ref.len();
        let maxk_t = ks_ref.iter().copied().max().unwrap_or(0);
        let take_t = if topk_on {
            n_stocks.saturating_sub(2).min(maxk_t)
        } else {
            0
        };
        let mut topk_total: Option<TopkAcc> = if topk_on {
            Some(TopkAcc::new(n, nlev_t))
        } else {
            None
        };""", 'topk setup')

rep("""                let mut hb = vec![0.0f32; n];
                let mut gb = vec![0.0f32; n];
                for b in b0..b1 {""",
"""                let mut hb = vec![0.0f32; n];
                let mut gb = vec![0.0f32; n];
                // topk 的块内缓冲（热路径上不再分配）
                let mut kh: Vec<u64> = vec![0u64; n];
                let mut kg: Vec<u64> = vec![0u64; n];
                let mut idx: Vec<u32> = vec![0u32; maxk_t];
                let mut gidx: Vec<u32> = vec![0u32; maxk_t];
                let mut pref: Vec<f64> = vec![0.0f64; maxk_t];
                // 子块 = topk_one_mat 的 NBLOCKS 分区与 [b0,b1) 的交（512 是 256 的两倍细分，
                // 所以每个 256 块正好被切成两段），保证两边各自的累加分组不变。
                let mut subs: Vec<(usize, usize)> = Vec::new();
                if topk_on {
                    let nbt = super::topk::NBLOCKS.min(n).max(1);
                    for j in 0..nbt {
                        let s0 = (j * n / nbt).max(b0);
                        let s1 = ((j + 1) * n / nbt).min(b1);
                        if s0 < s1 {
                            subs.push((s0, s1));
                        }
                    }
                } else {
                    subs.push((b0, b1));
                }
                let mut tks: Vec<TopkAcc> = Vec::with_capacity(subs.len());
                for (s0, s1) in subs.into_iter() {
                    let mut tacc: Option<TopkAcc> =
                        if topk_on { Some(TopkAcc::new(n, nlev_t)) } else { None };
                    for b in s0..s1 {""", 'closure head')

rep("""                        hb[b] = f32::NEG_INFINITY;
                        hb[c] = f32::NEG_INFINITY;
                        gb[b] = f32::NEG_INFINITY;
                        gb[c] = f32::NEG_INFINITY;
                        let (bi, h) = simd::argmax_first(&hb);""",
"""                        hb[b] = f32::NEG_INFINITY;
                        hb[c] = f32::NEG_INFINITY;
                        gb[b] = f32::NEG_INFINITY;
                        gb[c] = f32::NEG_INFINITY;
                        // ---- topk 合并段：h/gap 只算这一遍，按 topk 原来的顺序累计 ----
                        if let Some(t) = tacc.as_mut() {
                            if take_t > 0 {
                                for x in 0..n {
                                    kh[x] = simd::pack_desc_idx_asc(hb[x], x as u32);
                                }
                                kh.select_nth_unstable(take_t - 1);
                                kh[..take_t].sort_unstable();
                                for x in 0..n {
                                    kg[x] = simd::pack_desc_idx_asc(gb[x], x as u32);
                                }
                                kg.select_nth_unstable(take_t - 1);
                                kg[..take_t].sort_unstable();
                                for v in 0..take_t {
                                    idx[v] = kh[v] as u32;
                                    gidx[v] = kg[v] as u32;
                                }
                                let mut sf = 0.0f64;
                                for v in 0..take_t {
                                    let y = idx[v] as usize;
                                    sf += hb[y].max(0.0) as f64;
                                    pref[v] = sf;
                                }
                                absorb_topk(t, n, 0, nlev_t, ks_ref, &idx[..take_t], &hb, &gb, &pref);
                                sf = 0.0;
                                for v in 0..take_t {
                                    let y = gidx[v] as usize;
                                    sf += gb[y].max(0.0) as f64;
                                    pref[v] = sf;
                                }
                                absorb_topk(t, n, 1, nlev_t, ks_ref, &gidx[..take_t], &hb, &gb, &pref);
                                t.pairs += 1;
                            }
                        }
                        let (bi, h) = simd::argmax_first(&hb);""", 'pair topk block')

rep("""                (em, gm, g2m, sm, sgm, sg2m, npair)
            })
            .collect();""",
"""                    if let Some(a) = tacc {
                        tks.push(a);
                    }
                }
                (em, gm, g2m, sm, sgm, sg2m, npair, tks)
            })
            .collect();""", 'closure tail')

rep("""        for (em, gm, g2m, sm, sgm, sg2m, np) in parts.iter() {""",
"""        for (em, gm, g2m, sm, sgm, sg2m, np, tks) in parts.iter() {
            if let Some(t) = topk_total.as_mut() {
                for a in tks.iter() {
                    t.add(a);
                }
            }""", 'merge tks')

rep("""        mass: mass_vectors,
        npair: npair_eff,""",
"""        mass: mass_vectors,
        topk_total,
        npair: npair_eff,""", 'netmats init')
open(p,'w').write(s)
print('\n'.join(done))
