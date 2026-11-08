//! 沙箱专用：v1 anneal_core16 原样拷贝（src/anneal_regime_diff_metrics.rs 161~491 行），用于逐位对拍。
#![allow(dead_code)]
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const EPS: f32 = 1e-6;
const N_CORE: usize = 16;
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    #[inline]
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_DEAD_BEEF
            } else {
                seed
            },
        }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    #[inline]
    fn next_index(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        ((self.next_u64() >> 32).wrapping_mul(n as u64) >> 32) as usize
    }
}

// ============================================================================
// 核心退火计算（16 维归一化可还原性指标）
// ============================================================================

pub struct FastAnnealBuf {
    pub guess: Vec<f32>,
}

impl FastAnnealBuf {
    pub fn new() -> Self {
        Self {
            guess: Vec::with_capacity(4096),
        }
    }
}

/// 自适应退火步数：保证不同长度序列都能充分退火，同时通过除以 m_max 归一化消除步数尺度影响
#[inline]
pub fn adaptive_steps(n: usize, cap: usize) -> usize {
    (n.saturating_mul(n).saturating_mul(6)).max(600).min(cap)
}

/// 对任意实数序列运行确定性模拟退火（初始先验为升序），返回 16 个核心可还原性指标。
pub fn anneal_core16(seq: &[f32], m_max: usize, buf: &mut FastAnnealBuf) -> [f32; N_CORE] {
    anneal_core16_with_mode(seq, m_max, 0, buf)
}

/// 支持多种初始猜测先验模式：
/// mode=0: 升序 (Ascending)
/// mode=1: 降序 (Descending)
/// mode=2: U型 (两头大中间小)
/// mode=3: 倒U型 (中间大两头小)
pub fn anneal_core16_with_mode(
    seq: &[f32],
    m_max: usize,
    mode: u8,
    buf: &mut FastAnnealBuf,
) -> [f32; N_CORE] {
    let n = seq.len();
    if n < 3 || m_max < 10 {
        return [f32::NAN; N_CORE];
    }
    let mut sum = 0.0f64;
    for &v in seq {
        if !v.is_finite() {
            return [f32::NAN; N_CORE];
        }
        sum += v as f64;
    }
    let mean = sum / n as f64;
    let mut sq_sum = 0.0f64;
    for &v in seq {
        let d = v as f64 - mean;
        sq_sum += d * d;
    }
    let sigma2 = sq_sum / n as f64;
    if sigma2 <= 1e-20 {
        return [f32::NAN; N_CORE];
    }

    // 构造初始猜测序列
    buf.guess.clear();
    buf.guess.extend_from_slice(seq);
    buf.guess.sort_by(|a, b| a.total_cmp(b));
    if mode == 1 {
        // 降序先验
        buf.guess.reverse();
    } else if mode == 2 {
        // U型先验：大值放两端，小值放中间
        let sorted = buf.guess.clone();
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in sorted.iter().rev().enumerate() {
            if idx % 2 == 0 {
                buf.guess[left] = val;
                left += 1;
            } else {
                buf.guess[right] = val;
                right = right.saturating_sub(1);
            }
        }
    } else if mode == 3 {
        // 倒U型先验：小值放两端，大值放中间
        let sorted = buf.guess.clone();
        let mut left = 0usize;
        let mut right = n - 1;
        for (idx, &val) in sorted.iter().enumerate() {
            if idx % 2 == 0 {
                buf.guess[left] = val;
                left += 1;
            } else {
                buf.guess[right] = val;
                right = right.saturating_sub(1);
            }
        }
    }

    let mut s = 0.0f64;
    for k in 0..n {
        let d = buf.guess[k] as f64 - seq[k] as f64;
        s += d * d;
    }
    let denom = 2.0 * sigma2 * n as f64;
    let inv_denom = 1.0 / denom;
    let r0 = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
    let hl_target = (1.0 + r0) * 0.5;
    let s_tol = (sigma2 * n as f64 * 1e-10).max(1e-12);
    let inv_m = 1.0 / (m_max as f64 - 1.0).max(1.0);

    let mut rng = XorShift64::new(SEED ^ ((mode as u64) << 32));
    let mut prev_r = r0;
    let mut final_r = r0;
    let mut peak_r = r0;
    let mut max_dd = 0.0f64;

    let mut a1_step = m_max;
    let mut a2_step = m_max;
    let mut a3_step = m_max;

    let mut inertia_sum = 0.0f64;
    let mut declines = 0u32;
    let mut dr_cnt = 0u32;
    let mut dr_s1 = 0.0f64;
    let mut dr_s2 = 0.0f64;

    let s1_end = m_max / 3;
    let s2_end = 2 * (m_max / 3);
    let mut s1_cnt = 0u32;
    let mut s1_s1 = 0.0f64;
    let mut s1_s2 = 0.0f64;
    let mut s2_cnt = 0u32;
    let mut s2_decl = 0u32;
    let mut s3_cnt = 0u32;
    let mut s3_s1 = 0.0f64;
    let mut s3_s2 = 0.0f64;
    let mut r_at_s2 = r0;

    let mut d_cnt = 0u32;
    let mut d_pos = 0u32;
    let mut d_flips = 0u32;
    let mut prev_d_sign: i8 = 0;

    let mut actual_steps = 0usize;

    for t in 0..m_max {
        let i = rng.next_index(n);
        let mut j = rng.next_index(n);
        if j == i {
            j = (i + 1) % n;
        }

        if d_cnt < 2000 && buf.guess[i] != buf.guess[j] {
            let dv = seq[j] - seq[i];
            d_cnt += 1;
            if dv > 0.0 {
                d_pos += 1;
            }
            let cur_sign: i8 = if dv > 0.0 {
                1
            } else if dv < 0.0 {
                -1
            } else {
                0
            };
            if prev_d_sign != 0 && cur_sign != 0 && cur_sign != prev_d_sign {
                d_flips += 1;
            }
            if cur_sign != 0 {
                prev_d_sign = cur_sign;
            }
        }

        let gi = buf.guess[i];
        let gj = buf.guess[j];
        let ti = seq[i];
        let tj = seq[j];
        let ds = 2.0f32 * (gi - gj) * (ti - tj);
        let ct = sigma2 * (1.0 - t as f64 * inv_m).max(0.0);
        if (ds as f64) < 0.0 || (ds as f64) < ct {
            s += ds as f64;
            buf.guess.swap(i, j);
        }

        let cr = (1.0 - s * inv_denom).clamp(-1.0, 1.0);
        final_r = cr;
        actual_steps = t + 1;

        if t > 0 {
            let dv = cr - prev_r;
            dr_cnt += 1;
            dr_s1 += dv;
            dr_s2 += dv * dv;
            if cr < prev_r {
                declines += 1;
                if t >= s1_end && t < s2_end {
                    s2_decl += 1;
                }
            }
            if t < s1_end {
                s1_cnt += 1;
                s1_s1 += dv;
                s1_s2 += dv * dv;
            } else if t < s2_end {
                s2_cnt += 1;
            } else {
                s3_cnt += 1;
                s3_s1 += dv;
                s3_s2 += dv * dv;
            }
        }

        if a1_step == m_max && cr >= hl_target {
            a1_step = t;
        }
        if a2_step == m_max && cr >= 0.80 {
            a2_step = t;
        }
        if a3_step == m_max && cr >= 0.90 {
            a3_step = t;
        }
        inertia_sum += 1.0 - cr;
        if cr > peak_r {
            peak_r = cr;
        } else {
            let dd = peak_r - cr;
            if dd > max_dd {
                max_dd = dd;
            }
        }
        if t + 1 == s2_end {
            r_at_s2 = cr;
        }
        prev_r = cr;
        if s <= s_tol {
            if t + 1 < s2_end {
                r_at_s2 = cr;
            }
            break;
        }
    }

    let m_f = m_max as f32;
    let act_f = actual_steps.max(1) as f32;
    let r0_f = r0 as f32;
    let fin_f = final_r as f32;

    let dr_std = if dr_cnt >= 2 {
        let nd = dr_cnt as f64;
        ((dr_s2 - dr_s1 * dr_s1 / nd) / (nd - 1.0)).max(0.0).sqrt() as f32
    } else {
        0.0
    };
    let hot_std = if s1_cnt >= 2 {
        let nd = s1_cnt as f64;
        ((s1_s2 - s1_s1 * s1_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };
    let cold_std = if s3_cnt >= 2 {
        let nd = s3_cnt as f64;
        ((s3_s2 - s3_s1 * s3_s1 / nd) / (nd - 1.0)).max(0.0).sqrt()
    } else {
        0.0
    };

    let mut out = [f32::NAN; N_CORE];
    out[0] = r0_f;
    out[1] = fin_f;
    out[2] = fin_f - r0_f.abs();
    out[3] = (fin_f - r0_f) / (1.0 - r0_f + 1e-4);
    out[4] = a1_step as f32 / m_f;
    out[5] = a2_step as f32 / m_f;
    out[6] = a3_step as f32 / m_f;
    out[7] = (inertia_sum / act_f as f64) as f32;
    out[8] = declines as f32 / act_f;
    out[9] = max_dd as f32;
    out[10] = dr_std;
    out[11] = if s2_cnt > 0 {
        s2_decl as f32 / s2_cnt as f32
    } else {
        0.0
    };
    out[12] = (final_r - r_at_s2) as f32;
    out[13] = (hot_std / (cold_std + 1e-6)) as f32;
    out[14] = if d_cnt > 0 {
        d_pos as f32 / d_cnt as f32
    } else {
        0.5
    };
    out[15] = if d_cnt > 1 {
        d_flips as f32 / (d_cnt - 1) as f32
    } else {
        0.5
    };
    out
}
