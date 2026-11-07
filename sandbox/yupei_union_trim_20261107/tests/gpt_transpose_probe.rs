//! Independent layout experiment; production GPT code does not call this.
//! rustc --edition=2021 -O -C target-cpu=native tests/gpt_transpose_probe.rs -o /tmp/gpt_transpose_probe
//! Checks exact Pair copies, then times only allocation/Pair generation/writes.
use std::hint::black_box;
use std::time::Instant;

#[derive(Clone, Copy)]
struct Pair { g: f32, gs: f32, gnpos: f32, gcov: f32, self_assoc: f32 }
impl Pair {
    fn missing() -> Self { Self { g: f32::NAN, gs: f32::NAN, gnpos: f32::NAN,
        gcov: f32::NAN, self_assoc: f32::NAN } }
    fn bits(self) -> [u32; 5] { [self.g.to_bits(), self.gs.to_bits(),
        self.gnpos.to_bits(), self.gcov.to_bits(), self.self_assoc.to_bits()] }
}
fn valid(i: usize, j: usize, symmetric: bool, mask: usize) -> bool {
    i != j && match mask {
        0 => true,
        1 => i % 7 != 0 && if symmetric { j % 7 != 0 } else { j % 5 != 0 },
        2 => false,
        _ => (i + j) % 3 == 0,
    }
}
fn pair(i: usize, j: usize) -> (Pair, f32) {
    let a = (i.wrapping_mul(196613) ^ j.wrapping_mul(251)) as u32;
    let b = (j.wrapping_mul(65537) ^ i.wrapping_mul(127)) as u32;
    // Bit construction preserves signed zeros and nontrivial mantissas.
    let p = Pair { g: f32::from_bits(0x3f00_0000 | a & 0x007f_ffff),
        gs: f32::from_bits(0xbf00_0000 | b & 0x007f_ffff),
        gnpos: if a % 2 == 0 { 0. } else { -0. },
        gcov: (a % 100) as f32 / 100., self_assoc: (i % 1000) as f32 / 1000. };
    (p, (j % 1000) as f32 / 1000.)
}
struct Output { fwd: Vec<Pair>, rev: Option<Vec<Pair>>, allocation: f64,
    scan: f64, transpose: f64, total: f64 }

fn run(n: usize, symmetric: bool, mask: usize, staged: bool) -> Output {
    let total = Instant::now();
    let mut fwd = vec![Pair::missing(); n * n];
    let mut rev = if symmetric { None } else { Some(vec![Pair::missing(); n * n]) };
    let mut reverse_assoc = if staged { vec![f32::NAN; n * n] } else { Vec::new() };
    let allocation = total.elapsed().as_secs_f64();
    let scan = Instant::now();
    for i in 0..n {
        for j in if symmetric { i + 1 } else { 0 }..n {
            if !valid(i, j, symmetric, mask) { continue; }
            let (p, q) = pair(i, j);
            fwd[i * n + j] = p;
            if staged { reverse_assoc[i * n + j] = q; }
            else {
                let reverse = Pair { self_assoc: q, ..p };
                if let Some(r) = &mut rev { r[j * n + i] = reverse; }
                else { fwd[j * n + i] = reverse; }
            }
        }
    }
    let scan = scan.elapsed().as_secs_f64();
    let transpose = Instant::now();
    if staged {
        const TILE: usize = 32;
        for ii in (0..n).step_by(TILE) {
            for jj in (if symmetric { ii } else { 0 }..n).step_by(TILE) {
                for j in jj..(jj + TILE).min(n) {
                    for i in ii..(ii + TILE).min(n) {
                        if symmetric && i >= j { continue; }
                        let p = fwd[i * n + j];
                        let reverse = Pair { self_assoc: reverse_assoc[i * n + j], ..p };
                        if let Some(r) = &mut rev { r[j * n + i] = reverse; }
                        else { fwd[j * n + i] = reverse; }
                    }
                }
            }
        }
    }
    let transpose = transpose.elapsed().as_secs_f64();
    let total = total.elapsed().as_secs_f64();
    Output { fwd, rev, allocation, scan, transpose, total }
}
fn verify() -> usize {
    assert_eq!(std::mem::size_of::<Pair>(), 20);
    let mut cases = 0;
    for n in [0, 1, 2, 7, 17, 32, 33, 64, 257] {
        for symmetric in [true, false] {
            for mask in 0..4 {
                let a = run(n, symmetric, mask, false);
                let b = run(n, symmetric, mask, true);
                for (x, y) in a.fwd.iter().zip(&b.fwd) { assert_eq!(x.bits(), y.bits()); }
                if let (Some(a), Some(b)) = (&a.rev, &b.rev) {
                    for (x, y) in a.iter().zip(b) { assert_eq!(x.bits(), y.bits()); }
                }
                cases += 1;
            }
        }
    }
    cases
}
fn main() {
    let cases = verify();
    let n = 2048;
    let mut rows = Vec::new();
    for symmetric in [true, false] {
        for staged in [false, true, false, true] {
            let out = run(n, symmetric, 1, staged);
            black_box(&out.fwd);
            black_box(&out.rev);
            rows.push(format!("{{\"n\":{n},\"symmetric\":{symmetric},\"staged\":{staged},\"allocation_s\":{:.9},\"scan_s\":{:.9},\"transpose_s\":{:.9},\"total_s\":{:.9}}}",
                out.allocation, out.scan, out.transpose, out.total));
        }
    }
    println!("{{\"scope\":\"single_core_pair_write_layout_only_no_score_calculation\",\"exact_cases\":{cases},\"all_pair_fields_bits_equal\":true,\"runs\":[{}]}}", rows.join(","));
}
