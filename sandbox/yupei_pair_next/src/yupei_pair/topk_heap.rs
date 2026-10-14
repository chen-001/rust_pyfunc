pub type Entry = (f32, i32, i32);
// A worse entry is the smaller score, or the later pair among equal scores.
// This matches removing the LAST minimum from the old chronological list.
#[inline(always)]
fn worse(a: Entry, b: Entry) -> bool {
    a.0 < b.0 || (a.0 == b.0 && (a.1 > b.1 || (a.1 == b.1 && a.2 > b.2)))
}
pub struct FlatTop {
    pub data: Vec<Entry>,
    pub lens: Vec<usize>,
    pub cap: usize,
}
impl FlatTop {
    pub fn new(n: usize, cap: usize) -> Self {
        Self {
            data: vec![(0.0, 0, 0); n * cap],
            lens: vec![0; n],
            cap,
        }
    }
    pub fn threshold(&self, x: usize) -> f32 {
        if self.lens[x] < self.cap {
            f32::NEG_INFINITY
        } else {
            self.data[x * self.cap].0
        }
    }
    pub fn push(&mut self, x: usize, e: Entry) {
        let base = x * self.cap;
        let len = self.lens[x];
        let a = &mut self.data[base..base + self.cap];
        if len < self.cap {
            a[len] = e;
            self.lens[x] += 1;
            let mut p = len;
            while p > 0 {
                let q = (p - 1) / 2;
                if !worse(a[p], a[q]) {
                    break;
                }
                a.swap(p, q);
                p = q;
            }
        } else {
            a[0] = e;
            let mut p = 0;
            loop {
                let l = p * 2 + 1;
                if l >= len {
                    break;
                }
                let r = l + 1;
                let q = if r < len && worse(a[r], a[l]) { r } else { l };
                if !worse(a[q], a[p]) {
                    break;
                }
                a.swap(q, p);
                p = q;
            }
        }
    }
    pub fn row(&self, x: usize) -> &[Entry] {
        &self.data[x * self.cap..x * self.cap + self.lens[x]]
    }
    pub fn sort_rows(&mut self) {
        for x in 0..self.lens.len() {
            self.data[x * self.cap..x * self.cap + self.lens[x]].sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0)
                    .unwrap()
                    .then(a.1.cmp(&b.1))
                    .then(a.2.cmp(&b.2))
            });
        }
    }
}
