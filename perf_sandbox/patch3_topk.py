p='/home/chenzongwei/rust_pyfunc/src/yupei_pair/topk.rs'
s=open(p).read(); done=[]
def rep(old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep("const NBLOCKS: usize = 512;", "pub(crate) const NBLOCKS: usize = 512;", 'nblocks pub')
rep("""struct Acc {
    // [口径2][档nlev][统计4? 分开存] —— 分四个平面存，索引 = (crit*nlev + l)*n + y
    cnt: Vec<u32>,
    sum_gap: Vec<f64>,
    sum_h: Vec<f64>,
    sum_mass: Vec<f64>,
    pairs: u64,
    skipped: u64,
}""",
"""pub(crate) struct Acc {
    // [口径2][档nlev][统计4? 分开存] —— 分四个平面存，索引 = (crit*nlev + l)*n + y
    pub(crate) cnt: Vec<u32>,
    pub(crate) sum_gap: Vec<f64>,
    pub(crate) sum_h: Vec<f64>,
    pub(crate) sum_mass: Vec<f64>,
    pub(crate) pairs: u64,
    pub(crate) skipped: u64,
}""", 'acc pub')
rep("impl Acc {\n    fn new(", "impl Acc {\n    pub(crate) fn new(", 'acc new pub')
rep("fn absorb_topk(", "pub(crate) fn absorb_topk(", 'absorb pub')
# finalize_total：把 topk_one_mat 尾部抽出来
rep("""    let mut total = Acc::new(n, nlev);
    for a in accs.iter() {
        total.add(a);
    }
    let pairs = total.pairs;

    // names + cols：每口径四量，顺序 (crit × l × 4)""",
"""    let mut total = Acc::new(n, nlev);
    for a in accs.iter() {
        total.add(a);
    }
    let (names, cols) = finalize_total(&total, n, mask, ks, pre);
    eprintln!(
        "[topk] {nlev} 档 × 2 口径 {:.1}s，pairs={}",
        t0.elapsed().as_secs_f64(),
        total.pairs
    );
    return (names, cols);
}

/// 把累计器收成 (names, cols)。与 topk_one_mat 原有的尾部逐字一致。
pub(crate) fn finalize_total(
    total: &Acc,
    n: usize,
    mask: &[bool],
    ks: &[usize],
    pre: &str,
) -> (Vec<String>, Vec<Vec<f32>>) {
    let nlev = ks.len();
    let pairs = total.pairs;
    // names + cols：每口径四量，顺序 (crit × l × 4)""", 'finalize extract')
rep("""    eprintln!(
        "[topk] {nlev} 档 × 2 口径 {:.1}s，pairs={pairs}",
        t0.elapsed().as_secs_f64()
    );
    (names, cols)
}""", """    (names, cols)
}""", 'finalize tail')
open(p,'w').write(s)
print('\n'.join(done))
