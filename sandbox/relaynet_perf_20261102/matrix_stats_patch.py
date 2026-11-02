"""Parallelize independent matrix rows without changing sort/sum order."""
from pathlib import Path
p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
a = s.index("fn matrix_stats(")
b = s.index("\nconst MATRIX_STAT_NAMES", a)
s = s[:a] + '''fn matrix_stats(s: &Array2<f64>) -> Vec<Vec<f64>> {
    let n = s.nrows();
    let rows: Vec<[f64; 7]> = (0..n).into_par_iter().map_init(
        || Vec::<f64>::with_capacity(n),
        |v, i| {
            v.clear();
            v.extend(s.row(i).iter().enumerate().filter_map(|(j, &x)| {
                if j != i && x.is_finite() { Some(x) } else { None }
            }));
            if v.is_empty() { return [f64::NAN; 7]; }
            v.sort_by(|a, b| b.total_cmp(a));
            let t3 = v.len().min(3);
            let t10 = v.len().min(10);
            [v[0], v[..t3].iter().sum::<f64>() / t3 as f64,
             v[..t10].iter().sum::<f64>() / t10 as f64,
             v.iter().sum::<f64>() / v.len() as f64,
             v[(v.len() as f64 * 0.1) as usize],
             v.iter().filter(|&&x| x > 0.2).count() as f64 / v.len() as f64,
             v.iter().filter(|&&x| x > 0.3).count() as f64 / v.len() as f64]
        },
    ).collect();
    (0..7).map(|k| rows.iter().map(|r| r[k]).collect()).collect()
}
''' + s[b:]
p.write_text(s)
