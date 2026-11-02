"""Optional exact per-node basic statistics and triangle order-statistic selection."""
from pathlib import Path

p = Path(__file__).parent / "src/relaynet_fast.rs"
s = p.read_text()
marker = "// BEGIN EXACT SMALL STATS"
if marker in s:
    raise SystemExit("small stats patch already applied")
start = s.index("fn stat_undir(")
lo = s.index("    let mut deg = vec![0.0f64; n];", start)
hi = s.index("    cols.push(entsim);", lo) + len("    cols.push(entsim);")
old = s[lo:hi]
s = s[:lo] + '''    let select_triangle = std::env::var_os("RELAYNET_TRI_SELECT").is_some();
    if std::env::var_os("RELAYNET_BASIC_PARALLEL").is_some() {
        cols.extend(undir_basic_parallel_exact(g));
    } else {
''' + old + "\n    }" + s[hi:]
old_tri = '''                triw.sort_by(|x, y| x.total_cmp(y));
                let pos = 0.75 * (triw.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(triw.len() - 1);
                o[19] = triw[lo] + (triw[hi] - triw[lo]) * (pos - lo as f64);'''
assert old_tri in s
s = s.replace(old_tri, '''                if select_triangle { o[19] = triangle_p75_select_exact(&mut triw); }
                else {
''' + old_tri + "\n                }", 1)
s += r'''
// BEGIN EXACT SMALL STATS
fn undir_basic_parallel_exact(g: &UGraph) -> Vec<Vec<f64>> {
    let rows: Vec<[f64; 6]> = (0..g.n).into_par_iter().map_init(Vec::<f64>::new, |sims, i| {
        let a = &g.adj[i];
        let mut row = [a.len() as f64, a.iter().map(|&(_, x)| x as f64).sum(),
            f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        if !a.is_empty() {
            sims.clear(); sims.extend(a.iter().map(|&(_, x)| x as f64));
            sims.sort_by(|x, y| y.total_cmp(x));
            row[2] = sims[0];
            row[3] = sims.iter().take(3).sum::<f64>() / sims.len().min(3) as f64;
            row[4] = row[1] / a.len() as f64;
            // Keep this sorted sum distinct from the adjacency-order strength.
            let ss: f64 = sims.iter().sum();
            if ss > 0.0 {
                row[5] = -sims.iter().map(|&p| {
                    let q = p / ss;
                    if q > 0.0 { q * q.ln() } else { 0.0 }
                }).sum::<f64>();
            }
        }
        row
    }).collect();
    (0..6).map(|k| rows.iter().map(|row| row[k]).collect()).collect()
}

fn triangle_p75_select_exact(values: &mut [f64]) -> f64 {
    let pos = 0.75 * (values.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    let (left, upper, _) = values.select_nth_unstable_by(hi, |x, y| x.total_cmp(y));
    let high = *upper;
    let low = if lo == hi { high } else { *left.select_nth_unstable_by(lo, |x, y| x.total_cmp(y)).1 };
    low + (high - low) * (pos - lo as f64)
}
// END EXACT SMALL STATS
'''
p.write_text(s)
print("Applied optional exact basic rows and triangle quantile selection")
