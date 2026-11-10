//! Exact order statistics for EXZ's finite (p, q) values.
//!
//! Only the requested ranks are partitioned. No arithmetic or approximation is
//! involved, and downstream accumulation keeps its original traversal order.

/// Return the same bin cut values as sorting and indexing k * len / bins.
/// A negative zero requests the caller's original sort: partial_cmp considers
/// both zero signs equal, and only that old sort can preserve its chosen bits.
pub(crate) fn exact_quantile_cuts(values: &mut [f64], bins: usize) -> Option<Vec<f64>> {
    if bins <= 1 || values.is_empty() {
        return Some(Vec::new());
    }
    if values.iter().any(|v| v.to_bits() == (-0.0f64).to_bits()) {
        return None;
    }
    let n = values.len();
    let ranks: Vec<usize> = (1..bins).map(|k| (k * n / bins).min(n - 1)).collect();
    let mut cuts = vec![0.0; ranks.len()];
    select_ranks(values, 0, &ranks, &mut cuts);
    Some(cuts)
}

fn select_ranks(values: &mut [f64], offset: usize, ranks: &[usize], out: &mut [f64]) {
    if ranks.is_empty() {
        return;
    }
    let rank = ranks[ranks.len() / 2];
    let (lower, value, upper) =
        values.select_nth_unstable_by(rank - offset, |a, b| a.partial_cmp(b).unwrap());
    // bins may exceed the number of observations; fill duplicate ranks once.
    let first = ranks.partition_point(|&r| r < rank);
    let after = ranks.partition_point(|&r| r <= rank);
    out[first..after].fill(*value);
    select_ranks(lower, offset, &ranks[..first], &mut out[..first]);
    select_ranks(upper, rank + 1, &ranks[after..], &mut out[after..]);
}
