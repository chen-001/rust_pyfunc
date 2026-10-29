//! Signed temporal leadership against market/industry event flow.
//!
//! This is O(number of events + number of groups * trading-time buckets), not
//! another stock-pair kernel. A group's forward/backward exponential fields are
//! built once and queried at its member stocks' event buckets. Own-stock flow is
//! subtracted from both the numerator and the target energy. No lunch crossing,
//! future-day information, or price information is used.
use crate::yupei_gpt::model::{FactorColumn, Stock};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::{self, Error, ErrorKind};

const AM: usize = 720_000;
const PM: usize = 702_000;
const LENGTH: usize = AM + PM;
const SEGMENTS: [(usize, usize); 2] = [(0, AM), (AM, LENGTH)];
const DIRECTIONS: [&str; 4] = ["buy_buy", "buy_sell", "sell_buy", "sell_sell"];

fn bad(message: impl Into<String>) -> io::Error {
    Error::new(ErrorKind::InvalidData, message.into())
}
fn index(slot: u32) -> Option<usize> {
    match slot {
        0..720_000 => Some(slot as usize),
        1_260_000..1_962_000 => Some(AM + (slot - 1_260_000) as usize),
        _ => None,
    }
}
#[derive(Clone, Copy, Default)]
struct Own {
    energy: [[f64; 2]; 2],
    mass: [[f64; 2]; 2],
    // Own buy-before-sell minus own sell-before-buy, one value per session.
    cross_asym: [f64; 2],
}
struct Field {
    // [past buy, past sell, future buy, future sell], same-bucket half weights.
    values: Vec<[f64; 4]>,
    energy: [[f64; 2]; 2],
    mass: [[f64; 2]; 2],
}
#[derive(Clone, Copy)]
struct Scores {
    full: [f64; 4],
    am: [f64; 4],
    pm: [f64; 4],
}

fn validate(stocks: &[Stock]) -> io::Result<()> {
    if stocks.is_empty() {
        return Err(bad("leadlag requires at least one stock"));
    }
    for stock in stocks {
        let mut previous = None;
        for cell in &stock.cells {
            if index(cell.slot).is_none()
                || previous.is_some_and(|slot| cell.slot < slot)
                || !cell.buy.is_finite()
                || !cell.sell.is_finite()
                || cell.buy < 0.0
                || cell.sell < 0.0
            {
                return Err(bad(format!(
                    "{}: invalid/sorted-session event weights for leadlag",
                    stock.code
                )));
            }
            previous = Some(cell.slot);
        }
    }
    Ok(())
}

fn density(stocks: &[Stock], members: &[usize]) -> Vec<[f64; 2]> {
    let mut result = vec![[0.0; 2]; LENGTH];
    // members are code-sorted, so permutation of the input stock array cannot
    // change floating-point accumulation order in the common target flow.
    for &i in members {
        for cell in &stocks[i].cells {
            let t = index(cell.slot).expect("validated slots");
            result[t][0] += cell.buy as f64;
            result[t][1] += cell.sell as f64;
        }
    }
    result
}

fn make_field(flow: &[[f64; 2]], tau: f64) -> io::Result<Field> {
    let mut result = Field {
        values: vec![[0.0; 4]; LENGTH],
        energy: [[0.0; 2]; 2],
        mass: [[0.0; 2]; 2],
    };
    let decay = (-0.01 / tau).exp();
    for (session, &(lo, hi)) in SEGMENTS.iter().enumerate() {
        let mut history = [0.0; 2];
        for t in lo..hi {
            for side in 0..2 {
                let value = flow[t][side];
                let past = history[side] + 0.5 * value;
                result.values[t][side] = past;
                result.energy[session][side] += 2.0 * value * past;
                result.mass[session][side] += value;
                let next = (history[side] + value) * decay;
                history[side] = if next >= f64::MIN_POSITIVE { next } else { 0.0 };
            }
        }
        history.fill(0.0);
        for t in (lo..hi).rev() {
            for side in 0..2 {
                let value = flow[t][side];
                result.values[t][2 + side] = history[side] + 0.5 * value;
                let next = (history[side] + value) * decay;
                history[side] = if next >= f64::MIN_POSITIVE { next } else { 0.0 };
            }
        }
    }
    if result.energy.iter().flatten().any(|v| !v.is_finite()) {
        return Err(bad("leadlag target energy overflow"));
    }
    Ok(result)
}

fn own_statistics(stock: &Stock, decay_by_gap: &[f64]) -> Own {
    let mut result = Own::default();
    let mut history = [0.0; 2];
    let mut last_slot = 0;
    let mut last_session = 2;
    let mut cursor = 0;
    while cursor < stock.cells.len() {
        let slot = stock.cells[cursor].slot;
        let session = usize::from(slot >= 1_260_000);
        let mut weights = [0.0; 2];
        // Merge repeated buckets before assigning half of the synchronous term.
        while cursor < stock.cells.len() && stock.cells[cursor].slot == slot {
            weights[0] += stock.cells[cursor].buy as f64;
            weights[1] += stock.cells[cursor].sell as f64;
            cursor += 1;
        }
        if session != last_session {
            history.fill(0.0);
        } else {
            let d = decay_by_gap[(slot - last_slot) as usize];
            history[0] *= d;
            history[1] *= d;
        }
        result.cross_asym[session] += weights[1] * history[0] - weights[0] * history[1];
        for side in 0..2 {
            result.energy[session][side] +=
                2.0 * weights[side] * (history[side] + 0.5 * weights[side]);
            result.mass[session][side] += weights[side];
            history[side] += weights[side];
        }
        last_slot = slot;
        last_session = session;
    }
    result
}

#[inline]
fn normalized(value: f64, own_energy: f64, other_energy: f64) -> f64 {
    if own_energy > 0.0 && other_energy > 0.0 {
        value / (own_energy * other_energy).sqrt()
    } else {
        f64::NAN
    }
}

fn query(stock: &Stock, own: &Own, field: &Field, includes_self: bool) -> io::Result<Scores> {
    let mut asym = [[0.0; 4]; 2];
    let mut same_side_inner = [[0.0; 2]; 2];
    for cell in &stock.cells {
        let t = index(cell.slot).expect("validated slots");
        let session = usize::from(t >= AM);
        let f = field.values[t];
        let weights = [cell.buy as f64, cell.sell as f64];
        for a in 0..2 {
            for b in 0..2 {
                // Positive means this stock's a events precede the group's b events.
                asym[session][a * 2 + b] += weights[a] * (f[b + 2] - f[b]);
            }
            if includes_self {
                same_side_inner[session][a] += weights[a] * (f[a] + f[a + 2]);
            }
        }
    }
    let mut remaining = field.energy;
    if includes_self {
        for session in 0..2 {
            asym[session][1] -= own.cross_asym[session];
            asym[session][2] += own.cross_asym[session];
            for b in 0..2 {
                let scale = field.energy[session][b].max(own.energy[session][b]);
                let remainder = field.energy[session][b] + own.energy[session][b]
                    - 2.0 * same_side_inner[session][b];
                let mass_scale = field.mass[session][b].max(own.mass[session][b]);
                if field.mass[session][b] - own.mass[session][b] <= mass_scale * 1e-12 {
                    remaining[session][b] = 0.0;
                } else if remainder < -scale * 1e-10 {
                    return Err(bad(format!(
                        "{}: negative target energy after self exclusion",
                        stock.code
                    )));
                } else {
                    // Cancellation at machine precision is not evidence of a valid
                    // target vector. Leave the corresponding normalized value undefined.
                    remaining[session][b] = if remainder > scale * 1e-12 {
                        remainder
                    } else {
                        0.0
                    };
                }
            }
        }
    }
    let mut scores = Scores {
        full: [f64::NAN; 4],
        am: [f64::NAN; 4],
        pm: [f64::NAN; 4],
    };
    for a in 0..2 {
        for b in 0..2 {
            let k = a * 2 + b;
            scores.am[k] = normalized(asym[0][k], own.energy[0][a], remaining[0][b]);
            scores.pm[k] = normalized(asym[1][k], own.energy[1][a], remaining[1][b]);
            scores.full[k] = normalized(
                asym[0][k] + asym[1][k],
                own.energy[0][a] + own.energy[1][a],
                remaining[0][b] + remaining[1][b],
            );
        }
    }
    Ok(scores)
}

fn evaluate(
    stocks: &[Stock],
    members: &[usize],
    own: &[Own],
    field: &Field,
    includes_self: bool,
) -> io::Result<Vec<(usize, Scores)>> {
    members
        .par_iter()
        .map(|&i| query(&stocks[i], &own[i], field, includes_self).map(|s| (i, s)))
        .collect()
}

/// 36 factors. Target flows exclude the current stock. Unknown industries do
/// not become a fictitious peer group and are excluded from cross-industry flow.
/// Columns are signed leadership scores, not Pearson correlations or returns.
pub fn leadlag_columns(stocks: &[Stock]) -> io::Result<Vec<FactorColumn>> {
    validate(stocks)?;
    let n = stocks.len();
    let mut members: Vec<usize> = (0..n).collect();
    members.sort_by(|&i, &j| stocks[i].code.cmp(&stocks[j].code));
    let market = density(stocks, &members);
    let mut columns = Vec::with_capacity(36);
    let mut own_one_second = Vec::new();
    for (tau, label) in [(0.1, "t0p1"), (1.0, "t1"), (5.0, "t5")] {
        let decay_by_gap: Vec<f64> = (0..AM)
            .map(|gap| (-(gap as f64) * 0.01 / tau).exp())
            .collect();
        let own: Vec<Own> = stocks
            .par_iter()
            .map(|s| own_statistics(s, &decay_by_gap))
            .collect();
        let field = make_field(&market, tau)?;
        let scores = evaluate(stocks, &members, &own, &field, true)?;
        for (k, direction) in DIRECTIONS.iter().enumerate() {
            let mut full = vec![f64::NAN; n];
            for &(i, s) in &scores {
                full[i] = s.full[k];
            }
            columns.push(FactorColumn::new(
                format!("yr_leadlag_market_{direction}_{label}"),
                full,
            ));
        }
        if tau == 1.0 {
            for (segment, which) in [("am", 0), ("pm", 1), ("pm_minus_am", 2)] {
                for (k, direction) in DIRECTIONS.iter().enumerate() {
                    let mut values = vec![f64::NAN; n];
                    for &(i, s) in &scores {
                        values[i] = match which {
                            0 => s.am[k],
                            1 => s.pm[k],
                            _ => s.pm[k] - s.am[k],
                        };
                    }
                    columns.push(FactorColumn::new(
                        format!("yr_leadlag_market_{direction}_t1_{segment}"),
                        values,
                    ));
                }
            }
            own_one_second = own;
        }
    }

    let mut groups: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    let mut known = Vec::new();
    for &i in &members {
        if (1..=31).contains(&stocks[i].industry) {
            groups.entry(stocks[i].industry).or_default().push(i);
            known.push(i);
        }
    }
    let known_flow = if known.len() == members.len() {
        None
    } else {
        Some(density(stocks, &known))
    };
    let known_market = known_flow.as_ref().unwrap_or(&market);
    let mut within = vec![[f64::NAN; 4]; n];
    let mut outside = vec![[f64::NAN; 4]; n];
    for peers in groups.values() {
        let mut group_flow = density(stocks, peers);
        if peers.len() > 1 {
            let field = make_field(&group_flow, 1.0)?;
            for (i, s) in evaluate(stocks, peers, &own_one_second, &field, true)? {
                within[i] = s.full;
            }
        }
        if peers.len() < known.len() {
            for (flow, market_flow) in group_flow.iter_mut().zip(known_market) {
                for side in 0..2 {
                    flow[side] = (market_flow[side] - flow[side]).max(0.0);
                }
            }
            let field = make_field(&group_flow, 1.0)?;
            for (i, s) in evaluate(stocks, peers, &own_one_second, &field, false)? {
                outside[i] = s.full;
            }
        }
        // The group's fields and dense flow are dropped before the next industry.
    }
    for (label, which) in [
        ("industry", 0),
        ("cross_industry", 1),
        ("industry_minus_cross", 2),
    ] {
        for (k, direction) in DIRECTIONS.iter().enumerate() {
            let values = (0..n)
                .map(|i| match which {
                    0 => within[i][k],
                    1 => outside[i][k],
                    _ => within[i][k] - outside[i][k],
                })
                .collect();
            columns.push(FactorColumn::new(
                format!("yr_leadlag_{label}_{direction}_t1"),
                values,
            ));
        }
    }
    Ok(columns)
}
