//! Historical factors consume explicit daily snapshots; no implicit backfill.
//! Rolling N includes today. `previous_mean20` excludes today and requires 20
//! preceding trading dates. Missing observations never become zero or a shorter window.
use std::collections::{BTreeMap, BTreeSet};
use std::io;

use crate::yupei_gpt::model::{AuditRow, FactorColumn, NeighborState, Snapshot};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn mean(values: &[f64]) -> f64 {
    values.iter().map(|v| v / values.len() as f64).sum()
}

struct Day<'a> {
    snapshot: &'a Snapshot,
    codes: BTreeMap<&'a str, usize>,
    columns: BTreeMap<&'a str, &'a FactorColumn>,
    neighbors: BTreeMap<&'a str, &'a NeighborState>,
}

impl<'a> Day<'a> {
    fn new(snapshot: &'a Snapshot) -> io::Result<Self> {
        let n = snapshot.codes.len();
        if n == 0 || snapshot.schema.is_empty() || snapshot.config_hash.is_empty() {
            return Err(invalid(format!(
                "snapshot {} has empty codes/schema/config",
                snapshot.date
            )));
        }
        let mut codes = BTreeMap::new();
        for (i, code) in snapshot.codes.iter().enumerate() {
            if code.is_empty() || codes.insert(code.as_str(), i).is_some() {
                return Err(invalid(format!(
                    "snapshot {} duplicate/empty code {code:?}",
                    snapshot.date
                )));
            }
        }
        let mut columns = BTreeMap::new();
        for col in &snapshot.columns {
            if col.values.len() != n
                || col.name.is_empty()
                || col.values.iter().flatten().any(|v| !v.is_finite())
                || columns.insert(col.name.as_str(), col).is_some()
            {
                return Err(invalid(format!(
                    "snapshot {} invalid/duplicate column {}",
                    snapshot.date, col.name
                )));
            }
        }
        let mut neighbors = BTreeMap::new();
        for state in &snapshot.neighbors {
            if state.neighbors.len() != n
                || state.valid.len() != n
                || neighbors.insert(state.chain.as_str(), state).is_some()
            {
                return Err(invalid(format!(
                    "snapshot {} malformed neighbor state {}",
                    snapshot.date, state.chain
                )));
            }
            for (i, row) in state.neighbors.iter().enumerate() {
                let mut seen = BTreeSet::new();
                for code in row {
                    let Some(&j) = codes.get(code.as_str()) else {
                        return Err(invalid(format!(
                            "snapshot {} unknown neighbor {code}",
                            snapshot.date
                        )));
                    };
                    if i == j || !state.valid[i] || !state.valid[j] || !seen.insert(code) {
                        return Err(invalid(format!(
                            "snapshot {} invalid neighbor {} -> {code}",
                            snapshot.date, snapshot.codes[i]
                        )));
                    }
                }
            }
        }
        Ok(Self {
            snapshot,
            codes,
            columns,
            neighbors,
        })
    }

    fn value(&self, name: &str, code: &str) -> Option<f64> {
        let &i = self.codes.get(code)?;
        self.columns.get(name)?.values[i]
    }

    fn require(&self, name: &str) -> io::Result<&FactorColumn> {
        self.columns.get(name).copied().ok_or_else(|| {
            invalid(format!(
                "snapshot {} missing required base column {name}",
                self.snapshot.date
            ))
        })
    }
}

#[derive(Clone, Copy)]
enum Op {
    Mean,
    Min,
    Max,
    FoldMax,
}

struct Spec {
    chain: &'static str,
    metric: &'static str,
    window: usize,
    op: Op,
}

const SPECS: [Spec; 14] = [
    Spec {
        chain: "AAa",
        metric: "row_rho_skew",
        window: 5,
        op: Op::Mean,
    },
    Spec {
        chain: "ppp",
        metric: "row_gs_skew",
        window: 20,
        op: Op::Mean,
    },
    Spec {
        chain: "ppp",
        metric: "res_z_p90",
        window: 5,
        op: Op::Min,
    },
    Spec {
        chain: "ppp",
        metric: "r4_ov_g_k100",
        window: 10,
        op: Op::Mean,
    },
    Spec {
        chain: "mmm",
        metric: "r4_ov_z_k100",
        window: 10,
        op: Op::Mean,
    },
    Spec {
        chain: "ppp",
        metric: "row_self_assoc_p95",
        window: 20,
        op: Op::FoldMax,
    },
    Spec {
        chain: "ppa",
        metric: "row_gnpos_p90",
        window: 5,
        op: Op::Min,
    },
    Spec {
        chain: "AAp",
        metric: "e_w_std_soft",
        window: 5,
        op: Op::Max,
    },
    Spec {
        chain: "AAp",
        metric: "e_w_std_soft",
        window: 10,
        op: Op::Max,
    },
    Spec {
        chain: "pmp",
        metric: "x0topk20_count",
        window: 5,
        op: Op::Mean,
    },
    Spec {
        chain: "AAp",
        metric: "gtopk100_count",
        window: 10,
        op: Op::Mean,
    },
    Spec {
        chain: "AAm",
        metric: "res_g_max",
        window: 10,
        op: Op::Mean,
    },
    Spec {
        chain: "ppp",
        metric: "best_z_partner_reverse_rank",
        window: 5,
        op: Op::Mean,
    },
    Spec {
        chain: "ppp",
        metric: "best_z_partner_reverse_rank_fraction",
        window: 5,
        op: Op::Mean,
    },
];

struct Context<'a> {
    current: &'a Snapshot,
    days: BTreeMap<i64, Day<'a>>,
    calendar: &'a [i64],
    today: usize,
    allow: bool,
}

impl<'a> Context<'a> {
    /// offset=0 includes current; offset=1 ends at previous trading day.
    fn window(&self, count: usize, offset: usize) -> io::Result<(Vec<i64>, Vec<i64>, bool)> {
        let available = self.today + 1;
        if available < count + offset {
            if !self.allow {
                return Err(invalid(format!(
                    "date {} calendar has {available} dates; window requires {}",
                    self.current.date,
                    count + offset
                )));
            }
            return Ok((Vec::new(), Vec::new(), false));
        }
        let end = available - offset;
        let dates = self.calendar[end - count..end].to_vec();
        let missing: Vec<i64> = dates
            .iter()
            .copied()
            .filter(|d| !self.days.contains_key(d))
            .collect();
        if !missing.is_empty() && !self.allow {
            return Err(invalid(format!(
                "date {} missing required trading-date snapshots {missing:?}",
                self.current.date
            )));
        }
        Ok((dates, missing, true))
    }

    fn values(&self, dates: &[i64], name: &str, code: &str) -> Option<Vec<f64>> {
        dates
            .iter()
            .map(|d| self.days.get(d)?.value(name, code))
            .collect()
    }

    fn audit(
        &self,
        name: &str,
        values: &[Option<f64>],
        dates: &[i64],
        missing: &[i64],
        calendar_ok: bool,
        definition: &str,
    ) -> AuditRow {
        let n = values.iter().filter(|v| v.is_some()).count();
        let missing_codes: Vec<&str> = self
            .current
            .codes
            .iter()
            .zip(values)
            .filter_map(|(c, v)| v.is_none().then_some(c.as_str()))
            .take(20)
            .collect();
        AuditRow {
            name: name.into(), layer:"temporal".into(), n,
            value:Some(n as f64 / values.len() as f64),
            detail:serde_json::json!({
                "definition":definition, "dates":dates, "missing_dates":missing,
                "calendar_window_complete":calendar_ok, "allow_incomplete":self.allow,
                "null_count":values.len()-n, "null_code_examples":missing_codes,
                "null_reason":"missing calendar/history or stock observation, invalid direction, empty comparable neighborhood, or undefined denominator; never filled with zero"
            }).to_string(),
        }
    }

    fn require_bases(&self, name: &str, dates: &[i64]) -> io::Result<()> {
        self.days[&self.current.date].require(name)?;
        for date in dates {
            if let Some(day) = self.days.get(date) {
                day.require(name)?;
            }
        }
        Ok(())
    }
}

fn append(
    columns: &mut Vec<FactorColumn>,
    audits: &mut Vec<AuditRow>,
    ctx: &Context,
    name: String,
    values: Vec<Option<f64>>,
    dates: &[i64],
    missing: &[i64],
    calendar_ok: bool,
    definition: &str,
) {
    audits.push(ctx.audit(&name, &values, dates, missing, calendar_ok, definition));
    columns.push(FactorColumn { name, values });
}

fn rolling(
    ctx: &Context,
    spec: &Spec,
    columns: &mut Vec<FactorColumn>,
    audits: &mut Vec<AuditRow>,
) -> io::Result<()> {
    let base = format!("yr_{}_{}", spec.chain, spec.metric);
    let op_name = match spec.op {
        Op::Mean => "mean",
        Op::Min => "min",
        Op::Max => "max",
        Op::FoldMax => "absdev_csmean_max",
    };
    let name = format!("{base}_{op_name}_smooth_{}", spec.window);
    let (dates, missing, calendar_ok) = ctx.window(spec.window, 0)?;
    ctx.require_bases(&base, &dates)?;
    let ready = calendar_ok && missing.is_empty();
    let means: Vec<f64> = if ready && matches!(spec.op, Op::FoldMax) {
        dates
            .iter()
            .map(|d| {
                let v: Vec<f64> = ctx.days[d].columns[base.as_str()]
                    .values
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                if v.is_empty() {
                    f64::NAN
                } else {
                    mean(&v)
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let values = ctx
        .current
        .codes
        .iter()
        .map(|code| {
            if !ready {
                return None;
            }
            let mut v = ctx.values(&dates, &base, code)?;
            if matches!(spec.op, Op::FoldMax) {
                for (x, mean) in v.iter_mut().zip(&means) {
                    *x = (*x - mean).abs();
                }
            }
            if v.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let value = match spec.op {
                Op::Mean => mean(&v),
                Op::Min => v.into_iter().fold(f64::INFINITY, f64::min),
                Op::Max | Op::FoldMax => v.into_iter().fold(f64::NEG_INFINITY, f64::max),
            };
            value.is_finite().then_some(value)
        })
        .collect();
    let definition = if matches!(spec.op, Op::FoldMax) {
        "For each trading date, abs(base - that day's finite-stock cross-sectional mean), then maximum over complete 20-day window including today. This explicitly defined fold is not asserted identical to legacy fold."
    } else {
        "Complete trading-calendar rolling window including today; each stock requires one finite base value on every required date."
    };
    append(
        columns,
        audits,
        ctx,
        name,
        values,
        &dates,
        &missing,
        calendar_ok,
        definition,
    );
    Ok(())
}

fn breadth_history(
    ctx: &Context,
    chain: &str,
    columns: &mut Vec<FactorColumn>,
    audits: &mut Vec<AuditRow>,
) -> io::Result<()> {
    let base = format!("yr_{chain}_row_gnpos_p90");
    let (dates, missing, calendar_ok) = ctx.window(10, 0)?;
    ctx.require_bases(&base, &dates)?;
    let ready = calendar_ok && missing.is_empty();
    let medians: Vec<f64> = if ready {
        dates
            .iter()
            .map(|d| {
                let mut v: Vec<f64> = ctx.days[d].columns[base.as_str()]
                    .values
                    .iter()
                    .flatten()
                    .copied()
                    .collect();
                v.sort_by(f64::total_cmp);
                if v.is_empty() {
                    f64::NAN
                } else {
                    v[(v.len() - 1) / 2] / 2.0 + v[v.len() / 2] / 2.0
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    let values = ctx
        .current
        .codes
        .iter()
        .map(|code| {
            if !ready || medians.iter().any(|v| !v.is_finite()) {
                return None;
            }
            let v = ctx.values(&dates, &base, code)?;
            Some(v.iter().zip(&medians).filter(|(x, m)| x > m).count() as f64 / 10.0)
        })
        .collect();
    append(columns,audits,ctx,format!("{base}_above_csmedian_persistence_10"),values,&dates,&missing,calendar_ok,
        "Fraction of complete 10 trading dates including today with breadth strictly above that date's finite-stock cross-sectional median; comparisons use each date's own median.");

    let (dates, missing, calendar_ok) = ctx.window(20, 1)?;
    ctx.require_bases(&base, &dates)?;
    let ready = calendar_ok && missing.is_empty();
    let values = ctx
        .current
        .codes
        .iter()
        .map(|code| {
            if !ready {
                return None;
            }
            let current = ctx.days[&ctx.current.date].value(&base, code)?;
            let v = ctx.values(&dates, &base, code)?;
            let value = current - mean(&v);
            value.is_finite().then_some(value)
        })
        .collect();
    append(columns,audits,ctx,format!("{base}_change_vs_previous_mean20"),values,&dates,&missing,calendar_ok,
        "Today's breadth minus the mean of exactly 20 preceding trading dates, excluding today; 21 dated snapshots are needed including today. This is a level difference, not a percentage ratio.");
    Ok(())
}

fn edge_changes(
    ctx: &Context,
    chain: &str,
    columns: &mut Vec<FactorColumn>,
    audits: &mut Vec<AuditRow>,
) -> io::Result<()> {
    let (dates, missing, calendar_ok) = ctx.window(1, 1)?;
    let now = &ctx.days[&ctx.current.date];
    let state = now
        .neighbors
        .get(chain)
        .ok_or_else(|| invalid(format!("current snapshot lacks neighbor state {chain}")))?;
    let mut result = vec![[None; 3]; ctx.current.codes.len()];
    let mut eligible_sizes = vec![0usize; ctx.current.codes.len()];
    if calendar_ok && missing.is_empty() {
        let previous = &ctx.days[&dates[0]];
        let old = previous
            .neighbors
            .get(chain)
            .ok_or_else(|| invalid(format!("previous snapshot lacks neighbor state {chain}")))?;
        let eligible: BTreeSet<&str> = ctx
            .current
            .codes
            .iter()
            .enumerate()
            .filter_map(|(i, c)| {
                let &j = previous.codes.get(c.as_str())?;
                (state.valid[i] && old.valid[j]).then_some(c.as_str())
            })
            .collect();
        for (i, code) in ctx.current.codes.iter().enumerate() {
            if !eligible.contains(code.as_str()) {
                continue;
            }
            eligible_sizes[i] = eligible.len().saturating_sub(1);
            if eligible_sizes[i] == 0 {
                continue;
            }
            let j = previous.codes[code.as_str()];
            let today: BTreeSet<&str> = state.neighbors[i]
                .iter()
                .map(String::as_str)
                .filter(|c| eligible.contains(c))
                .collect();
            let yesterday: BTreeSet<&str> = old.neighbors[j]
                .iter()
                .map(String::as_str)
                .filter(|c| eligible.contains(c))
                .collect();
            let intersection = today.intersection(&yesterday).count();
            let union = today.union(&yesterday).count();
            result[i][0] = (!today.is_empty())
                .then(|| (today.len() - intersection) as f64 / today.len() as f64);
            result[i][1] = (!yesterday.is_empty())
                .then(|| (yesterday.len() - intersection) as f64 / yesterday.len() as f64);
            result[i][2] = (union > 0).then(|| intersection as f64 / union as f64);
        }
    }
    for (k, suffix) in ["new_share", "lost_share", "jaccard"]
        .into_iter()
        .enumerate()
    {
        let definition=format!("Positive-GS top100 neighbor {suffix}, comparing previous trading date on the intersection of direction-valid stock codes. New/current neighbors; lost/previous neighbors; intersection/union. Empty denominator is null. Missing/IPO/delisted/inactive partner codes are excluded from BOTH days. This measures retained top100 neighborhoods, not every positive GS edge.");
        let values = result.iter().map(|v| v[k]).collect();
        append(
            columns,
            audits,
            ctx,
            format!("yr_{chain}_positive_gs_neighbors_{suffix}_previous_day"),
            values,
            &dates,
            &missing,
            calendar_ok,
            &definition,
        );
    }
    audits.push(AuditRow{name:format!("yr_{chain}_history_common_valid_partners"),layer:"temporal_domain".into(),
        n:eligible_sizes.iter().filter(|&&v|v>0).count(),value:eligible_sizes.iter().copied().max().map(|v|v as f64),
        detail:"Maximum common direction-valid partner count; zero means no comparable domain, not observed loss.".into()});
    Ok(())
}

/// Compute a deterministic schema for chains actually present in today's base
/// columns. Invalid snapshot structure always errors; `allow_incomplete` only
/// permits missing dates/insufficient stock history to produce explicit nulls.
pub fn temporal_columns(
    current: &Snapshot,
    history: &[Snapshot],
    trading_days: &[i64],
    allow_incomplete: bool,
) -> io::Result<(Vec<FactorColumn>, Vec<AuditRow>)> {
    if trading_days.is_empty() || trading_days.windows(2).any(|w| w[0] >= w[1]) {
        return Err(invalid(
            "trading_days must be nonempty, unique, strictly increasing",
        ));
    }
    let today = trading_days.binary_search(&current.date).map_err(|_| {
        invalid(format!(
            "target date {} absent from trading calendar",
            current.date
        ))
    })?;
    let current_day = Day::new(current)?;
    let expected: BTreeSet<&str> = current_day.columns.keys().copied().collect();
    let mut days = BTreeMap::new();
    for snapshot in history {
        if snapshot.date >= current.date || days.contains_key(&snapshot.date) {
            return Err(invalid(format!(
                "historical date {} is duplicate or not earlier than {}",
                snapshot.date, current.date
            )));
        }
        if snapshot.schema != current.schema || snapshot.config_hash != current.config_hash {
            return Err(invalid(format!(
                "historical date {} schema/config mismatch",
                snapshot.date
            )));
        }
        if trading_days.binary_search(&snapshot.date).is_err() {
            return Err(invalid(format!(
                "historical date {} absent from trading calendar",
                snapshot.date
            )));
        }
        let day = Day::new(snapshot)?;
        if day.columns.keys().copied().collect::<BTreeSet<_>>() != expected {
            return Err(invalid(format!(
                "historical date {} column schema differs",
                snapshot.date
            )));
        }
        days.insert(snapshot.date, day);
    }
    if history.is_empty() && !allow_incomplete {
        return Err(invalid(
            "explicit historical snapshots required; none supplied",
        ));
    }
    // Extension placeholders such as yr_mmm_*_industry_activity_residual may
    // exist even when mmm was not computed. Only exact network base columns
    // establish a chain, never an arbitrary matching prefix.
    let chains: BTreeSet<&str> = SPECS
        .iter()
        .filter(|spec| {
            current_day
                .columns
                .contains_key(format!("yr_{}_{}", spec.chain, spec.metric).as_str())
        })
        .map(|spec| spec.chain)
        .chain(["AAa", "ppp", "mmm"].into_iter().filter(|chain| {
            current_day
                .columns
                .contains_key(format!("yr_{chain}_row_gnpos_p90").as_str())
        }))
        .collect();
    days.insert(current.date, current_day);
    let ctx = Context {
        current,
        days,
        calendar: trading_days,
        today,
        allow: allow_incomplete,
    };
    let (mut columns, mut audits) = (Vec::new(), Vec::new());
    for spec in &SPECS {
        if chains.contains(spec.chain) {
            rolling(&ctx, spec, &mut columns, &mut audits)?;
        }
    }
    for chain in ["AAa", "ppp", "mmm"] {
        if chains.contains(chain) {
            breadth_history(&ctx, chain, &mut columns, &mut audits)?;
            edge_changes(&ctx, chain, &mut columns, &mut audits)?;
        }
    }
    Ok((columns, audits))
}
