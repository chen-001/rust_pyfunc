//! Coalesce adjacent requested factors while retaining the existing thread-per-factor workers.
//! A group is read once; matrices are moved to consumers, never cloned.
use crate::factor_store_v5::{FactorBatchRaw, FactorStoreReader};
use ndarray::Array2;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

enum State {
    Empty,
    Loading,
    ReadyRaw {
        raw: Arc<FactorBatchRaw>,
        taken: Vec<bool>,
        remaining: usize,
    },
    Ready(Vec<Option<Array2<f32>>>),
    Consumed,
    Failed(String),
}
struct Group {
    start: usize,
    end: usize,
    state: Mutex<State>,
    wake: Condvar,
}
pub struct FactorReadCache {
    groups: BTreeMap<usize, Arc<Group>>,
    pub reads: AtomicUsize,
}
impl FactorReadCache {
    pub fn new(
        reader: &FactorStoreReader,
        cols: &[usize],
        batch: usize,
        matrix_bytes: usize,
        max_batch_bytes: usize,
    ) -> Result<Self, String> {
        if batch == 0 || max_batch_bytes == 0 {
            return Err("io_batch_size/io_batch_bytes 必须 > 0".into());
        }
        let max_cols = batch.min((max_batch_bytes / matrix_bytes.max(1)).max(1));
        let mut cols = cols.to_vec();
        cols.sort_unstable();
        if cols.windows(2).any(|w| w[0] == w[1]) {
            return Err("读取计划中因子列重复".into());
        }
        let layouts = reader.group_layout();
        let mut groups = BTreeMap::new();
        let mut pos = 0;
        while pos < cols.len() {
            let start = cols[pos];
            let group_end = layouts
                .iter()
                .find(|(_, _, off, n)| start >= *off && start < off + n)
                .map(|(_, _, off, n)| off + n)
                .ok_or("读取因子列越界")?;
            let mut next = pos + 1;
            let mut raw_bytes = reader.factor_read_bytes(start)?;
            while next < cols.len()
                && next - pos < max_cols
                && cols[next] == cols[next - 1] + 1
                && cols[next] < group_end
            {
                let added = reader.factor_read_bytes(cols[next])?;
                if raw_bytes.saturating_add(added) > max_batch_bytes {
                    break;
                }
                raw_bytes = raw_bytes.saturating_add(added);
                next += 1;
            }
            let g = Arc::new(Group {
                start,
                end: cols[next - 1] + 1,
                state: Mutex::new(State::Empty),
                wake: Condvar::new(),
            });
            for &col in &cols[pos..next] {
                groups.insert(col, g.clone());
            }
            pos = next;
        }
        Ok(Self {
            groups,
            reads: AtomicUsize::new(0),
        })
    }
    pub fn take(
        &self,
        reader: &FactorStoreReader,
        col: usize,
        dates: &[i32],
        stocks: &[String],
        maps: &[(Vec<usize>, Vec<usize>)],
    ) -> Result<Array2<f32>, String> {
        let group = self.groups.get(&col).ok_or("因子不在读取计划中")?;
        let mut state = group.state.lock().map_err(|e| e.to_string())?;
        loop {
            match &mut *state {
                State::ReadyRaw {
                    raw,
                    taken,
                    remaining,
                } => {
                    let idx = col - group.start;
                    if *taken.get(idx).ok_or("因子列偏移越界")? {
                        return Err("因子矩阵被重复消费".into());
                    }
                    taken[idx] = true;
                    *remaining -= 1;
                    let raw_arc = raw.clone();
                    if *remaining == 0 {
                        *state = State::Consumed;
                    }
                    drop(state);
                    return reader.scatter_factor_from_raw_batch(&raw_arc, col, dates, stocks, maps);
                }
                State::Ready(values) => {
                    return values[col - group.start]
                        .take()
                        .ok_or_else(|| "因子矩阵被重复消费".into())
                }
                State::Consumed => return Err("因子矩阵被重复消费".into()),
                State::Failed(error) => return Err(error.clone()),
                State::Loading => state = group.wake.wait(state).map_err(|e| e.to_string())?,
                State::Empty => {
                    *state = State::Loading;
                    drop(state);
                    self.reads.fetch_add(1, Ordering::Relaxed);
                    let len = group.end - group.start;
                    let next_state = match reader.try_read_factors_raw_batch(group.start, group.end)
                    {
                        Ok(Some(raw)) => State::ReadyRaw {
                            raw: Arc::new(raw),
                            taken: vec![false; len],
                            remaining: len,
                        },
                        Ok(None) => {
                            let fallback = if len == 1 {
                                reader
                                    .read_factor_to_matrix_fast(col, dates, stocks, maps)
                                    .map(|m| vec![m])
                            } else {
                                reader.read_factors_batch_v7_fast(
                                    group.start,
                                    group.end,
                                    dates,
                                    stocks,
                                    maps,
                                )
                            };
                            match fallback {
                                Ok(values) => State::Ready(values.into_iter().map(Some).collect()),
                                Err(e) => State::Failed(e),
                            }
                        }
                        Err(e) => State::Failed(e),
                    };
                    state = group.state.lock().map_err(|e| e.to_string())?;
                    *state = next_state;
                    group.wake.notify_all();
                }
            }
        }
    }
}
