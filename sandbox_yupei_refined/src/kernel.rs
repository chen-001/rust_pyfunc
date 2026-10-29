//! Correctly oriented, session-separated 10 ms exponential event kernels.
//!
//! Each tile owns one leading stock. Within a time block its two decayed event
//! fields are generated once, then reused while scanning every following stock.
//! This preserves the original sparse block algorithm without merging all event
//! pairs separately. All arithmetic inside a tile uses f64; the public matrices
//! are f32, and normalization is performed in f64 from those public values.
use crate::model::{Kernel, Stock};
use rayon::prelude::*;
use std::cell::RefCell;
use std::io::{self, Error, ErrorKind};

const BLOCK: u32 = 4096;
const SESSIONS: [(u32, u32); 2] = [(0, 720_000), (1_260_000, 1_962_000)];
const MIN_NORMAL: f64 = f64::MIN_POSITIVE;

thread_local! {
    // One scratch field per worker rather than one 64 KiB field per stock.
    static FIELD: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) };
}

fn invalid(message: impl Into<String>) -> io::Error {
    Error::new(ErrorKind::InvalidInput, message.into())
}

#[inline]
fn component_sum(lead: &[Vec<f32>; 4], k: usize, a: usize, b: usize) -> f64 {
    let mut value = 0.0;
    for p in 0..2 {
        if a != 0 && a != p + 1 {
            continue;
        }
        for q in 0..2 {
            if b == 0 || b == q + 1 {
                value += lead[p * 2 + q][k] as f64;
            }
        }
    }
    value
}

/// Compute all four directed event products and nine normalized direction pairs.
///
/// Direction indices are all=0, buy=1, sell=2. `lead[p*2+q][i*n+j]`
/// always means stock i's direction p precedes stock j's direction q. Events
/// in the same 10 ms bucket contribute half in each temporal direction. Both
/// sessions are half-open; lunchtime is never bridged by the exponential state.
/// Empty directional energy is legitimate inactivity, while its correlations
/// are undefined and therefore stay NaN, including diagonal entries.
pub fn compute_kernel(stocks: &[Stock], tau: f64) -> io::Result<Kernel> {
    if !tau.is_finite() || tau <= 0.0 {
        return Err(invalid("kernel tau must be finite and positive"));
    }
    let n = stocks.len();
    if n == 0 {
        return Err(invalid("kernel needs at least one stock"));
    }
    let nn = n
        .checked_mul(n)
        .ok_or_else(|| invalid("kernel matrix size overflow"))?;
    let row_size = n
        .checked_mul(4)
        .and_then(|v| v.checked_add(2))
        .ok_or_else(|| invalid("kernel tile size overflow"))?;
    for stock in stocks {
        let mut previous = None;
        for cell in &stock.cells {
            if !SESSIONS
                .iter()
                .any(|&(lo, hi)| cell.slot >= lo && cell.slot < hi)
            {
                return Err(invalid(format!(
                    "{}: slot {} is outside continuous sessions",
                    stock.code, cell.slot
                )));
            }
            if previous.is_some_and(|p| cell.slot < p) {
                return Err(invalid(format!(
                    "{}: event slots are not sorted",
                    stock.code
                )));
            }
            if !cell.buy.is_finite() || !cell.sell.is_finite() || cell.buy < 0.0 || cell.sell < 0.0
            {
                return Err(invalid(format!(
                    "{}: buy/sell event weights must be finite and nonnegative",
                    stock.code
                )));
            }
            previous = Some(cell.slot);
        }
    }

    let decay = (-0.01 / tau).exp();
    // Four accumulated components per following stock, then two persistent fields.
    let mut tiles: Vec<Vec<f64>> = (0..n).map(|_| vec![0.0; row_size]).collect();
    let mut cursors = vec![0usize; n];
    let mut offsets = vec![0usize; n + 1];
    let mut block_cells: Vec<(u32, f64, f64)> = Vec::new();

    for (session_start, session_end) in SESSIONS {
        for tile in &mut tiles {
            tile[row_size - 2..].fill(0.0);
        }
        // ceil coverage is intentional: the last block is shorter than BLOCK.
        let mut start = session_start;
        while start < session_end {
            let end = (start + BLOCK).min(session_end);
            let width = (end - start) as usize;
            block_cells.clear();
            for (i, stock) in stocks.iter().enumerate() {
                offsets[i] = block_cells.len();
                let mut cursor = cursors[i];
                while cursor < stock.cells.len() && stock.cells[cursor].slot < end {
                    let cell = &stock.cells[cursor];
                    debug_assert!(cell.slot >= start);
                    block_cells.push((cell.slot - start, cell.buy as f64, cell.sell as f64));
                    cursor += 1;
                }
                cursors[i] = cursor;
            }
            offsets[n] = block_cells.len();
            if block_cells.is_empty() {
                let factor = (-(width as f64) * 0.01 / tau).exp();
                for tile in &mut tiles {
                    for value in &mut tile[row_size - 2..] {
                        *value *= factor;
                        if *value < MIN_NORMAL {
                            *value = 0.0;
                        }
                    }
                }
                start = end;
                continue;
            }
            tiles.par_iter_mut().enumerate().for_each(|(j, tile)| {
                let (row, state) = tile.split_at_mut(n * 4);
                let source = &block_cells[offsets[j]..offsets[j + 1]];
                if source.is_empty() && state[0] == 0.0 && state[1] == 0.0 {
                    return;
                }
                FIELD.with(|buffer| {
                    let mut field = buffer.borrow_mut();
                    field.resize(width * 2, 0.0);
                    let (mut buy_history, mut sell_history) = (state[0], state[1]);
                    let mut cursor = 0;
                    for k in 0..width {
                        let (mut buy, mut sell) = (0.0, 0.0);
                        while cursor < source.len() && source[cursor].0 as usize == k {
                            buy += source[cursor].1;
                            sell += source[cursor].2;
                            cursor += 1;
                        }
                        field[k * 2] = buy_history + 0.5 * buy;
                        field[k * 2 + 1] = sell_history + 0.5 * sell;
                        buy_history = (buy_history + buy) * decay;
                        sell_history = (sell_history + sell) * decay;
                    }
                    state[0] = if buy_history >= MIN_NORMAL {
                        buy_history
                    } else {
                        0.0
                    };
                    state[1] = if sell_history >= MIN_NORMAL {
                        sell_history
                    } else {
                        0.0
                    };
                    for i in 0..n {
                        let target = &block_cells[offsets[i]..offsets[i + 1]];
                        if target.is_empty() {
                            continue;
                        }
                        let acc = &mut row[i * 4..i * 4 + 4];
                        let (mut pp, mut pm, mut mp, mut mm) = (acc[0], acc[1], acc[2], acc[3]);
                        for &(slot, buy, sell) in target {
                            let gp = field[slot as usize * 2];
                            let gm = field[slot as usize * 2 + 1];
                            // Source j leads target i: p(source) then q(target).
                            pp += gp * buy;
                            pm += gp * sell;
                            mp += gm * buy;
                            mm += gm * sell;
                        }
                        acc.copy_from_slice(&[pp, pm, mp, mm]);
                    }
                });
            });
            start = end;
        }
    }
    debug_assert!(stocks
        .iter()
        .zip(&cursors)
        .all(|(s, &c)| c == s.cells.len()));

    let mut lead: [Vec<f32>; 4] = std::array::from_fn(|_| vec![0.0; nn]);
    let [pp, pm, mp, mm] = &mut lead;
    pp.par_chunks_mut(n)
        .zip(pm.par_chunks_mut(n))
        .zip(mp.par_chunks_mut(n))
        .zip(mm.par_chunks_mut(n))
        .enumerate()
        .for_each(|(j, (((pp, pm), mp), mm))| {
            let row = &tiles[j];
            for i in 0..n {
                pp[i] = row[i * 4] as f32;
                pm[i] = row[i * 4 + 1] as f32;
                mp[i] = row[i * 4 + 2] as f32;
                mm[i] = row[i * 4 + 3] as f32;
            }
        });
    drop(tiles);
    if lead
        .iter()
        .any(|matrix| matrix.iter().any(|v| !v.is_finite()))
    {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "event kernel exceeds finite f32 matrix range",
        ));
    }
    let energy: [Vec<f64>; 3] = std::array::from_fn(|a| {
        (0..n)
            .map(|i| 2.0 * component_sum(&lead, i * n + i, a, a))
            .collect()
    });
    let mut rho: [Vec<f32>; 9] = std::array::from_fn(|_| vec![f32::NAN; nn]);
    for a in 0..3 {
        for b in 0..3 {
            rho[a * 3 + b]
                .par_chunks_mut(n)
                .enumerate()
                .for_each(|(i, row)| {
                    if energy[a][i] <= 0.0 {
                        return;
                    }
                    for (j, value) in row.iter_mut().enumerate() {
                        if energy[b][j] <= 0.0 {
                            continue;
                        }
                        let numerator = component_sum(&lead, i * n + j, a, b)
                            + component_sum(&lead, j * n + i, b, a);
                        *value = (numerator / (energy[a][i] * energy[b][j]).sqrt()) as f32;
                    }
                });
        }
    }
    Ok(Kernel {
        n,
        lead,
        energy,
        rho,
    })
}
