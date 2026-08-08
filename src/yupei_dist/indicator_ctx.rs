//! 指标上下文（正式库内存版）: 21 张矩阵 + 每股统计量 + 行业 + 前一交易日矩阵。
//! 与 sandbox 版 API 保持一致（indicators/*.rs 无需改动即可编译）。

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use super::matrix_stage::{StockStats, MATRIX_SPECS};

/// 37 矩阵全集（与 matrix_stage.rs MATRIX_SPECS 顺序一致; 指标模块遍历用）。
/// 布局: cnt×9τ, vol×6τ, logvol×4τ, flow same/opp×4τ, urg same/opp×3τ, ext same/opp×2τ。
pub const MATRIX_LIST: [&str; 37] = [
    "cnt_t005", "cnt_t01", "cnt_t02", "cnt_t05", "cnt_t1", "cnt_t3", "cnt_t5", "cnt_t10", "cnt_t30",
    "vol_t02", "vol_t05", "vol_t1", "vol_t3", "vol_t5", "vol_t30",
    "logvol_t05", "logvol_t1", "logvol_t3", "logvol_t30",
    "flow_same_t02", "flow_opp_t02", "flow_same_t05", "flow_opp_t05",
    "flow_same_t1", "flow_opp_t1", "flow_same_t5", "flow_opp_t5",
    "urg_same_t1", "urg_opp_t1", "urg_same_t5", "urg_opp_t5", "urg_same_t30", "urg_opp_t30",
    "ext_same_t1", "ext_opp_t1", "ext_same_t5", "ext_opp_t5",
];

/// signed 族净矩阵清单（same − opp; 供 directionality/leadership 等有向净指标用）。
pub const NET_LIST: [&str; 9] = [
    "flow_t02", "flow_t05", "flow_t1", "flow_t5",
    "urg_t1", "urg_t5", "urg_t30",
    "ext_t1", "ext_t5",
];

/// 矩阵 τ（秒）: "cnt_t005" → 0.05; "flow_same_t5" → 5.0; "cnt_t1" → 1.0（与 MATRIX_SPECS 一致）。
/// 命名规则: 以 0 开头的多位数表示小数秒（005→0.05, 02→0.2）, 否则为整秒（1→1, 30→30）。
pub fn matrix_tau(name: &str) -> f64 {
    let t = name.rsplit('_').next().unwrap_or("t1");
    let s = t.trim_start_matches('t');
    let v: f64 = s.parse().unwrap_or(1.0);
    if s.len() > 1 && s.starts_with('0') { v / 100.0 } else { v }
}

/// 当日备份集（内存）
pub struct MatrixSet {
    pub n: usize,
    pub codes: Vec<String>,
    pub stats: Vec<StockStats>,
    pub mats: HashMap<String, Vec<f32>>,
    pub industry: Option<Vec<i16>>,
}

/// 前一交易日矩阵（只存需要的有向矩阵）
pub struct PrevMats {
    pub n: usize,
    pub codes: Vec<String>,
    pub mats: HashMap<String, Vec<f32>>,
}

pub struct PrevDay<'a> {
    pub set: &'a PrevMats,
    pub code_to_idx: HashMap<&'a str, usize>,
}

impl<'a> PrevDay<'a> {
    pub fn new(set: &'a PrevMats) -> Self {
        let code_to_idx = set
            .codes
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();
        PrevDay { set, code_to_idx }
    }
    pub fn prev_idx(&self, code: &str) -> Option<usize> {
        self.code_to_idx.get(code).copied()
    }
}

#[derive(Clone, Debug)]
pub struct IndicatorResult {
    pub name: String,
    pub values: Vec<f32>,
}

impl IndicatorResult {
    pub fn new(name: impl Into<String>, values: Vec<f32>) -> Self {
        IndicatorResult { name: name.into(), values }
    }
}

pub struct IndicatorCtx<'a> {
    pub set: &'a MatrixSet,
    pub prev: Option<PrevDay<'a>>,
    sym_cache: RefCell<HashMap<String, Arc<Vec<f32>>>>,
    rowsum_cache: RefCell<HashMap<String, Arc<Vec<f64>>>>,
}

impl<'a> IndicatorCtx<'a> {
    pub fn new(set: &'a MatrixSet, prev: Option<PrevDay<'a>>) -> Self {
        IndicatorCtx {
            set,
            prev,
            sym_cache: RefCell::new(HashMap::new()),
            rowsum_cache: RefCell::new(HashMap::new()),
        }
    }

    pub fn n(&self) -> usize {
        self.set.n
    }
    pub fn codes(&self) -> &[String] {
        &self.set.codes
    }
    pub fn stats(&self) -> &[StockStats] {
        &self.set.stats
    }
    pub fn industry(&self) -> Option<&[i16]> {
        self.set.industry.as_deref()
    }
    pub fn date(&self) -> i64 {
        0
    }

    pub fn matrix(&self, name: &str) -> Option<&[f32]> {
        self.set.mats.get(name).map(|m| m.as_slice())
    }

    pub fn has_matrix(&self, name: &str) -> bool {
        self.set.mats.contains_key(name)
    }

    pub fn matrix_names(&self) -> Vec<&'static str> {
        MATRIX_SPECS.iter().map(|s| s.name).collect()
    }

    /// 对称化: S_sym = S_dir + S_dirᵀ（对角 0）。结果缓存。
    pub fn symmetric(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        if let Some(v) = self.sym_cache.borrow().get(name) {
            return Some(v.clone());
        }
        let d = self.matrix(name)?;
        let n = self.n();
        let mut out = vec![0.0f32; n * n];
        for i in 0..n {
            let row = &d[i * n..(i + 1) * n];
            for j in 0..n {
                if i != j {
                    out[i * n + j] = row[j] + d[j * n + i];
                }
            }
        }
        let arc = Arc::new(out);
        self.sym_cache.borrow_mut().insert(name.to_string(), arc.clone());
        Some(arc)
    }

    pub fn row_sum_sym(&self, name: &str) -> Option<Arc<Vec<f64>>> {
        if let Some(v) = self.rowsum_cache.borrow().get(name) {
            return Some(v.clone());
        }
        let sym = self.symmetric(name)?;
        let n = self.n();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let row = &sym[i * n..(i + 1) * n];
            out.push(row.iter().map(|&v| v as f64).sum());
        }
        let arc = Arc::new(out);
        self.rowsum_cache.borrow_mut().insert(name.to_string(), arc.clone());
        Some(arc)
    }

    pub fn row_sum_dir(&self, name: &str) -> Option<Vec<f64>> {
        let d = self.matrix(name)?;
        let n = self.n();
        Some(
            (0..n)
                .map(|i| d[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum())
                .collect(),
        )
    }

    /// 有向净矩阵（signed 族）: net = same - opp。名称形如 "{family}_t{tau}"
    /// （如 "flow_t5" → flow_same_t5 − flow_opp_t5; "urg_t30" → urg_same_t30 − urg_opp_t30）。缓存。
    pub fn matrix_net(&self, name: &str) -> Option<Arc<Vec<f32>>> {
        let key = format!("net_{name}");
        if let Some(v) = self.sym_cache.borrow().get(&key) {
            return Some(v.clone());
        }
        let (family, tau) = name.rsplit_once('_')?;
        let same = self.matrix(&format!("{family}_same_{tau}"))?;
        let opp = self.matrix(&format!("{family}_opp_{tau}"))?;
        let n = self.n();
        let mut out = vec![0.0f32; n * n];
        for i in 0..n * n {
            out[i] = same[i] - opp[i];
        }
        let arc = Arc::new(out);
        self.sym_cache.borrow_mut().insert(key, arc.clone());
        Some(arc)
    }

    pub fn col_sum_dir(&self, name: &str) -> Option<Vec<f64>> {
        let d = self.matrix(name)?;
        let n = self.n();
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                out[j] += d[i * n + j] as f64;
            }
        }
        Some(out)
    }
}
