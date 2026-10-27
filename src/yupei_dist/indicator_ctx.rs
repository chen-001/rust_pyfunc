//! 指标上下文（正式库内存版）: 28 张矩阵 + 每股统计量 + 行业 + 前一交易日矩阵。
//! 与 sandbox 版 API 保持一致（indicators/*.rs 无需改动即可编译）。

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use super::matrix_stage::{StockStats, MATRIX_SPECS};

/// 28 矩阵全集（与 matrix_stage.rs MATRIX_SPECS 顺序一致; 指标模块遍历用）。
/// 布局: cnt×3τ, rvol×3τ, logvol×4τ, flow same/opp×4τ, urg same/opp×3τ, ext same/opp×2τ。
/// （cnt 9τ→3τ 与 vol→rvol 替换于 2026-10：τ 裁剪经穷举验证，见 matrix_stage.rs 注释）
pub const MATRIX_LIST: [&str; 28] = [
    "cnt_t005",
    "cnt_t1",
    "cnt_t30",
    "rvol_t005",
    "rvol_t1",
    "rvol_t30",
    "logvol_t05",
    "logvol_t1",
    "logvol_t3",
    "logvol_t30",
    "flow_same_t02",
    "flow_opp_t02",
    "flow_same_t05",
    "flow_opp_t05",
    "flow_same_t1",
    "flow_opp_t1",
    "flow_same_t5",
    "flow_opp_t5",
    "urg_same_t1",
    "urg_opp_t1",
    "urg_same_t5",
    "urg_opp_t5",
    "urg_same_t30",
    "urg_opp_t30",
    "ext_same_t1",
    "ext_opp_t1",
    "ext_same_t5",
    "ext_opp_t5",
];

/// signed 族净矩阵清单（same − opp; 供 directionality/leadership 等有向净指标用）。
pub const NET_LIST: [&str; 9] = [
    "flow_t02", "flow_t05", "flow_t1", "flow_t5", "urg_t1", "urg_t5", "urg_t30", "ext_t1", "ext_t5",
];

/// 矩阵 τ（秒）: "cnt_t005" → 0.05; "flow_same_t5" → 5.0; "cnt_t1" → 1.0（与 MATRIX_SPECS 一致）。
/// 命名规则: 以 0 开头的多位数表示小数秒（005→0.05, 02→0.2）, 否则为整秒（1→1, 30→30）。
pub fn matrix_tau(name: &str) -> f64 {
    let t = name.rsplit('_').next().unwrap_or("t1");
    let s = t.trim_start_matches('t');
    let v: f64 = s.parse().unwrap_or(1.0);
    // 尾码是去掉小数点的秒数: "005"→0.05, "05"→0.5, "02"→0.2, "1"→1, "30"→30。
    // 以 0 开头即小数码，位数决定 10 的次数: 值 = v / 10^(len-1)。
    // （旧实现对任何以 0 开头的码一律 /100，把 t05 算成 0.05、t02 算成 0.02，差 10 倍。）
    if s.len() > 1 && s.starts_with('0') {
        v / 10f64.powi(s.len() as i32 - 1)
    } else {
        v
    }
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
        IndicatorResult {
            name: name.into(),
            values,
        }
    }
}

pub struct IndicatorCtx<'a> {
    pub set: &'a MatrixSet,
    pub prev: Option<PrevDay<'a>>,
    /// 对称化/行和缓存：Mutex 包装（非 RefCell），让同一模块的矩阵循环
    /// 可以 rayon 并行——并行时 ctx 被多线程共享，RefCell 不是 Sync。
    /// 串行路径下 Mutex 的开销 = 一次无竞争加锁，可忽略。
    sym_cache: Mutex<HashMap<String, Arc<Vec<f32>>>>,
    rowsum_cache: Mutex<HashMap<String, Arc<Vec<f64>>>>,
    /// 精简模式：只算这些列名。None = 全量。
    /// 值取自 lite_names.rs 的 YUPEI_DIST_LITE_NAMES（或调用方给定子集）。
    pub keep: Option<HashSet<&'static str>>,
}

impl<'a> IndicatorCtx<'a> {
    pub fn new(set: &'a MatrixSet, prev: Option<PrevDay<'a>>) -> Self {
        IndicatorCtx {
            set,
            prev,
            sym_cache: Mutex::new(HashMap::new()),
            rowsum_cache: Mutex::new(HashMap::new()),
            keep: None,
        }
    }

    /// 精简模式构造：keep=要算的完整列名集合。
    pub fn with_keep(
        set: &'a MatrixSet,
        prev: Option<PrevDay<'a>>,
        keep: HashSet<&'static str>,
    ) -> Self {
        IndicatorCtx { keep: Some(keep), ..Self::new(set, prev) }
    }

    /// 该模块在矩阵 mat 上是否有要算的列。
    /// prefix 是列名中矩阵名之前的部分（如 "dep_"、"maxst_"）。
    /// keep=None（全量模式）恒真；keep=Some 时查 "{prefix}{mat}" 是否任一幸存列的前缀。
    /// 实现：幸存列名是 "{prefix}{mat}_{suffix}" 或 "{prefix}{mat}" 结尾，
    /// 用 starts_with("{prefix}{mat}") 判定（mat 名含 '_'，不会误配 cnt_t1 vs cnt_t30）。
    pub fn wants(&self, prefix: &str, mat: &str) -> bool {
        match &self.keep {
            None => true,
            Some(k) => {
                let head = format!("{prefix}{mat}");
                k.iter().any(|n| n.starts_with(head.as_str()))
            }
        }
    }

    /// 同 wants，但模块在该矩阵上出多个前缀的列时任一命中即算。
    pub fn wants_any(&self, prefixes: &[&str], mat: &str) -> bool {
        match &self.keep {
            None => true,
            Some(k) => prefixes.iter().any(|p| {
                let head = format!("{p}{mat}");
                k.iter().any(|n| n.starts_with(head.as_str()))
            }),
        }
    }

    /// 单个无矩阵列（标量/非 mat 命名）是否幸存。
    pub fn keep_name(&self, name: &'static str) -> bool {
        match &self.keep {
            None => true,
            Some(k) => k.contains(name),
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
        if let Some(v) = self.sym_cache.lock().unwrap().get(name) {
            return Some(v.clone());
        }
        let d = self.matrix(name)?;
        let n = self.n();
        // 对称化按行并行：单张矩阵 n² 次读写（5153² ≈ 2650 万），27 张串行要 3~4 秒。
        let mut out = vec![0.0f32; n * n];
        out.par_chunks_mut(n).enumerate().for_each(|(i, row_out)| {
            let row = &d[i * n..(i + 1) * n];
            for j in 0..n {
                if i != j {
                    row_out[j] = row[j] + d[j * n + i];
                }
            }
        });
        let arc = Arc::new(out);
        self.sym_cache
            .lock()
            .unwrap()
            .insert(name.to_string(), arc.clone());
        Some(arc)
    }

    pub fn row_sum_sym(&self, name: &str) -> Option<Arc<Vec<f64>>> {
        if let Some(v) = self.rowsum_cache.lock().unwrap().get(name) {
            return Some(v.clone());
        }
        let sym = self.symmetric(name)?;
        let n = self.n();
        // 行和同样按行并行（n² 次读取）
        let out: Vec<f64> = sym
            .par_chunks(n)
            .map(|row| row.iter().map(|&v| v as f64).sum())
            .collect();
        let arc = Arc::new(out);
        self.rowsum_cache
            .lock()
            .unwrap()
            .insert(name.to_string(), arc.clone());
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
        if let Some(v) = self.sym_cache.lock().unwrap().get(&key) {
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
        self.sym_cache.lock().unwrap().insert(key, arc.clone());
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
