//! 指标上下文: 矩阵（mmap）、统计量、行业、前一日数据, 以及常用辅助（对称化等）。
//! 每个降维指标模块的入口: compute(&IndicatorCtx) -> Vec<IndicatorResult>。

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::matrix_stage::{StockStats, MATRIX_SPECS};

/// 单只矩阵文件的 mmap 视图
pub struct MmapMatrix {
    pub n: usize,
    pub mmap: memmap2::Mmap,
}

impl MmapMatrix {
    pub fn data(&self) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(self.mmap.as_ptr() as *const f32, self.n * self.n)
        }
    }
}

/// 一个日期的完整备份集（矩阵 + 代码 + 统计量）
pub struct BackupSet {
    pub date: i64,
    pub n: usize,
    pub codes: Vec<String>,
    pub stats: Vec<StockStats>,
    pub mats: HashMap<String, MmapMatrix>,
    pub industry: Option<Vec<i16>>,
}

impl BackupSet {
    /// 从备份目录加载（date 目录）
    pub fn load(dir: &Path, date: i64, load_mats: bool) -> std::io::Result<BackupSet> {
        let d = dir.join(date.to_string());
        let codes = crate::matrix_store::read_codes(&d)?;
        let stats = crate::matrix_store::read_stats(&d)?;
        let mut mats = HashMap::new();
        if load_mats {
            for spec in MATRIX_SPECS.iter() {
                let p = d.join("mats").join(format!("{}.bin", spec.name));
                if p.exists() {
                    let f = std::fs::File::open(&p)?;
                    let mmap = unsafe { memmap2::Mmap::map(&f)? };
                    mats.insert(spec.name.to_string(), MmapMatrix { n: codes.len(), mmap });
                }
            }
        }
        let n = codes.len();
        let industry = crate::industry::load_industry_bin(&d.join("industry.bin"), &codes).ok();
        let industry = industry.filter(|v| v.len() == n);
        Ok(BackupSet { date, n, codes, stats, mats, industry })
    }
}

/// 前一日备份（时间动态类指标用; 按代码对齐）
pub struct PrevDay<'a> {
    pub set: &'a BackupSet,
    pub code_to_idx: HashMap<&'a str, usize>,
}

impl<'a> PrevDay<'a> {
    pub fn new(set: &'a BackupSet) -> Self {
        let code_to_idx = set.codes.iter().enumerate().map(|(i, c)| (c.as_str(), i)).collect();
        PrevDay { set, code_to_idx }
    }
    /// 当前指标代码在 prev 集中的索引
    pub fn prev_idx(&self, code: &str) -> Option<usize> {
        self.code_to_idx.get(code).copied()
    }
}

/// 指标输出
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

/// 指标上下文（借用备份集; prev 可空）
pub struct IndicatorCtx<'a> {
    pub set: &'a BackupSet,
    pub prev: Option<PrevDay<'a>>,
    sym_cache: RefCell<HashMap<String, Arc<Vec<f32>>>>,
    rowsum_cache: RefCell<HashMap<String, Arc<Vec<f64>>>>,
}

impl<'a> IndicatorCtx<'a> {
    pub fn new(set: &'a BackupSet, prev: Option<PrevDay<'a>>) -> Self {
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
        self.set.date
    }

    /// 有向矩阵（S_dir[i][j] = i 领先 j）; 无则 None
    pub fn matrix(&self, name: &str) -> Option<&[f32]> {
        self.set.mats.get(name).map(|m| m.data())
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

    /// 对称矩阵的行和（缓存）
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

    /// 有向矩阵行和（OutStrength 方向: Σ_j S[i][j]）
    pub fn row_sum_dir(&self, name: &str) -> Option<Vec<f64>> {
        let d = self.matrix(name)?;
        let n = self.n();
        Some((0..n).map(|i| d[i * n..(i + 1) * n].iter().map(|&v| v as f64).sum()).collect())
    }

    /// 有向矩阵列和（InStrength 方向: Σ_i S[i][j]）
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

/// 加载日期备份集（供二进制入口用）
pub fn load_backup(outdir: &str, date: i64, load_mats: bool) -> std::io::Result<BackupSet> {
    BackupSet::load(Path::new(outdir), date, load_mats)
}
