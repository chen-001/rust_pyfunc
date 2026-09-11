//! store8：让沙箱直接读真实 colblk 库（factor_store_yupei_dist）。
//!
//! 生产路径 `factor_store_v5::read_factor_into_fast` 的投影区分支是：
//!   1. 按 (off, csz) 从 factors.proj 一次 pread 出该因子的 f32 值段；
//!   2. 用共享 row_order 的第 r 行拿到 (date_id, code_id)；
//!   3. 映射到模板矩阵的 (row, col) 后写入。
//! 其中第 2、3 步是纯查表，与因子无关 —— 所以 Python 侧
//! （`dump_store_meta.py`）已把它折叠成一个扁平散点下标 `tgt[r] = row*N + col`。
//! 本模块只做「pread + 散点写」，与生产逐位同构。
//!
//! 元数据：/home/chenzongwei/neu_lab/store_meta/{meta.json, shard_<i>.bin}

use std::fs::{self, File};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use ndarray::Array2;

const MAGIC: u32 = 0x5338_444D; // "S8DM"

struct Shard {
    file: File,
    /// 每因子的 (proj 内偏移, 字节数)
    val_index: Vec<(u64, u64)>,
    /// 本片每条记录在模板扁平矩阵里的下标 row*N+col
    tgt: Vec<u32>,
}

pub struct Store8 {
    pub t: usize,
    pub n: usize,
    pub factor_names: Vec<String>,
    store_dir: PathBuf,
    shards: Vec<Shard>,
}

impl Store8 {
    /// `meta_dir`：dump_store_meta.py 的产物目录；`store_dir`：colblk 库根目录。
    pub fn open(meta_dir: &str, store_dir: &str) -> Result<Self, String> {
        let meta = fs::read_to_string(format!("{meta_dir}/meta.json"))
            .map_err(|e| format!("读 meta.json 失败: {e}"))?;
        let t: usize = json_get(&meta, "\"T\":").ok_or("meta.json 缺 T")?;
        let n: usize = json_get(&meta, "\"N\":").ok_or("meta.json 缺 N")?;
        let n_shards: usize = json_get(&meta, "\"n_shards\":").ok_or("meta.json 缺 n_shards")?;
        let factor_names = json_str_array(&meta, "\"factor_names\":").ok_or("meta.json 缺 factor_names")?;

        let store = PathBuf::from(store_dir);
        let mut shards = Vec::with_capacity(n_shards);
        for i in 0..n_shards {
            let p = format!("{meta_dir}/shard_{i}.bin");
            let bytes = fs::read(&p).map_err(|e| format!("读 {p} 失败: {e}"))?;
            let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
            if magic != MAGIC {
                return Err(format!("{p} magic 不匹配: {magic:#x}"));
            }
            let _tt = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
            let _nn = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
            let fc = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
            let n_rows = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
            let mut pos = 24usize;
            let mut val_index = Vec::with_capacity(fc);
            for _ in 0..fc {
                let off = u64::from_le_bytes(bytes[pos..pos + 8].try_into().unwrap());
                let csz = u64::from_le_bytes(bytes[pos + 8..pos + 16].try_into().unwrap());
                val_index.push((off, csz));
                pos += 16;
            }
            let tgt = bytes[pos..pos + n_rows * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect::<Vec<u32>>();
            let proj_path = store.join(format!("shard_{i}")).join("factors.proj");
            let file = File::open(&proj_path).map_err(|e| format!("打开 {proj_path:?} 失败: {e}"))?;
            shards.push(Shard { file, val_index, tgt });
        }
        if factor_names.len() != shards[0].val_index.len() {
            return Err(format!(
                "因子名 {} 个与分片索引 {} 个不一致",
                factor_names.len(),
                shards[0].val_index.len()
            ));
        }
        Ok(Self { t, n, factor_names, store_dir: store, shards })
    }

    pub fn n_factors(&self) -> usize {
        self.factor_names.len()
    }

    /// 读第 col 个因子为 (T, N) 模板矩阵（跨 8 个分片累加，与生产同构）。
    pub fn read_factor(&self, col: usize) -> Result<Array2<f32>, String> {
        let mut out = Array2::<f32>::from_elem((self.t, self.n), f32::NAN);
        {
            let flat = out.as_slice_mut().expect("模板矩阵必须连续");
            let mut buf: Vec<u8> = Vec::new();
            for (si, sh) in self.shards.iter().enumerate() {
                let (off, csz) = sh.val_index[col];
                let len = csz as usize;
                if buf.len() < len {
                    buf.resize(len, 0);
                }
                let got = sh
                    .file
                    .read_at(&mut buf[..len], off)
                    .map_err(|e| format!("shard_{si} pread 失败: {e}"))?;
                if got != len {
                    return Err(format!("shard_{si} pread 短读 {got}/{len}"));
                }
                let n_rows = len / 4;
                for r in 0..n_rows {
                    let v = f32::from_le_bytes(buf[r * 4..r * 4 + 4].try_into().unwrap());
                    // 与生产一致：非有限值统一写成 f32::NAN
                    flat[sh.tgt[r] as usize] = if v.is_finite() { v } else { f32::NAN };
                }
            }
        }
        Ok(out)
    }

    pub fn store_dir(&self) -> &PathBuf {
        &self.store_dir
    }
}

// ---- 极简 JSON 取值（meta.json 由我们自己生成，格式固定，不引 serde）----
fn json_get<T: std::str::FromStr>(s: &str, key: &str) -> Option<T> {
    // key 形如 "\"T\":" 或 "\"T\": "（json.dumps 默认带空格）
    let i = s.find(key)? + key.len();
    let rest = &s[i..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn json_str_array(s: &str, key: &str) -> Option<Vec<String>> {
    let i = s.find(key)? + key.len();
    let start = s[i..].find('[')? + i + 1;
    let end = s[start..].find(']')? + start;
    Some(
        s[start..end]
            .split(',')
            .map(|x| x.trim().trim_matches('"').to_string())
            .filter(|x| !x.is_empty())
            .collect(),
    )
}
