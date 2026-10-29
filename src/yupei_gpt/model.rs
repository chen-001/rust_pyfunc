use serde::{Deserialize, Serialize};
/// 上午09:30-11:30，下午13:00-14:57；开闭市竞价不纳入。
/// slot按真实钟表10ms偏移，午休不会被压成相邻事件。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Cell {
    pub slot: u32,
    pub buy: f32,
    pub sell: f32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stock {
    pub code: String,
    pub industry: i32,
    pub cells: Vec<Cell>,
    pub n_buy: usize,
    pub n_sell: usize,
    pub amount: f64,
    /// 连续交易237个分钟格；两个时段分隔在index120。
    pub minute_buy: Vec<f64>,
    pub minute_sell: Vec<f64>,
    /// 分钟内末笔/首笔对数收益；缺成交为NaN，单笔为0。
    pub minute_ret: Vec<f64>,
    /// 分钟末笔价格；无成交为NaN，不做前向填充。
    pub minute_close: Vec<f64>,
}
#[derive(Clone, Debug)]
pub struct Kernel {
    pub n: usize,
    /// 行先动/列后动；下标[先动方向*2+后动方向]，方向买0卖1。
    pub lead: [Vec<f32>; 4],
    /// direction: 全部0, 主买1, 主卖2；无自能量时为0，仅代表活动量为0。
    pub energy: [Vec<f64>; 3],
    /// rho[a*3+b]，双向指数核内积的正确能量归一化；无能量时NaN。
    pub rho: [Vec<f32>; 9],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FactorColumn {
    pub name: String,
    pub values: Vec<Option<f64>>,
}
impl FactorColumn {
    pub fn new(name: impl Into<String>, values: Vec<f64>) -> Self {
        Self {
            name: name.into(),
            values: values
                .into_iter()
                .map(|v| v.is_finite().then_some(v))
                .collect(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NeighborState {
    pub chain: String,
    /// 能量和配对集合足以计算该行；空伙伴不自动代表无效。
    pub valid: Vec<bool>,
    /// 第i行自身作为指定方向配对成员的正GS最强100伙伴。代码用于跨日对齐。
    pub neighbors: Vec<Vec<String>>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Snapshot {
    pub schema: String,
    pub config_hash: String,
    pub date: i64,
    pub codes: Vec<String>,
    pub columns: Vec<FactorColumn>,
    pub neighbors: Vec<NeighborState>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditRow {
    pub name: String,
    pub layer: String,
    pub n: usize,
    pub value: Option<f64>,
    pub detail: String,
}
#[derive(Clone, Debug)]
pub struct NetworkOutput {
    pub columns: Vec<FactorColumn>,
    pub neighbors: NeighborState,
    pub audits: Vec<AuditRow>,
}
#[derive(Clone, Debug)]
pub struct Config {
    pub date: i64,
    pub level2_root: std::path::PathBuf,
    pub basic_info_dir: std::path::PathBuf,
    pub vars_dir: std::path::PathBuf,
    /// 历史快照目录：temporal 阶段从中按日期取 .ygsnap.json。
    pub snap_dir: Option<std::path::PathBuf>,
    pub allow_incomplete_history: bool,
    /// 显式允许诊断相关无定义；保留null并警告，绝不当0参与比较。
    pub allow_uncomputable_corr: bool,
    /// 显式测试股票子集，None=全部白名单。不得按行号隐式截断。
    pub codes: Option<Vec<String>>,
    pub chains: Vec<String>,
}
