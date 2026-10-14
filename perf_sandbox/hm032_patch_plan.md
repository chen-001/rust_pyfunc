# hm032 提速补丁（可回滚），未应用

**状态：未动正式库。** `src/okay_database/hm032.rs` 一个字没改。等陈冲冲说可以动再动。
**前置条件：hm32 那批任务（PID 2338178）必须已经跑完。** 原因见最后「重建的危险」。

## 一、改哪两个函数

只改 `src/okay_database/hm032.rs` 里两个函数：`haha` 和 `max_eigenvalue`（用量上）。
入口 `compute_stock`、`names()`、`compute_cross`、110 个因子名、输出顺序全部不动。

## 二、改动前（原文）

```rust
// 行 358-388
fn max_eigenvalue(m: &[f64], n: usize) -> (f64, Vec<f64>) {
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut new_v = vec![0.0f64; n];
    let mut temp = vec![0.0f64; n];
    matvec(m, n, &v, &mut temp);
    let mut eigenvalue = dot(&v, &temp);
    let max_iter = 30;
    let tolerance = 1e-4;
    for _ in 0..max_iter {
        let prev_eigenvalue = eigenvalue;
        matvec(m, n, &v, &mut new_v);
        let norm = dot(&new_v, &new_v).sqrt();
        if norm < 1e-5 { break; }
        for x in new_v.iter_mut() { *x /= norm; }
        matvec(m, n, &new_v, &mut temp);
        eigenvalue = dot(&new_v, &temp);
        std::mem::swap(&mut v, &mut new_v);
        let rel_error = (eigenvalue - prev_eigenvalue).abs();
        if rel_error < tolerance * eigenvalue.abs() { break; }
    }
    (eigenvalue, v)
}

// 行 504-521
fn haha(zp: &[f64], zv: &[f64]) -> [f64; PER_GROUP] {
    let n = zp.len();
    let mut m = vec![0.0f64; n * n];          // ← 全天 4828² × 8B = 186MB
    for i in 0..n {
        let pi = zp[i];
        let vi = zv[i];
        let row = &mut m[i * n..(i + 1) * n];
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = if i >= j { (vi - zv[j]).abs() } else { (pi - zp[j]).abs() };
        }
    }
    let (eigenvalue, v) = max_eigenvalue(&m, n);
    vec_stats(eigenvalue, &v)
}
```

## 三、改动后

`max_eigenvalue`、`matvec`、`vec_stats`、`standardize` 全部保留不动（回退路径还要用）。
新增下面这些东西，并把 `haha` 改成"先判非有限值、再选路"：

```rust
// ---- 新增 1：树状数组 ----
struct Fenwick { t: Vec<f64> }
impl Fenwick {
    fn new(n: usize) -> Self { Fenwick { t: vec![0.0; n + 1] } }
    #[inline]
    fn add(&mut self, pos: usize, v: f64) {
        let mut i = pos + 1;
        while i < self.t.len() { self.t[i] += v; i += i & i.wrapping_neg(); }
    }
    #[inline]
    fn prefix(&self, k: usize) -> f64 {
        let mut i = k; let mut s = 0.0;
        while i > 0 { s += self.t[i]; i -= i & i.wrapping_neg(); }
        s
    }
}

// ---- 新增 2：按数值取秩 ----
/// 返回 (rank[i], 升序数值表)。调用前必须保证 vals 里没有 NaN/inf。
fn ranks(vals: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap());
    let mut rank = vec![0usize; n];
    let mut sorted = vec![0.0f64; n];
    for (pos, &i) in idx.iter().enumerate() { rank[i] = pos; sorted[pos] = vals[i]; }
    (rank, sorted)
}

#[inline]
fn upper_bound(a: &[f64], q: f64) -> usize {
    let mut lo = 0usize; let mut hi = a.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if a[mid] <= q { lo = mid + 1 } else { hi = mid }
    }
    lo
}

// ---- 新增 3：不建矩阵的 M*x，O(n log n) ----
/// y[i] = Σ_{j<=i} |zv[i]-zv[j]|*x[j] + Σ_{j>i} |zp[i]-zp[j]|*x[j]
fn matvec_free(zp: &[f64], zv: &[f64], rp_: &[usize], rv: &[usize],
               sp: &[f64], sv: &[f64], x: &[f64], y: &mut [f64]) {
    let n = x.len();
    {
        let mut bit_x = Fenwick::new(n);
        let mut bit_vx = Fenwick::new(n);
        let (mut tot_x, mut tot_vx) = (0.0f64, 0.0f64);
        for i in 0..n {
            let q = zv[i];
            bit_x.add(rv[i], x[i]);
            bit_vx.add(rv[i], q * x[i]);
            tot_x += x[i];
            tot_vx += q * x[i];
            let k = upper_bound(sv, q);
            let lx = bit_x.prefix(k);
            let lvx = bit_vx.prefix(k);
            y[i] = q * lx - lvx + (tot_vx - lvx) - q * (tot_x - lx);
        }
    }
    {
        let mut bit_x = Fenwick::new(n);
        let mut bit_px = Fenwick::new(n);
        let (mut tot_x, mut tot_px) = (0.0f64, 0.0f64);
        for i in (0..n).rev() {
            if i + 1 < n {
                let j = i + 1;
                bit_x.add(rp_[j], x[j]);
                bit_px.add(rp_[j], zp[j] * x[j]);
                tot_x += x[j];
                tot_px += zp[j] * x[j];
            }
            let q = zp[i];
            let k = upper_bound(sp, q);
            let lx = bit_x.prefix(k);
            let lpx = bit_px.prefix(k);
            y[i] += q * lx - lpx + (tot_px - lpx) - q * (tot_x - lx);
        }
    }
}

// ---- 新增 4：幂迭代，逐句与 max_eigenvalue 相同，只把 matvec 换成 matvec_free ----
fn max_eigenvalue_free(zp: &[f64], zv: &[f64], n: usize) -> (f64, Vec<f64>) {
    let (rp_, sp) = ranks(zp);
    let (rv, sv) = ranks(zv);
    let mut v = vec![1.0f64 / (n as f64).sqrt(); n];
    let mut new_v = vec![0.0f64; n];
    let mut temp = vec![0.0f64; n];
    matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &v, &mut temp);
    let mut eigenvalue = dot(&v, &temp);
    for _ in 0..30 {
        let prev_eigenvalue = eigenvalue;
        matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &v, &mut new_v);
        let norm = dot(&new_v, &new_v).sqrt();
        if norm < 1e-5 { break; }
        for x in new_v.iter_mut() { *x /= norm; }
        matvec_free(zp, zv, &rp_, &rv, &sp, &sv, &new_v, &mut temp);
        eigenvalue = dot(&new_v, &temp);
        std::mem::swap(&mut v, &mut new_v);
        let rel_error = (eigenvalue - prev_eigenvalue).abs();
        if rel_error < 1e-4 * eigenvalue.abs() { break; }
    }
    (eigenvalue, v)
}

// ---- 新增 5：老函数改名，作为回退路径 ----
fn haha_dense(zp: &[f64], zv: &[f64]) -> [f64; PER_GROUP] { /* 原 haha 函数体，一个字不改 */ }

fn haha(zp: &[f64], zv: &[f64]) -> [f64; PER_GROUP] {
    // 有非有限值（NaN / ±inf）时走原来的稠密路径，结果与改动前逐位相同
    if zp.iter().any(|x| !x.is_finite()) || zv.iter().any(|x| !x.is_finite()) {
        return haha_dense(zp, zv);
    }
    let n = zp.len();
    if n == 0 { return [f64::NAN; PER_GROUP]; }
    let (eigenvalue, v) = max_eigenvalue_free(zp, zv, n);
    vec_stats(eigenvalue, &v)
}
```

**为什么要那行非有限值判断**：2.2%（157/7240，20250303 实测）的股票-日在某个时段里会出现
标准差为 0 导致的 ±inf。老代码那种情况下整段输出退化成 NaN；新算法里 `partial_cmp` 遇到 NaN 会
直接 panic，遇到 inf 排序还能走但结果不可比。所以这一档必须原路返回，行为逐位不变。

## 四、怎么应用 / 怎么退回

应用（**只在 hm32 跑完之后做**）：

1. 备份：`cp src/okay_database/hm032.rs perf_sandbox/hm032.rs.orig`
2. 按上面改 `src/okay_database/hm032.rs`
3. 重建：`cd /home/chenzongwei/rust_pyfunc && ./alter.sh`（会先 maturin develop，再编 worker 并部署）
4. 验证：先自己跑一遍 `perf_sandbox/hm32_bench/main.rs` 的对照，再叫复现守卫跑基准

退回（任选一种，都是分钟级）：

- `cd /home/chenzongwei/rust_pyfunc && git checkout -- src/okay_database/hm032.rs && ./alter.sh`
- 或者 `cp perf_sandbox/hm032.rs.orig src/okay_database/hm032.rs && ./alter.sh`

只改 .rs 不跑 `alter.sh` 的话，运行中的进程完全不受影响（`.so` 和 worker 二进制都没变）。

## 五、重建的危险（为什么要等 hm32 跑完）

`alter.sh` 会覆盖 `python/rust_pyfunc/rust_pyfunc_worker` 和 Python 扩展。
正在跑的 hm32 父进程会不断 fork 新的 worker，**新 fork 出来的 worker 会加载新实现**，
于是同一批 base 数据前半段用旧算法、后半段用新算法。虽然两者差 1e-15，
但这不是可以接受的状态，所以：**不跑完不许重建。**

## 六、验收标准

1. 复现守卫的基准：hm032 的 110 个 unfold 值，至少 8 组 (日期, 股票)，改前改后逐个相同。
2. 我自己已测的两组，共 12 组 (日期, 股票)，每组 110 个值全比（`verify_group2.py` 的结果）：

   | 比对 | 12 组里的最大绝对差 |
   |---|---|
   | 旧算法 vs 库输出 `okay_compute_stock` | 2.244e-04 |
   | 新算法 vs 库输出 | 2.244e-04 |
   | 旧算法 vs 新算法（f64 逐值） | 5.184e-11（值域到 5500，相对约 1e-14） |

   11 个正常样本里，"旧 vs 库"和"新 vs 库"的差值逐例完全相同，也就是差异全部来自
   库里把结果存成 f32 这一步，新算法没引入任何可见偏差。
   第 12 个是 000016@20250303（输入含非有限值），bench 里的 fast 分支只挡了 NaN 没挡 inf，
   那一例的 fast 输出是无效值——这正好证明上面那行 `is_finite` 判断是必须的。
3. 含非有限值的样本（000016@20250303）必须逐位相同（走回退路径）。
4. 性能：`perf_sandbox/hm32_bench` 对照，单只计算段 1327ms → 16.9ms（20 只抽样均值，
   测速条件：单进程、不开并行）。全系列按 200 并行折算：33 小时 → 约 0.5 ~ 1.2 小时。

## 七、实测代码与原始输出

- `perf_sandbox/hm32_bench/main.rs` —— 分阶段计时 + 新旧两版对照 + `dump` 模式出 110 个值
- `perf_sandbox/hm32_bench/verify_group2.py` —— 第二组 12 个 (日期, 股票) 的三向比对
- `perf_sandbox/hm32_bench/strat_raw.txt` —— 全市场 20 只抽样的原始输出
- `perf_sandbox/hm32_slow_rootcause.md` —— 根因报告
