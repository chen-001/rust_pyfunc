"""hm0~hm92 逐系列改写的作业规范（给并行 agent 看的唯一说明，写在代码注释里）。

════════════════════════════════════════════════════════════════════════
一、这个作业要产出什么
════════════════════════════════════════════════════════════════════════

对旧 okay 文件（/home/chenzongwei/pythoncode/成果hmXX_*/hmXX_okay.py，用旧框架
design_whatever_out 写的）做 Rust 复现，产出两个文件：

  1. src/okay_database/hmXXX.rs
     提供三个函数（名字固定，注册表生成器按这三个名字调用）：
         pub fn names() -> Vec<String>
         pub fn compute_stock(code: &str, date: i64) -> Option<Vec<f32>>   // 逐股系列用
         pub fn compute_cross(date: i64) -> Option<(Vec<String>, Vec<f32>)> // 横截面/分钟用
     不适用的那个返回 None。逐股系列的输出必须与旧代码 names_single 的顺序逐个对齐。

  2. <旧 okay 文件所在目录>/hmXX_okay_database.py
     照抄 /home/chenzongwei/rust_pyfunc/lib/okay_database/hm29_okay_database.py 的形状：
     四个入口 run_base / run_db / run_h5 / run_h1，路径指向因子库
     /nas197/user_home/chenzongwei/factor_library 下的 factor_base / dailybank / hf_data / hf_data1。
     pipeline_groups 里 pipeline='okay_hmXXX'，names 用 rp.okay_series_names('hmXXX') 取。

严禁改动现有的 hmXX_okay.py；严禁改别的系列的文件。

════════════════════════════════════════════════════════════════════════
二、旧框架的语义（必须逐条对上，这是最容易踩的地方）
════════════════════════════════════════════════════════════════════════

旧框架在 /home/chenzongwei/pythoncode/design_whatever_out.py。要点：

【数据切分】get_factor() 里的 self.read_trade_data_single() 按 get_many 返回若干段：
    get_many=1 → [allday, morning, afternoon]
    get_many=2 → [allday, open30, morning, afternoon, close30]
    其余（get_many=0 且 read_full=0）→ 单个 DataFrame，旧代码自己再切窗口
  段顺序必须与 names 的拼接顺序一致（自己读旧代码确认 names1..names5 的拼接顺序）。

【撤单过滤】read_trade_data_single(with_retreat=0) 会丢掉 flag==32 的行。
  ！！Rust 侧的 with_retreat 参数不可靠：库里那条判断是按字节比 "32"，
  而 2025 年前后成交明细里撤单标记写的是 "32.0"，过滤不生效。
  所以一律用 read_trade_fast_inner(code, date, true, false, usize::MAX) 读进来，
  然后自己 .filter(|r| r.flag != 32)。

【时间】逐笔记录里 TradeRecord.time_us 是 epoch 微秒，已经加了 8 小时（北京时区）。
  当天的钟点秒 = time_us.rem_euclid(86_400_000_000)。
  旧代码的 exchtime 是 pandas Timestamp，between_time 是**精确到微秒的闭区间**，
  不能按整秒截断（否则 14:57:00.3 这种会被错误纳入）。

【下午平移】只有旧代码显式调了 adjust_afternoon 时才用 with_afternoon_adjust=true。
  只在真实钟点上切窗口的（比如 hm29 的 09:30~14:57）必须用 false。

【get_factor 的返回值】返回一个扁平 list，长度 = 每个时段的指标数 × 时段数，
  顺序就是 names_single 的顺序。Rust 侧输出顺序必须完全一致。

【横截面】旧代码里出现 dwo.de_cross / 全市场 rank 之类的，整段要放到 compute_cross，
  用 kind='cross_section' 的 pipeline_groups（一次算一天全市场，返回 (codes, 扁平值)）。

【分钟数据】继承 dwo.FactorDesign、用 new_get_field_min_data 取字段的，属于分钟系列，
  旧代码的调用是 Go(base_dir, date, date, ...)，也就是一天一次、返回全市场的一张表。
  这类走 compute_cross。minute 数据的读法见 rust-pipeline-minute skill。

【pandas 语义对照】
    df.corr()        → Pearson，成对完整观测（任一 NaN 丢掉这一对）；样本<2 或某列方差为 0 返回 NaN
    df.rank()        → 默认 method='average'，升序，并列取平均名次，NaN 保持 NaN
    df.quantile(q)   → 默认线性插值（(n-1)*q 位置）
    df.groupby(...)  → 分组聚合；注意空行不计入
    between_time 两端都含

════════════════════════════════════════════════════════════════════════
三、开发流程（每一步都要做，不许跳）
════════════════════════════════════════════════════════════════════════

1. 读旧 okay 文件，写清楚：get_many / read_full / change_time、段顺序、
   每个时段的指标算式、names_single 的顺序、names_save（要保存的 base 名）。
   把这些先以注释形式写在 hmXXX.rs 顶上，方便复核。

2. 写 src/okay_database/hmXXX.rs。

3. 构建（必须走这个脚本，它带串行锁，多人同时构建会排队）：
       bash /home/chenzongwei/rust_pyfunc/lib/okay_database/build_locked.sh

4. 对照验证（这是硬门槛，不通过不算做完）：
       /home/chenzongwei/.conda/envs/chenzongwei311/bin/python \
         /home/chenzongwei/rust_pyfunc/lib/okay_database/verify_series.py \
         --series hmXXX --okay "<旧 okay 文件绝对路径>" \
         --pairs "20250303:000001,20250304:600519,20180601:000002,20231020:300750"
   要求：同一段数据上「旧 50 值 / 新 50 值，一致」，结论「通过」。
   不一致就回去改，直到通过。至少覆盖 4 组 (日期, 股票)，包含一个 2018 年附近的日期。

5. 写 hmXX_okay_database.py，参数（ver、入选名单）从旧 okay 文件里原样搬过来。

6. 只做小样本冒烟（一天、n_jobs 不要超过 8），确认能写出 base。**不要**跑全历史，
   全历史由队列统一串行提交。

════════════════════════════════════════════════════════════════════════
四、边界与纪律
════════════════════════════════════════════════════════════════════════

- 因子个数恒定：算不出来的股票返回 None（整条 NaN），不要条件跳过、不要少输出几列。
- 只改自己的两个文件。构建走带锁脚本。不要动 factor_taskd、不要提交任务系统。
- 不要改 python/rust_pyfunc/ 下的生产代码（除非确有必要，且要先说明）。
- 遇到旧代码里有看不懂的调用，去看 design_whatever_out.py 里对应的实现，不要猜。
- 汇报时说清楚：验证跑了哪几组 (日期,股票)、结论、还有哪些没覆盖到的分支。

════════════════════════════════════════════════════════════════════════
五、开工前必查的三件事（都是别人踩过的）
════════════════════════════════════════════════════════════════════════

1. 旧 names_single 里可能有重名。列式 store 的组合校验不允许因子名重复，
   一旦 names() 里出现重名，单日冒烟那一步就会报「组合 store 因子名重复」。
   处理办法：第二次及以后出现的重名加 `_dup2` / `_dup3` 后缀（值不变、列数不变），
   并在文件头写明。先在旧文件里搜一遍 names_single 有没有重复项。

2. worker 链路里不能调用 #[pyfunction] 标记的函数。worker 二进制不带 libpython，
   链路上碰到 PyResult/PyErr 会让 worker 链接失败，而构建表面还是「成功」，
   只有 .so 更新、worker 不更新，后面所有人跑 pipeline 都会拿到错值而不报错。
   典型是 crate::time_series::dtw_distance。需要 DTW 就在自己文件里写纯 Rust 实现
   （算法照 src/time_series/mod.rs 抄）。

3. 半成品要临时离开，就在文件第一行加 `// REGISTRY_SKIP`，注册表会跳过它，
   构建不受影响；改完删掉那一行再构建。绝对不要留一个编不过的文件，
   那会把所有人的构建锁住。

另外两个已知的旧表达式坑：

- dwo.get_abs(np.log(fac))：fac 为 0 时 log 给 -inf，截面均值被污染成 -inf，整列变 inf。
  按「非有限 log 抹成 NaN、均值只在有限值上求」处理，并把偏差写进文件头。
- 依赖 numpy/polars 排序并列顺序的取值（argsort、sort_values、多列 sort、unique 输出顺序）：
  本机 numpy 对 float64 走 AVX-512 位网络，polars 多线程下顺序不定，旧代码自己跨机器/跨进程
  就不可复现。能按源码标量复刻就复刻（见 hm039.rs），复刻不了就按已知不可复现处理，
  把实测数字和影响面写清，并确认这些量不在 names_save 里。


════════════════════════════════════════════════════════════════════════
六、数值求和与超越函数：库函数不能随便替
════════════════════════════════════════════════════════════════════════

1. pandas 的 DataFrame.sum() / mean() 用的是 numpy 的**成对求和**（PW_BLOCKSIZE=128、
   8 路展开），不是顺序累加；pandas 的 groupby().sum() 对 float64 又用 Kahan 补偿求和；
   而 polars 的 sum().over(group) 是顺序 f32 累加、全局 sum() 是成对求和。
   这四种混用会差 1 ulp，看不出问题，但只要下游有「比阈值」「取分位」「排名」这类操作，
   1 ulp 就能把掩码翻面，因子值差几十个百分点。
   凡是「先求和/求均值，再拿结果比阈值、排名、取分位」的系列，必须照 pandas/numpy 那一套实现。

2. numpy 的 np.log 不是 glibc 的 log：numpy wheel 里带 Intel SVML（__svml_log8/log4/log2），
   float64 的 np.log 在支持 AVX-512 的机器上走 AVX-512 汇编，和 Rust 的 f64::ln 有
   0.01%~0.2% 的输入差 1 ulp。想让两者逐位一致，只能把 SVML 用 intrinsic 翻成 Rust，
   硬依赖 AVX-512，且 numpy 自己换 CPU 也会换路径。所以这类差异按「已知不可复现」处理，
   把实测数字和影响面写进文件头即可，不要去翻 SVML。


════════════════════════════════════════════════════════════════════════
七、第三方库的实现细节（照语义写会错的那种）
════════════════════════════════════════════════════════════════════════

- scipy 的 stats.f.cdf 走的是 cephes 的 fdtr/incbet，**不是** special.betainc。
  尾巴 ≤1.11e-16 时 cephes 把它钉成 1-1.11e-16，所以 p 值的下限是 1.11e-16 而不是 0。
  照抄 betainc 会差 1.11e-4，刚好越过 1e-4 验收线。有 p 值 / Chow 检验 / 方差分析的系列都要注意。
- pandas 的 `(a - a.mean()) / a.std()` 混着字符串列时，单调用会抛 TypeError，
  但在 groupby.apply 里会自动改用 include_groups=False 重跑，结果与 pandas 1.x 的 numeric_only 语义一致。
  别看到 TypeError 就以为旧代码坏了，先跑全量比数。
- dwo.get_abs 是 |x − 当日截面均值|（df.T.mean() 是对日期求均值），和因子库
  factor_production.write_selected_base 里的 _fold 实现一致，不需要自己另算。


- 凡是分母里含「两个大数相减」或近奇异设计矩阵的量（相关系数、回归 r2、各种比值），
  f64 顺序求和会把末位差异放大成倍数级。对照实例：hm038 第一版用自己写的顺序求和 + 手写 X'X 循环，
  r2 那一族每对差 26 处、相对误差到 98%（集合竞价格内成交价完全相同 → 标准化后 y 是常数 →
  ss_tot 只剩 1ulp 舍入，r2 落到 -1e6 量级）。
  修法两条：标准化的均值与平方偏差和照抄 numpy 的 pairwise（PW_BLOCKSIZE=128、8 路展开）；
  回归照抄 rp.ols（ndarray 的 .dot() 算 X'X/X'y，再照抄 solve_linear_system3 的 LU），不要自己写循环。

"""

# 本文件只是规范文本，不参与运行。
