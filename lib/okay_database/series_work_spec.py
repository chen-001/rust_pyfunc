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
"""

# 本文件只是规范文本，不参与运行。
