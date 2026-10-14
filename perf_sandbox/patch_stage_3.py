import re
# ---- compute.rs ----
p='/home/chenzongwei/rust_pyfunc/src/yupei_dist/compute.rs'
s=open(p).read(); done=[]
def rep(f,old,new,tag):
    global s
    if new in s: done.append('SKIP '+tag); return
    assert old in s, 'MISS '+tag
    s=s.replace(old,new,1); done.append('OK '+tag)
rep(p,"""pub fn compute_yupei_dist_partial(
    date: i64,
) -> std::io::Result<(""","""pub fn compute_yupei_dist_partial(
    date: i64,
) -> std::io::Result<(""", 'sig-check')
old_sig = """pub fn compute_yupei_dist_partial(
    date: i64,
) -> std::io::Result<("""
new_sig = """/// need = 需要计算的权重家族位（matrix_stage::FAM_*）。只算请求的家族，
/// 输出里未被请求的矩阵会是全 0 —— 调用方必须保证 need 覆盖它要用的矩阵。
pub fn compute_yupei_dist_partial_need(
    date: i64,
    need: u8,
) -> std::io::Result<("""
rep(p, old_sig, new_sig, 'need sig')
rep(p, "    let (mats, stats) = compute_matrices(&stocks);",
       "    let (mats, stats) = compute_matrices(&stocks, need);", 'cm call')
# 旧入口变薄包装，插在 _need 之前
rep(p, "/// need = 需要计算的权重家族位",
"""/// 全家族版本（旧入口）：py_yupei_dist 用得上全部 37 张。
pub fn compute_yupei_dist_partial(
    date: i64,
) -> std::io::Result<(
    Vec<String>,
    usize,
    Vec<crate::yupei_dist::matrix_stage::StockStats>,
    HashMap<String, Vec<f32>>,
    Vec<i16>,
)> {
    compute_yupei_dist_partial_need(date, crate::yupei_dist::matrix_stage::FAM_ALL)
}

/// need = 需要计算的权重家族位""", 'old wrapper')
open(p,'w').write(s)
print('compute.rs:', done)

# ---- mod.rs ----
q='/home/chenzongwei/rust_pyfunc/src/yupei_pair/mod.rs'
t=open(q).read(); done2=[]
old = """    let (codes, n, stats, hm, industry) =
        crate::yupei_dist::compute::compute_yupei_dist_partial(date)?;"""
new = """    // 只算请求的矩阵用到的权重家族。默认 15 张只要 cnt+vol，其余 22 张不参与计算。
    let need = {
        let m = crate::yupei_dist::matrix_stage::family_mask_of(mats);
        if m == 0 { crate::yupei_dist::matrix_stage::FAM_ALL } else { m }
    };
    let (codes, n, stats, hm, industry) =
        crate::yupei_dist::compute::compute_yupei_dist_partial_need(date, need)?;"""
assert old in t, 'MISS mod call'
t = t.replace(old, new, 1)
open(q,'w').write(t)
print('mod.rs: OK')
