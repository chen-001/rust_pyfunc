"""在 lib.rs / factor_pipeline.rs / worker_pipeline.rs / __init__.pyi 里注册基因动态规划四个横截面 pipeline。
apply(path, text) 只做插入，锚点在 HEAD 版本和工作区版本里都存在，便于只提交本次改动的部分。"""
import sys
SERIES = [
    ("genedp_day_vs_day_align", "基因动态规划·今天对昨天（42,924 列）"),
    ("genedp_self_repeat_align", "基因动态规划·自比对找重复（30,528 列）"),
    ("genedp_mirror_align", "基因动态规划·镜像比对（16,917 列）"),
    ("genedp_buysell_fold", "基因动态规划·买卖配对折叠（249,603 列）"),
]

def ins_after(t, anchor, add):
    assert t.count(anchor) == 1, anchor
    return t.replace(anchor, anchor + add)

def ins_before(t, anchor, add):
    assert t.count(anchor) == 1, anchor
    return t.replace(anchor, add + anchor)

def lib(t):
    t = ins_after(t, "pub mod anneal_cross_peer_v3_metrics;\n",
        "pub mod genedp_common;\n" + "".join(f"pub mod {s};\n" for s, _ in SERIES))
    anchor = """    let _ = m.add_function(wrap_pyfunction!(
        anneal_cross_peer_v3_metrics::py_anneal_cross_peer_v3_names,
        m
    )?)?;
"""
    add = ""
    for s, _ in SERIES:
        for f in (f"py_{s}", f"py_{s}_names"):
            add += f"""    let _ = m.add_function(wrap_pyfunction!(
        {s}::{f},
        m
    )?)?;
"""
    return ins_after(t, anchor, add)

def pipe(t):
    add = ""
    for s, desc in SERIES:
        add += f'''/// {desc}：读当天（和前一交易日）全市场原始数据，一次一天。
pub fn pipeline_{s}(date: i64, expected_len: usize) -> Vec<TaskResult> {{
    let n = crate::{s}::{s}_names().len();
    if expected_len != n {{
        eprintln!("pipeline_{s} expected_len错误 [{{date}}]: {{expected_len}} != {{n}}");
        return Vec::new();
    }}
    match crate::{s}::compute_{s}_full(date) {{
        Ok((codes, vals)) => vals
            .chunks(expected_len)
            .zip(codes.iter())
            .map(|(facs, code)| TaskResult {{
                date,
                code: code.clone(),
                timestamp: 0,
                facs: facs.to_vec(),
            }})
            .collect(),
        Err(e) => {{
            eprintln!("pipeline_{s} error [{{date}}]: {{e:?}}");
            Vec::new()
        }}
    }}
}}

'''
    t = ins_before(t, "/// 柜员游戏扩展系列3：全市场横截面联动与同伴可还原性（anneal_cross_peer，11760 列）。\n", add)
    t = ins_after(t, '        "anneal_cross_peer_v3",\n', "".join(f'        "{s}",\n' for s, _ in SERIES))
    return t

def worker(t):
    t = ins_after(t, "    pipeline_anneal_cross_peer_v3, pipeline_anneal_cross_peer_v3_profile,\n",
        "    " + ", ".join(f"pipeline_{s}" for s, _ in SERIES) + ",\n")
    anchor = """                    } else if pipeline_name == "anneal_cross_peer_v3" {
                        pipeline_anneal_cross_peer_v3(date, expected_len)
"""
    add = "".join(f"""                    }} else if pipeline_name == "{s}" {{
                        pipeline_{s}(date, expected_len)
""" for s, _ in SERIES)
    return ins_after(t, anchor, add)

def pyi(t):
    t = ins_after(t, '    "py_anneal_cross_peer_v3_profile_names",\n',
        "".join(f'    "py_{s}",\n    "py_{s}_names",\n' for s, _ in SERIES))
    add = "\n"
    for s, desc in SERIES:
        add += f'''
def py_{s}(date: int) -> Tuple[List[str], List[float]]:
    """{desc}。单日全市场（50 线程），返回 (股票代码, 展平因子值)，每股 len(py_{s}_names()) 列。
    依赖撤单数据的列名含 cxl，撤单数据缺失的日子为空值。"""
    ...

def py_{s}_names() -> List[str]:
    """py_{s} 的列名（与输出逐列对齐）。"""
    ...
'''
    return t.rstrip("\n") + "\n" + add

FILES = {"src/lib.rs": lib, "src/factor_pipeline.rs": pipe, "src/bin/worker_pipeline.rs": worker,
         "python/rust_pyfunc/__init__.pyi": pyi}

if __name__ == "__main__":
    for path, fn in FILES.items():
        p = "/home/chenzongwei/rust_pyfunc/" + path
        t = open(p).read()
        open(p, "w").write(fn(t))
        print("ok", path)


# ---- 结果消息超过 4 GB 时的长度前缀扩展（买卖配对折叠一天约 5.1 GB）----
IPC_W_OLD = '''    let data =
        bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    let len = data.len() as u32;
    writer.write_all(&len.to_le_bytes())?;
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

/// 读一个 IPC 消息'''
IPC_W_NEW = '''    let data =
        bincode::serialize(msg).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    // 不足 4 GB：4 字节长度（与历史格式逐字节相同）。
    // 4 GB 及以上（如基因动态规划·买卖配对折叠一天约 5.1 GB）：先写 u32::MAX 作标记，再写 8 字节长度。
    if data.len() < u32::MAX as usize {
        writer.write_all(&(data.len() as u32).to_le_bytes())?;
    } else {
        writer.write_all(&u32::MAX.to_le_bytes())?;
        writer.write_all(&(data.len() as u64).to_le_bytes())?;
    }
    writer.write_all(&data)?;
    writer.flush()?;
    Ok(())
}

/// 读一个 IPC 消息'''
IPC_R_OLD = '''pub fn ipc_read_result<R: std::io::Read>(reader: &mut R) -> std::io::Result<ResultMessage> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let len = u32::from_le_bytes(len_bytes) as usize;
'''
IPC_R_NEW = '''pub fn ipc_read_result<R: std::io::Read>(reader: &mut R) -> std::io::Result<ResultMessage> {
    let mut len_bytes = [0u8; 4];
    reader.read_exact(&mut len_bytes)?;
    let mut len = u32::from_le_bytes(len_bytes) as usize;
    if len == u32::MAX as usize {
        // 4 GB 及以上的结果：标记之后是 8 字节真实长度（见 ipc_write_result）
        let mut long_bytes = [0u8; 8];
        reader.read_exact(&mut long_bytes)?;
        len = u64::from_le_bytes(long_bytes) as usize;
    }
'''

def ipc(t):
    assert t.count(IPC_W_OLD) == 1 and t.count(IPC_R_OLD) == 1
    return t.replace(IPC_W_OLD, IPC_W_NEW).replace(IPC_R_OLD, IPC_R_NEW)

_pipe_reg = pipe
def pipe_all(t):
    return ipc(_pipe_reg(t))
FILES["src/factor_pipeline.rs"] = pipe_all
