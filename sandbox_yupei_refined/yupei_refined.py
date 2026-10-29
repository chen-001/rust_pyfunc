"""玉佩精炼版：单日探索入口。所有因子计算在独立二进制完成。

首次构建：在本目录执行 bash build.sh；不安装或修改 rust_pyfunc。
单日调用：df = compute_yupei_refined(20241231, daily_only=True, allow_uncomputable_corr=True)
有历史时：df = compute_yupei_refined(20241231, history=[...])
history 必须是该版本输出的 .snapshot.json。完整滚动需要前20个交易日，
不会自动补算；缺历史须显式 daily_only=True，相关列保持缺失。
每列的含义及验证结果见同目录 outputs 中的 XLSX。
"""
from pathlib import Path
import json
import subprocess
import tempfile


def compute_yupei_refined(date, *, jobs=50, daily_only=False, history=(), codes=None, output_dir=None, allow_uncomputable_corr=False):
    """返回以股票代码为索引的DataFrame；审计/耗时/快照路径放在df.attrs。

    codes仅用于显式小样本核对；省略时计算当天全部白名单股票。
    无法计算的诊断相关默认报错；显式allow_uncomputable_corr=True时警告并
    保留缺失和逐项审计，绝不把缺失当0。
    本函数只计算传入的一个日期，不提交任务、不回测。
    """
    import pandas as pd
    root = Path(__file__).resolve().parent
    binary = root / "target/release/yupei_refined"
    if not binary.is_file():
        raise FileNotFoundError(f"请先在 {root} 执行 bash build.sh")
    if not history and not daily_only:
        raise ValueError("请传入历史快照；单日探索显式设 daily_only=True，历史列将保留缺失")
    outdir = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix=f"yupei_{date}_", dir=root / "outputs" if (root / "outputs").is_dir() else None))
    outdir.mkdir(parents=True, exist_ok=True)
    output = outdir / f"{int(date)}.json"
    command = [str(binary), "--date", str(int(date)), "--jobs", str(int(jobs)), "--output", str(output)]
    if daily_only:
        command.append("--daily-only")
    if allow_uncomputable_corr:
        command.append("--allow-uncomputable-corr")
    for path in history:
        command += ["--history", str(Path(path).resolve())]
    if codes is not None:
        command += ["--codes", ",".join(str(code).zfill(6) for code in codes)]
    subprocess.run(command, check=True)
    data = json.loads(output.read_text())
    frame = pd.DataFrame({c["name"]: c["values"] for c in data.pop("columns")}, index=data.pop("codes"), dtype=float)
    frame.index.name = "code"
    frame.attrs.update(data, output_path=str(output))
    # Only label existing Rust outputs here; Python never synthesizes factors.
    diagnostic = [name for name in frame if "_diag_" in name or "_response_valid_minutes_" in name]
    controls = [name for name in frame if name.startswith("yr_control_")]
    frame.attrs["diagnostic_columns"] = diagnostic
    frame.attrs["control_columns"] = controls
    frame.attrs["factor_columns"] = [name for name in frame if name not in diagnostic and name not in controls]
    return frame
