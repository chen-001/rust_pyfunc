"""hm94 生产化验收与入库：把 hmokay4 的第 2~4 步串起来交给任务系统跑。

为什么需要这层包装：hmokay4 的完整入口是 others/hmokay4/hmokay4.py，它依次跑
  第 2 步 prepare     本机历史验收（把研究库原值搬进生产 colblk + 抽样日独立重算核对 + 质量表 + tail_two）
  第 3 步 postprocess 生成 h5（factor_base / hf_data / hf_data1）并回填 names_in_db
  第 4 步 dailybank   把已验收的名单写进 dailybank
这三步都是长活，走任务系统才能在看板里看进度、能调并行、也不怕会话结束。

本文件不含任何因子计算代码，真正干活的是被调起的 hmokay4。
并行度：--local-jobs 取任务系统注入的 FACTOR_N_JOBS；--n-jobs 固定 30，这是交付脚本
自己的上限（同事的生产机最多 30 个 CPU）。
"""
import os
import subprocess
import sys

PKG = "/home/chenzongwei/pythoncode/others/hmokay4"
PY = "/home/chenzongwei/.conda/envs/chenzongwei311/bin/python"
TARGET = "/home/chenzongwei/pythoncode/成果hm94_挂单猫/hm94.py"

local_jobs = os.environ.get("FACTOR_N_JOBS", "200")
cmd = [PY, "-u", "hmokay4.py", TARGET, "--steps", "2,3,4",
       "--n-jobs", "30", "--local-jobs", str(local_jobs)]
print("运行:", " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, cwd=PKG).returncode)
