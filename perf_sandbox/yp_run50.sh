#!/bin/bash
# 50 并行那一趟：沙箱里改后的库 + 上游分段计时；跑默认 15 张，出时间 + 全量逐位比对
cd /home/chenzongwei/rust_pyfunc/perf_sandbox
YUPEI_PROF=1 PYTHONPATH=/home/chenzongwei/rust_pyfunc/perf_sandbox/yp_pkg \
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python -u yp_run50.py > yp_run50.log 2>&1
grep -E '\[prof\]|FIFTY_DEFAULT_SECONDS|一致|逐位' yp_run50.log
