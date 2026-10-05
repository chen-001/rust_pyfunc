#!/bin/bash
# 带串行锁的构建入口：多个 agent 同时改 Rust 时，构建排队，不要各跑各的。
#
# 用法：bash lib/okay_database/build_locked.sh
#
# 为什么不用 `exec 9>lock; flock 9`：那样 fd 会被 cargo 拉起的 sccache 守护进程继承，
# sccache 常驻不退出，flock 就一直不释放，后面所有构建全卡死。
# 这里用 flock -o（--close）：执行命令前先关掉锁 fd，锁由 flock 进程自己持有，
# cargo/sccache 拿不到这个 fd，命令一结束锁就释放。
set -e
cd /home/chenzongwei/rust_pyfunc
export PATH="/home/chenzongwei/.cargo/bin:/home/chenzongwei/.local/bin:$PATH"
export CONDA_PREFIX=/home/chenzongwei/.conda/envs/chenzongwei311

echo "[构建] 等待构建锁 ..."
flock -o -x -w 3600 /tmp/okay_database_build.lock -c '
  cd /home/chenzongwei/rust_pyfunc
  export PATH="/home/chenzongwei/.cargo/bin:/home/chenzongwei/.local/bin:$PATH"
  export CONDA_PREFIX=/home/chenzongwei/.conda/envs/chenzongwei311
  echo "[构建] 拿到锁，开始构建"
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python lib/okay_database/gen_registry.py
  timeout 900 ./alter.sh 2>&1 | grep -E "^error|error\[|Finished|💥|✅" | tail -20
'
echo "[构建] 结束"
