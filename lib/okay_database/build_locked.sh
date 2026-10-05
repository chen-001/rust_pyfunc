#!/bin/bash
# 带串行锁的构建入口：多个 agent 同时改 Rust 时，构建排队。
#
# 用法：bash lib/okay_database/build_locked.sh
#
# 两个坑都在这里规避掉了：
#  1) 用 flock -o（--close）：执行命令前先关掉锁 fd，cargo/sccache 拿不到它，
#     否则 sccache 常驻会把锁一直占着。
#  2) alter.sh 的输出先落文件再看，不要接管道。接了管道以后，如果 cargo/rustc 被
#     timeout 杀掉但留下孤儿进程，它们继承的管道写端不会关闭，整条管道等不到 EOF，
#     flock 就一直不释放，全队构建全卡死（2026-10-05 实际发生过一次，卡了 1 小时 12 分）。
set -e
cd /home/chenzongwei/rust_pyfunc
LOG=/tmp/okay_database_build_last.log

echo "[构建] 等待构建锁 ..."
flock -o -x -w 2400 /tmp/okay_database_build.lock -c '
  cd /home/chenzongwei/rust_pyfunc
  export PATH="/home/chenzongwei/.cargo/bin:/home/chenzongwei/.local/bin:$PATH"
  export CONDA_PREFIX=/home/chenzongwei/.conda/envs/chenzongwei311
  echo "[构建] 拿到锁，开始构建"
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python lib/okay_database/gen_registry.py
  timeout 900 ./alter.sh > /tmp/okay_database_build_last.log 2>&1
  echo "alter.sh 退出码 $?"
'
echo "[构建] 结束，日志尾部："
grep -E "^error|error\[|Finished|💥|✅|退出码" "$LOG" | tail -20 || tail -20 "$LOG"
