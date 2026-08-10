#!/usr/bin/env bash
# 沙箱构建: 编译 dev_sandbox_neutralize 并复制 .so 到 python 包目录
set -e
cd "$(dirname "$0")"
cargo build --release
cp target/release/libdev_sandbox_neutralize.so /home/chenzongwei/rust_pyfunc/python/rust_pyfunc/dev_sandbox_neutralize.so
echo "OK: built dev_sandbox_neutralize"
