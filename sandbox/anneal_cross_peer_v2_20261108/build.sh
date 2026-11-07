#!/bin/bash
# 构建沙箱 .so 到 package/acp2_sandbox.so（不安装、不影响主项目）
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
cargo build --release --jobs 32 --target-dir "$(pwd)/target" --config "build.rustflags=['-C','target-cpu=native','-C','relocation-model=pic']"
mkdir -p package
cp target/release/libacp2_sandbox.so package/acp2_sandbox.so
echo "built: $(pwd)/package/acp2_sandbox.so"
