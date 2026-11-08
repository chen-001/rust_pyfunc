#!/bin/bash
# 柜员游戏拓展1 v3 沙箱：独立 target 目录编译，产物放 package/，不安装、不影响主项目。
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
cargo build --release --lib --jobs 16 --target-dir "$(pwd)/target"
mkdir -p package
cp target/release/libanneal_regime_v3_sbx.so package/anneal_regime_v3_sbx.so
echo "built: $(pwd)/package/anneal_regime_v3_sbx.so"
