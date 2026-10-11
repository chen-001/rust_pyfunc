#!/bin/bash
# sandbox_yupei_pair 独立 crate 的统一构建入口（与主项目 alter.sh 纪律一致）。
# 用法: bash build.sh          → release 构建（默认，本 crate 只做性能计算）
#       bash build.sh --check  → 只类型检查不产二进制
cd "$(dirname "$0")"
if [ "$1" = "--check" ]; then
    cargo check --release
else
    cargo build --release
fi
