#!/bin/bash
# ── 快速开发构建（有优化，增量编译）──
#
# 用法:
#   bash dev.sh            → 默认 release-fast：有优化(opt-3)、无 LTO、增量编译、
#                            运行时性能 ≈ release（损失 <5%），日常开发/验证用
#   bash dev.sh --release  → 有优化但关闭 LTO，codegen 16（旧语义，验证性能用）
#
# 与 alter.sh 的区别:
#   alter.sh（默认）: 与 dev.sh 相同 profile（release-fast）+ 额外部署 worker 二进制
#   ./alter.sh release : fat LTO + codegen-units=1 → ~10 分钟（仅最终发布，需用户同意）
#
# 历史教训（2026-08-31）：旧版 dev.sh 无参 = maturin develop（opt-level 0），
# 数值内核比 release 慢 10~50 倍，曾把 513MB debug 构建覆盖到 release 安装，
# 导致回测整体变慢 40 倍。Cargo.toml [profile.dev] 已提升为 opt-3 作为兜底，
# 且 maturin 守卫（~/.local/bin/maturin）会直接拦截裸 maturin develop。
#
# dev.sh 不构建 worker 二进制（worker 只在 multiprocess 模式需要，开发阶段不需要）

__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    export PATH="/opt/anaconda3/bin:$PATH"
fi
conda activate chenzongwei311

export PATH="/home/chenzongwei/.local/bin:$PATH"

if [ "$1" == "--release" ]; then
    # 有优化但关闭 LTO：验证性能用（旧语义保留）
    /home/chenzongwei/.local/bin/mold -run maturin develop --release -- -C lto=no -C codegen-units=16 2>&1
else
    # 默认 release-fast：有优化 + 增量，运行时性能≈release（2026-09-01 起）
    /home/chenzongwei/.local/bin/mold -run maturin develop --profile release-fast 2>&1
fi
