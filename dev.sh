#!/bin/bash
# ── 快速开发构建（无 LTO，增量编译）──
#
# 用法:
#   bash dev.sh           → dev 模式（无优化，最快增量编译），用于验证逻辑
#   bash dev.sh --release → 有优化但无 LTO，用于验证性能
#
# 与 alter.sh 的区别:
#   alter.sh: --release + LTO + codegen-units=1 → 9 分钟（仅用于最终发布）
#   dev.sh:   无 LTO + codegen-units=16        → 增量约 30 秒（日常开发）
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
    # 有优化但关闭 LTO：验证性能用
    /home/chenzongwei/.local/bin/mold -run maturin develop --release -- -C lto=no -C codegen-units=16 2>&1
else
    # dev 模式：opt-level=0，无 LTO，最快增量编译
    /home/chenzongwei/.local/bin/mold -run maturin develop 2>&1
fi
