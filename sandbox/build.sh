#!/bin/bash
# ── 沙箱快速构建 ──
# 用法:
#   bash build.sh           → dev profile（无优化，最快，秒级）用于逻辑验证
#   bash build.sh --release → 有优化但无 LTO（约 10 秒）用于性能验证
#
# 构建后 Python 中可直接 import dev_sandbox

__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    export PATH="/opt/anaconda3/bin:$PATH"
fi
conda activate chenzongwei311

export PATH="/home/chenzongwei/.local/bin:$PATH"

if [ "$1" == "--release" ]; then
    /home/chenzongwei/.local/bin/mold -run maturin develop --release
else
    /home/chenzongwei/.local/bin/mold -run maturin develop
fi
