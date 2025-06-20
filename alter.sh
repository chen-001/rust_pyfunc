#!/bin/bash

# 1. 确保 Conda 已初始化到 Bash (如果尚未初始化)
# 这通常在第一次使用conda时设置，但对于非交互式脚本可能需要显式执行。
# 注意：你需要根据你的conda安装路径调整下面的路径。
# 常见的Miniconda或Anaconda安装会放在用户HOME目录下。
# 查找你的conda.sh文件：通常在 ~/miniconda3/etc/profile.d/conda.sh 或 ~/anaconda3/etc/profile.d/conda.sh
__conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/opt/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/opt/anaconda3/bin:$PATH"
    fi
fi

# 2. 激活你的 Conda 环境
conda activate chenzongwei311

# 3. 切换到你的项目目录 (如果当前目录不是)
# cd /path/to/your/maturin/project

# 4. 运行 maturin develop
maturin develop --release

# 如果需要，你也可以在脚本结束时停用环境
# conda deactivate