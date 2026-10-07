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

# 4. 选择构建 profile：
#      ./alter.sh            → 默认 release-fast（日常开发，编译快 5~10 倍）
#      ./alter.sh release    → 正式 release（fat LTO + codegen-units=1，极致性能，发布用）
PROFILE="${1:-release-fast}"
case "$PROFILE" in
  release|release-fast) ;;
  *) echo "❌ 未知 profile: $PROFILE，可选: release | release-fast"; exit 1 ;;
esac

# 5. 运行 maturin develop (使用 mold 链接器 + sccache 编译缓存加速)
#
# 为什么每一处都要判退出码：这个脚本原来没有 set -e，也没有判 maturin / cargo 的退出码。
# 一旦编译失败（典型是 PATH 里没有 ~/.cargo/bin，maturin 直接报 rustc not found），
# 脚本会继续往下走，执行 rm + cp 把 target 里**上一次编译留下的** worker 拷过去，
# 最后照样打印「✅ 已部署」。于是构建日志看着成功、实际上 worker 是旧的，
# 而 pipeline 级改动只在 worker 里生效（.so 更新了、worker 没更新），
# 结果是「明明改了却没生效」，非常难查。2026-10-06 已经在建自相关那次踩到。
# 结论：日志里那行「已部署」不能当成功证据，要看 .so 与 worker 的时间戳。
export PATH="/home/chenzongwei/.local/bin:$PATH"
if ! /home/chenzongwei/.local/bin/mold -run maturin develop --profile "$PROFILE"; then
  echo "❌ maturin develop 失败（profile=$PROFILE），已中断，不做 worker 部署。"
  echo "   常见原因：PATH 里没有 ~/.cargo/bin（rustc/cargo 找不到）。"
  exit 1
fi

# 6. 编译并部署 worker 二进制（多进程因子流水线 mode="multiprocess" 需要）
if ! /home/chenzongwei/.local/bin/mold -run cargo build --profile "$PROFILE" --bin rust_pyfunc_worker; then
  echo "❌ worker 二进制编译失败（profile=$PROFILE），已中断，不覆盖现有 worker。"
  exit 1
fi
# 先 rm 再 cp：避免 "Text file busy"——worker 进程可能正在运行，
# Linux 允许 unlink 正在执行的二进制（旧进程继续用旧 inode），新文件可正常写入。
if [ ! -f "target/$PROFILE/rust_pyfunc_worker" ]; then
  echo "❌ target/$PROFILE/rust_pyfunc_worker 不存在，拒绝部署（避免把陈旧 worker 当成新的）。"
  exit 1
fi
rm -f "python/rust_pyfunc/rust_pyfunc_worker"
cp "target/$PROFILE/rust_pyfunc_worker" "python/rust_pyfunc/rust_pyfunc_worker" || {
  echo "❌ worker 部署失败（cp 出错）"; exit 1;
}
echo "✅ worker 二进制（$PROFILE）已部署到 python/rust_pyfunc/rust_pyfunc_worker"
echo "   校验时间戳：$(date -r 'python/rust_pyfunc/rust_pyfunc_worker' '+%Y-%m-%d %H:%M:%S')"

# 如果需要，你也可以在脚本结束时停用环境
# conda deactivate