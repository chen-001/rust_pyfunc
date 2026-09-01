#!/bin/bash
# ── maturin 守卫安装脚本（幂等）────────────────────────────────────────────
# 用途：把防事故守卫包装器安装到两处：
#   1. ~/.local/bin/maturin          （拦截"裸命令"调用）
#   2. $CONDA_PREFIX/bin/maturin     （拦截"绝对路径"调用）
# 原 maturin 会被备份为同目录 maturin.real（已存在则不覆盖）。
# 卸载/还原：rm maturin && mv maturin.real maturin
# 触发场景：重装 conda 环境、新电脑、或 conda 升级覆盖了守卫之后。
# 背景：2026-08-31 裸跑 `maturin develop`（opt-level 0 dev 构建）覆盖 release
#       安装，回测慢 40 倍。详情见 AGENTS.md「构建纪律」。
set -euo pipefail
cd "$(dirname "$0")"
GUARD_SRC="$(pwd)/maturin_guard.py"
CONDA_BIN="${CONDA_PREFIX:-/home/chenzongwei/.conda/envs/chenzongwei311}/bin"

install_one() {
  local bin_dir="$1"
  [ -d "$bin_dir" ] || { echo "跳过（目录不存在）: $bin_dir"; return 0; }
  if [ ! -f "$bin_dir/maturin.real" ]; then
    if [ -f "$bin_dir/maturin" ] && ! grep -q "MATERIAL: maturin-guard" "$bin_dir/maturin" 2>/dev/null        && ! grep -q "maturin 守卫包装器" "$bin_dir/maturin" 2>/dev/null; then
      cp "$bin_dir/maturin" "$bin_dir/maturin.real"
      echo "已备份原 maturin → $bin_dir/maturin.real"
    elif [ -f "$bin_dir/maturin" ]; then
      echo "跳过备份：$bin_dir/maturin 已是守卫（若有 maturin.real 则已含真 maturin）"
    else
      echo "警告：$bin_dir/maturin 不存在，仅安装守卫（无 maturin.real，需确保 PATH 中其它位置有真 maturin）"
    fi
  fi
  cp "$GUARD_SRC" "$bin_dir/maturin"
  chmod +x "$bin_dir/maturin"
  echo "已安装守卫: $bin_dir/maturin"
}

install_one "$HOME/.local/bin"
install_one "$CONDA_BIN"
echo "✅ maturin 守卫安装完成。当前 maturin 版本（应正常输出版本号）:"
"$CONDA_BIN/maturin" --version
