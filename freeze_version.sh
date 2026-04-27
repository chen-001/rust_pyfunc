#!/bin/bash
# freeze_version.sh — 将当前项目打包为版本锁定的独立 wheel
#
# 用法:
#   ./freeze_version.sh                    # 从 Cargo.toml 自动读取版本号
#   ./freeze_version.sh 0_76_8             # 手动指定版本标签
#
# 输出:
#   dist_versions/rust_pyfunc_<version>-<ver>-cp*-cp*-linux_x86_64.whl
#
# 安装后使用:
#   import rust_pyfunc_0_76_8 as rp

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── 1. 确定版本标签 ────────────────────────────────────────────
if [ $# -ge 1 ]; then
    VERSION_TAG="$1"
else
    # 从 Cargo.toml 读取版本号，把点替换为下划线
    VERSION_TAG=$(grep '^version = ' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '.' '_')
fi

PACKAGE_NAME="rust_pyfunc_${VERSION_TAG}"
TMPDIR="/tmp/build_${PACKAGE_NAME}_$$"
OUTPUT_DIR="${PROJECT_DIR}/dist_versions"
PYTHON_SRC="${PROJECT_DIR}/python/rust_pyfunc"

echo "========================================"
echo " 打包版本: ${PACKAGE_NAME}"
echo " 临时目录: ${TMPDIR}"
echo " 输出目录: ${OUTPUT_DIR}"
echo "========================================"

# ── 2. 创建临时构建目录 ────────────────────────────────────────
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

# ── 3. 复制 Python 源码，并重命名包目录 ────────────────────────
cp -r "$PYTHON_SRC" "${TMPDIR}/${PACKAGE_NAME}"

# ── 4. 修补 import 语句（绝对自导入 → 相对导入） ────────────────
#    这些文件用了 from rust_pyfunc.xxx import yyy（绝对导入），
#    包名改成 rust_pyfunc_0_76_8 后必须改为相对导入 from .xxx import yyy

cd "${TMPDIR}/${PACKAGE_NAME}"

echo "[patch] __init__.py ..."
# __init__.py 第 3 行: from rust_pyfunc.rust_pyfunc import *  →  from .rust_pyfunc import *
sed -i 's/^from rust_pyfunc\.rust_pyfunc import \*/from .rust_pyfunc import */' __init__.py
# __init__.py 第 5 行: from rust_pyfunc import *  →  删除（冗余）
sed -i '/^from rust_pyfunc import \*$/d' __init__.py
# __init__.py 第 6 行: from rust_pyfunc.rolling_future import ...  →  from .rolling_future import ...
sed -i 's/^from rust_pyfunc\.rolling_future import/from .rolling_future import/' __init__.py
# __init__.py 第 7 行: from rust_pyfunc.rolling_past import ...  →  from .rolling_past import ...
sed -i 's/^from rust_pyfunc\.rolling_past import/from .rolling_past import/' __init__.py

echo "[patch] rolling_future.py ..."
sed -i 's/^from rust_pyfunc\.rust_pyfunc import/from .rust_pyfunc import/' rolling_future.py

echo "[patch] rolling_past.py ..."
sed -i 's/^from rust_pyfunc\.rust_pyfunc import/from .rust_pyfunc import/' rolling_past.py

echo "[patch] treevisual.py ..."
sed -i 's/^from rust_pyfunc\.rust_pyfunc import/from .rust_pyfunc import/' treevisual.py

echo "[patch] pandas_corrwith.py ..."
sed -i 's/^from rust_pyfunc\.rust_pyfunc import/from .rust_pyfunc import/' pandas_corrwith.py

# 确认修补结果
echo ""
echo "修补后的导入语句："
grep -n '^from \.rust_pyfunc import' __init__.py rolling_future.py rolling_past.py treevisual.py pandas_corrwith.py 2>/dev/null || true
echo ""

# ── 5. 复制 Rust 源码 + 辅助文件 ────────────────────────────────
cp -r "${PROJECT_DIR}/src" "$TMPDIR/src"
cp "${PROJECT_DIR}/Cargo.toml" "$TMPDIR/Cargo.toml"
[ -f "${PROJECT_DIR}/Cargo.lock" ] && cp "${PROJECT_DIR}/Cargo.lock" "$TMPDIR/Cargo.lock" || true
[ -f "${PROJECT_DIR}/rust-toolchain.toml" ] && cp "${PROJECT_DIR}/rust-toolchain.toml" "$TMPDIR/" || true
cp "${PROJECT_DIR}/README.md" "$TMPDIR/README.md" 2>/dev/null || touch "$TMPDIR/README.md"

# ── 6. 创建临时的 pyproject.toml ──────────────────────────────
cat > "$TMPDIR/pyproject.toml" << PYPROJECT
[build-system]
requires = ["maturin>=1.3,<2.0"]
build-backend = "maturin"

[project]
name = "${PACKAGE_NAME}"
requires-python = ">=3.8"
dynamic = ["version"]

[tool.maturin]
features = ["pyo3/extension-module"]
python-source = "."
module-name = "${PACKAGE_NAME}.rust_pyfunc"
PYPROJECT

echo "pyproject.toml:"
cat "$TMPDIR/pyproject.toml"
echo ""

# ── 7. 构建 wheel ─────────────────────────────────────────────
cd "$TMPDIR"

echo "开始构建（当前 Python: $(python3 --version)）..."
echo ""

CARGO_TARGET_DIR="${PROJECT_DIR}/target" maturin build --release 2>&1

echo ""
echo "构建完成。"

# ── 8. 收集产物 ────────────────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cp "${PROJECT_DIR}"/target/wheels/"${PACKAGE_NAME}"*.whl "$OUTPUT_DIR/"

# ── 9. 清理 ────────────────────────────────────────────────────
rm -rf "$TMPDIR"

# ── 10. 输出结果 ───────────────────────────────────────────────
echo ""
echo "========================================"
echo " ✅ 打包完成"
echo "========================================"
ls -lh "${OUTPUT_DIR}/${PACKAGE_NAME}"*.whl 2>/dev/null || echo "（wheel 文件未找到）"

echo ""
echo "安装命令："
echo "  pip install ${OUTPUT_DIR}/${PACKAGE_NAME}-*.whl"
echo ""
echo "使用方式："
echo "  import ${PACKAGE_NAME} as rp"
echo "  rp.fast_rank(...)"
echo ""
