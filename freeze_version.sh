#!/bin/bash
# freeze_version.sh — 将当前项目打包为版本锁定的独立 wheel
#
# 用法:
#   ./freeze_version.sh                                    # 从 Cargo.toml 自动读取版本号
#   ./freeze_version.sh 0_76_8                             # 手动指定版本标签
#   ./freeze_version.sh --manylinux2014                    # 构建 glibc 2.17+ 兼容 wheel
#   ./freeze_version.sh 0_76_8 --manylinux_2_31            # 构建 glibc 2.31+ 兼容 wheel
#
# 输出:
#   dist_versions/rust_pyfunc_<version>-<ver>-cp*-cp*-linux_x86_64.whl
#   加上 --manylinux* 参数后:
#   dist_versions/rust_pyfunc_<version>-<ver>-cp*-cp*-manylinux_*_x86_64.whl
#
# 安装后使用:
#   import rust_pyfunc_0_76_8 as rp

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# ── 0. 确保 zig 可用（用于 --manylinux* 交叉链接）─────────────
ZIG_BIN="$(python3 -c 'import ziglang, os; print(os.path.join(os.path.dirname(ziglang.__file__), "zig"))' 2>/dev/null || true)"
ZIG_DIR="${PROJECT_DIR}/.maturin_zig"
if [ -n "$ZIG_BIN" ] && [ -x "$ZIG_BIN" ]; then
    mkdir -p "${ZIG_DIR}/bin"
    ln -sf "$ZIG_BIN" "${ZIG_DIR}/bin/zig"
    # 创建 cargo 包装器：把 cargo → cargo-zigbuild
    # 注意：
    #   - CARGO_BUILD_TARGET 在包装器内部设置，这样 maturin 读不到带后缀的 triple
    #   - CARGO 指向真正的 cargo，避免 cargo-zigbuild 子调用又回到包装器形成死循环
    CARGO_ZIGBUILD="/home/chenzongwei/.conda/envs/chenzongwei311/bin/cargo-zigbuild"
    REAL_CARGO="$(command -v cargo)"
    cat > "${ZIG_DIR}/bin/cargo" << CARGO_WRAPPER
#!/bin/bash
export CARGO_BUILD_TARGET="x86_64-unknown-linux-gnu.2.31"
export CARGO="${REAL_CARGO}"
exec ${CARGO_ZIGBUILD} "\$@"
CARGO_WRAPPER
    chmod +x "${ZIG_DIR}/bin/cargo"
fi

# ── 1. 确定版本标签与兼容性参数 ────────────────────────────────
VERSION_TAG=""
COMPAT_TAG=""

for arg in "$@"; do
    case "$arg" in
        --manylinux*)
            COMPAT_TAG="${arg#--}"
            ;;
        *)
            VERSION_TAG="$arg"
            ;;
    esac
done

if [ -z "$VERSION_TAG" ]; then
    VERSION_TAG=$(grep '^version = ' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/' | tr '.' '_')
fi

PACKAGE_NAME="rust_pyfunc_${VERSION_TAG}"
TMPDIR="/tmp/build_${PACKAGE_NAME}_$$"
OUTPUT_DIR="${PROJECT_DIR}/dist_versions"
PYTHON_SRC="${PROJECT_DIR}/python/rust_pyfunc"

echo "========================================"
echo " 打包版本: ${PACKAGE_NAME}"
echo " 兼容目标: ${COMPAT_TAG:-系统原生 (glibc $(ldd --version 2>&1 | head -1 | grep -oP '\d+\.\d+$'))}"
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

build_for_python() {
    local python_bin="$1"
    local compat="$2"
    local label="$3"

    cd "$TMPDIR"

    local opts="--release --interpreter $python_bin"
    if [ -n "$compat" ]; then
        opts="$opts --compatibility $compat --no-default-features"
    fi

    echo "━━━ 构建 [$label] ━━━"
    echo "解释器: $($python_bin --version 2>&1)"
    echo "选项: maturin build $opts"
    echo ""

    # 需要 zig 做交叉链接
    if [ -n "$compat" ]; then
        # 用 cargo-zigbuild 替代 cargo —— 自动用 zig 编译和链接，
        # 产生目标 glibc 版本的二进制
        export PATH="${ZIG_DIR}/bin:$PATH"  # 内有 cargo 包装器（含 CARGO/CARGO_BUILD_TARGET 设置）
        echo "  [zig] 通过 cargo-zigbuild 构建 (glibc 2.31)"
    fi
    CARGO_TARGET_DIR="${PROJECT_DIR}/target" maturin build $opts 2>&1

    echo ""
    echo "[$label] 构建完成"
    echo ""
}

# 第一次构建：当前 Python，遵循 --manylinux* 参数
CURRENT_PYTHON=$(command -v python3)
build_for_python "$CURRENT_PYTHON" "$COMPAT_TAG" "当前环境"

# 第二次构建：Python 3.9 + manylinux_2_31
PY39="/opt/anaconda3/bin/python3.9"
if [ -x "$PY39" ]; then
    build_for_python "$PY39" "manylinux_2_31" "Python 3.9 + manylinux_2_31"
else
    echo "[跳过] python3.9 未找到: ${PY39}"
fi

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
