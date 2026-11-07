#!/bin/bash
# yupei_union_all 精简版沙箱：编译本目录 src/（即实测过的完整源码），不安装、不部署 worker。
# src/ 不入库；缺失时从主项目复制并打 optimization.patch（先核对被改文件的指纹没变）。
# 用法：bash build.sh   产物：package/rust_pyfunc/rust_pyfunc.abi3.so
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f src/lib.rs ]; then
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYBASE'
from pathlib import Path
import hashlib, json, shutil
root = Path('../..')
want = json.loads(Path('baseline_sources.json').read_text())
got = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in want}
bad = [f for f in want if got[f] != want[f]]
assert not bad, f'主项目这些文件在 f9d5de5 之后改过，补丁可能打不上: {bad}'
shutil.copytree(root / 'src', Path('src'))
for name in ('Cargo.toml', 'Cargo.lock'):
    shutil.copy2(root / name, Path(name))
PYBASE
  patch -p1 < optimization.patch
fi
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
# 独立 target 目录：不覆盖主项目 target/release-fast 里的产物
cargo build --lib --profile release-fast --jobs 16 --target-dir "$(pwd)/target"
mkdir -p package/rust_pyfunc
cp target/release-fast/librust_pyfunc.so package/rust_pyfunc/rust_pyfunc.abi3.so
printf 'from .rust_pyfunc import *\n' > package/rust_pyfunc/__init__.py
echo "built: $(pwd)/package/rust_pyfunc/rust_pyfunc.abi3.so"
