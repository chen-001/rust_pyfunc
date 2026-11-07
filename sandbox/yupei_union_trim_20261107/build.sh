#!/bin/bash
# yupei_union_all 精简版沙箱：编译本目录 src/（即实测过的完整源码），不安装、不部署 worker。
# src/ 不入库；缺失时从固定提交重建冻结源码，不依赖共享主目录的当前改动。
# 用法：bash build.sh [--release]。使用与冻结基线相同的优化 release-fast 配置。
# 产物：package/rust_pyfunc/rust_pyfunc.abi3.so；不安装到 Python 环境。
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -f src/lib.rs ]; then
  if [ -f frozen_source_base.json ]; then
    /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYFROZEN'
from pathlib import Path
import hashlib, io, json, os, subprocess, tarfile
meta = json.loads(Path('frozen_source_base.json').read_text())
patch = Path('frozen_baseline.patch').read_bytes()
assert hashlib.sha256(patch).hexdigest() == meta['patch_sha256'], '冻结补丁指纹不符'
env = dict(os.environ, HTTP_PROXY='http://127.0.0.1:10808', HTTPS_PROXY='http://127.0.0.1:10808')
archive = subprocess.check_output(['git', '-C', '../..', 'archive', meta['revision'],
                                   'src', 'Cargo.toml', 'Cargo.lock'], env=env)
with tarfile.open(fileobj=io.BytesIO(archive)) as source:
    for member in source.getmembers():
        if member.isfile():
            dest = Path(member.name)
            assert not dest.is_absolute() and '..' not in dest.parts
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source.extractfile(member).read())
subprocess.run(['patch', '-p1'], input=patch, check=True)
for name, digest in meta['cargo_hashes'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest, name
PYFROZEN
  else
  /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYBASE'
from pathlib import Path
import hashlib, json, shutil
root = Path('../..')
want = json.loads(Path('baseline_sources.json').read_text())
got = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in want}
bad = [f for f in want if got[f] != want[f]]
assert not bad, f'主项目这些文件在 f9d5de5 之后改过，补丁可能打不上: {bad}'
shutil.copytree(root / 'src', Path('src'))
Path('python/rust_pyfunc').mkdir(parents=True, exist_ok=True)
shutil.copy2(root / 'python/rust_pyfunc/__init__.pyi', 'python/rust_pyfunc/__init__.pyi')
for name in ('Cargo.toml', 'Cargo.lock'):
    shutil.copy2(root / name, Path(name))
PYBASE
  patch -p1 < optimization.patch
  # Preserve the exact pre-optimization sandbox snapshot, including dependencies
  # that already differed from the concurrently edited main checkout.
  if [ -f baseline_snapshot.patch ]; then
    patch -p1 < baseline_snapshot.patch
  fi
  fi
  if [ -f speed.patch ]; then
    /home/chenzongwei/.conda/envs/chenzongwei311/bin/python - <<'PYSPEED'
from pathlib import Path
import hashlib, json
want = json.loads(Path('speed_baseline_manifest.json').read_text())['source_hashes']
bad = [name for name, digest in want.items()
       if not Path(name).is_file() or hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest]
assert not bad, f'冻结的精简版源码不匹配，不能应用提速补丁: {bad}'
import shutil
if not Path('frozen/src').exists():
    Path('frozen').mkdir(exist_ok=True)
    shutil.copytree('src', 'frozen/src')
PYSPEED
    patch -p1 < speed.patch
  fi
fi
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
# 独立 target 目录：不覆盖主项目 target/release-fast 里的产物
cargo build --lib --profile release-fast --jobs 16 --target-dir "$(pwd)/target"
mkdir -p package/rust_pyfunc
cp target/release-fast/librust_pyfunc.so package/rust_pyfunc/rust_pyfunc.abi3.so
printf 'from .rust_pyfunc import *\n' > package/rust_pyfunc/__init__.py
echo "built: $(pwd)/package/rust_pyfunc/rust_pyfunc.abi3.so"
