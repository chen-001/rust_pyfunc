#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/chenzongwei/.cargo/bin:/home/chenzongwei/.conda/envs/chenzongwei311/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
export CARGO_TARGET_DIR=/home/chenzongwei/rust_pyfunc/target
"$PYO3_PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json
files = sorted(Path('src').rglob('*.rs')) + [Path('Cargo.toml'), Path('Cargo.lock')]
Path('build_sources.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, sort_keys=True))
PY
cargo build --offline --profile release-fast -j 12
"$PYO3_PYTHON" - <<'PY'
from pathlib import Path
import hashlib, json
files = sorted(Path('src').rglob('*.rs')) + [Path('Cargo.toml'), Path('Cargo.lock')]
sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
if sources != json.loads(Path('build_sources.json').read_text()):
    raise RuntimeError('Sources changed during compilation; rebuild before benchmarking')
binary = Path('/home/chenzongwei/rust_pyfunc/target/release-fast/librelaynet_perf_20261102.so')
Path('relaynet_perf_20261102.build.json.new').write_text(json.dumps(dict(sources=sources, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(), profile='release-fast'), sort_keys=True))
PY
cp "$CARGO_TARGET_DIR/release-fast/librelaynet_perf_20261102.so" ./relaynet_perf_20261102.so.new
mv ./relaynet_perf_20261102.so.new ./relaynet_perf_20261102.so
mv ./relaynet_perf_20261102.build.json.new ./relaynet_perf_20261102.build.json
