#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/home/chenzongwei/.cargo/bin:$PATH"
export PYO3_PYTHON=/home/chenzongwei/.conda/envs/chenzongwei311/bin/python
export RUSTC_WRAPPER=""
# Isolated library only: never install over rust_pyfunc or rebuild its worker.
cargo build --release --offline -j 8
cp target/release/libdirect_segments_sandbox.so direct_segments_sandbox.so.tmp
mv direct_segments_sandbox.so.tmp direct_segments_sandbox.so
