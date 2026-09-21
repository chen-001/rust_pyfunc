#!/bin/bash
set -e
cd "$(dirname "$0")"
cat body.rs main.rs > gen.rs
~/.cargo/bin/rustc -O --edition 2021 -o mprob_rs gen.rs
